"""Policy head on the frozen Vyuha accumulator.
q = Wp * [screlu(acc_stm), screlu(acc_nstm)] + bp   (32-d)
logit(move) = q . (Ef[pt,from] + Et[pt,to]) + bf[pt,from] + bt[pt,to]   (squares oriented to side to move)
Trained with softmax cross-entropy over legal quiet moves; label = the move Leela played.
"""
import argparse, math, os, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Net, batch_features, REC, HL, dev

D = 32
POL = np.dtype([('rec', REC), ('n', 'u1'), ('label', 'u1'), ('mv', '<u2', 64)])


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(2 * HL, D)
        self.Ef = nn.Embedding(6 * 64, D); self.Et = nn.Embedding(6 * 64, D)
        self.bf = nn.Embedding(6 * 64, 1); self.bt = nn.Embedding(6 * 64, 1)
        for e in (self.Ef, self.Et): nn.init.normal_(e.weight, std=0.1)
        for e in (self.bf, self.bt): nn.init.zeros_(e.weight)

    def forward(self, act, fi, ti, mask):
        q = self.q(act)                                   # B,D
        emb = self.Ef(fi) + self.Et(ti)                   # B,64,D
        lg = (emb * q[:, None, :]).sum(-1) + self.bf(fi).squeeze(-1) + self.bt(ti).squeeze(-1)
        return lg.masked_fill(~mask, -1e9)


def acts(net, i_s, i_n):
    with torch.no_grad():
        a = torch.clamp(net.ft(i_s) + net.ftb, 0, 1) ** 2
        b = torch.clamp(net.ft(i_n) + net.ftb, 0, 1) ** 2
        return torch.cat([a, b], 1)


def orient(sq, stm, kmir):
    s = sq ^ (stm[:, None] * 56)
    return torch.where(kmir[:, None], s ^ 7, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='C:/scdata/pol.bin')
    ap.add_argument('--value', default='E:/Chess/Vyuha/nets/n1-latest.pt')
    ap.add_argument('--out', default='E:/Chess/Vyuha/nets/policy.bin')
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--batch', type=int, default=8192)
    a = ap.parse_args()
    net = Net().to(dev); net.load_state_dict(torch.load(a.value)); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    pol = Policy().to(dev)
    opt = torch.optim.Adam(pol.parameters(), lr=2e-3)
    arr = np.fromfile(a.data, dtype=POL)
    n = len(arr); nval = 200000
    print('positions', n, flush=True)
    rec = arr['rec']
    sq = torch.from_numpy(np.ascontiguousarray(rec['sq'])).to(dev)
    pc = torch.from_numpy(np.ascontiguousarray(rec['pc'])).to(dev)
    st = torch.from_numpy(np.ascontiguousarray(rec['stm'])).to(dev)
    nm = torch.from_numpy(arr['n'].astype(np.int64)).to(dev)
    lb = torch.from_numpy(arr['label'].astype(np.int64)).to(dev)
    mv = torch.from_numpy(np.ascontiguousarray(arr['mv']).view(np.int16)).to(dev)
    del arr, rec
    steps = int((n - nval) * a.epochs / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.02, 0.5 * (1 + math.cos(math.pi * s / steps))))
    ar64 = torch.arange(64, device=dev)

    def batch(idx):
        s_, p_, t_ = sq[idx].long(), pc[idx].long(), st[idx].long()
        i_s, i_n, _ = batch_features(s_, p_, t_)
        act = acts(net, i_s, i_n)
        k = torch.gather(s_, 1, (p_ == (5 + 6 * t_[:, None])).int().argmax(1, keepdim=True)).squeeze(1)
        kmir = ((k ^ (t_ * 56)) & 7) >= 4
        m = mv[idx].long()
        fr, to, pt = m & 63, (m >> 6) & 63, (m >> 12) & 7
        fi = pt * 64 + orient(fr, t_, kmir); ti = pt * 64 + orient(to, t_, kmir)
        mask = ar64[None, :] < nm[idx][:, None]
        return act, fi.clamp(0, 383), ti.clamp(0, 383), mask, lb[idx]

    t0 = time.time(); step = 0
    perm = torch.randperm(n - nval, device=dev) + nval
    pi = 0
    while step < steps:
        if pi + a.batch > len(perm):
            perm = torch.randperm(n - nval, device=dev) + nval; pi = 0
        idx = perm[pi:pi + a.batch]; pi += a.batch
        act, fi, ti, mask, y = batch(idx)
        loss = F.cross_entropy(pol(act, fi, ti, mask), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                vi = torch.arange(0, nval, device=dev)
                accs = []
                for j in range(0, nval, 20000):
                    act, fi, ti, mask, y = batch(vi[j:j + 20000])
                    lg = pol(act, fi, ti, mask)
                    accs.append((lg.argmax(1) == y).float().mean().item())
                top1 = sum(accs) / len(accs)
            print(f'step {step}/{steps} loss {loss.item():.3f} val-top1 {top1:.3f} {time.time()-t0:.0f}s', flush=True)
    with torch.no_grad():
        Wp = pol.q.weight.t().contiguous().cpu().numpy().astype(np.float32)   # [2HL][D]
        bp = pol.q.bias.cpu().numpy().astype(np.float32)
        Ef = pol.Ef.weight.cpu().numpy().astype(np.float32); Et = pol.Et.weight.cpu().numpy().astype(np.float32)
        bf = pol.bf.weight.squeeze(1).cpu().numpy().astype(np.float32); bt = pol.bt.weight.squeeze(1).cpu().numpy().astype(np.float32)
    with open(a.out, 'wb') as f:
        f.write(b'SCP1'); f.write(struct.pack('<2i', HL, D))
        for x in (Wp, bp, Ef, Et, bf, bt): f.write(np.ascontiguousarray(x).tobytes())
    print('exported', a.out, flush=True)


if __name__ == '__main__':
    main()
