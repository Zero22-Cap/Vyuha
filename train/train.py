"""Vyuha NNUE trainer (GPU).
Net: (10 king buckets x 768 -> HL) x 2 perspectives -> SCReLU -> 8 output buckets.
Heads: value (trained on SF score/result blend) + uncertainty (L1 on |target - value|, detached features).
"""
import argparse, glob, math, os, struct, time
import numpy as np
import torch
import torch.nn as nn

HL = 1024
KB = 10
NIN = KB * 768
PAD = NIN
QA, QB = 255, 64
SCALE = 180.0          # cp = net_output * SCALE ; cp ~ SF internal / 2
LAYOUT = [0, 1, 2, 3, 3, 2, 1, 0,
          4, 4, 5, 5, 5, 5, 4, 4,
          6, 6, 7, 7, 7, 7, 6, 6,
          8, 8, 8, 8, 8, 8, 8, 8] + [9] * 32
REC = np.dtype([('sq', 'u1', 32), ('pc', 'u1', 32), ('score', '<i2'), ('result', 'i1'),
                ('stm', 'u1'), ('move', '<u2'), ('ply', '<u2')])
dev = torch.device('cuda')
LAY = torch.tensor(LAYOUT, device=dev, dtype=torch.long)


def persp_idx(sq, pc, valid, ptype, pcol, ksq, p):
    """sq,pc: (B,32) long. ksq: (B,) king square of perspective p. p: (B,) 0/1."""
    flip = (p * 56).unsqueeze(1)
    kk = ksq ^ (p * 56)
    mir = (kk & 7) >= 4
    kk = torch.where(mir, kk ^ 7, kk)
    s = sq ^ flip
    s = torch.where(mir.unsqueeze(1), s ^ 7, s)
    rel = torch.where(pcol == p.unsqueeze(1), ptype, ptype + 6)
    idx = LAY[kk].unsqueeze(1) * 768 + rel * 64 + s
    return torch.where(valid, idx, torch.full_like(idx, PAD))


def batch_features(sq, pc, stm):
    sq = sq.long(); pc = pc.long(); stm = stm.long()
    valid = pc != 255
    pcl = torch.where(valid, pc, torch.zeros_like(pc))
    ptype = pcl % 6
    pcol = pcl // 6
    wk = torch.gather(sq, 1, (pc == 5).int().argmax(1, keepdim=True)).squeeze(1)
    bk = torch.gather(sq, 1, (pc == 11).int().argmax(1, keepdim=True)).squeeze(1)
    nstm = 1 - stm
    ks = torch.where(stm == 0, wk, bk)
    kn = torch.where(stm == 0, bk, wk)
    i_stm = persp_idx(sq, pc, valid, ptype, pcol, ks, stm)
    i_nstm = persp_idx(sq, pc, valid, ptype, pcol, kn, nstm)
    npc = valid.sum(1)
    bucket = ((npc - 2) // 4).clamp(0, 7)
    return i_stm, i_nstm, bucket


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ft = nn.EmbeddingBag(NIN + 1, HL, mode='sum', padding_idx=PAD)
        nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()
        self.ftb = nn.Parameter(torch.zeros(HL))
        self.out = nn.Linear(2 * HL, 8)
        self.unc = nn.Linear(2 * HL, 8)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.normal_(self.unc.weight, std=0.02)

    def forward(self, i_stm, i_nstm, bucket):
        a = torch.clamp(self.ft(i_stm) + self.ftb, 0, 1) ** 2
        b = torch.clamp(self.ft(i_nstm) + self.ftb, 0, 1) ** 2
        x = torch.cat([a, b], 1)
        v = self.out(x).gather(1, bucket[:, None]).squeeze(1)
        u = self.unc(x.detach()).gather(1, bucket[:, None]).squeeze(1)
        return v, u

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-1.98, 1.98)
            self.ft.weight[PAD].zero_()
            self.out.weight.clamp_(-1.98, 1.98)
            self.unc.weight.clamp_(-1.98, 1.98)


def export(net, path):
    with torch.no_grad():
        ftw = torch.round(net.ft.weight[:NIN] * QA).to(torch.int16).cpu().numpy()
        ftb = torch.round(net.ftb * QA).to(torch.int16).cpu().numpy()
        ow = torch.round(net.out.weight * QB).to(torch.int16).cpu().numpy()
        ob = torch.round(net.out.bias * QA * QB).to(torch.int32).cpu().numpy()
        uw = torch.round(net.unc.weight * QB).to(torch.int16).cpu().numpy()
        ub = torch.round(net.unc.bias * QA * QB).to(torch.int32).cpu().numpy()
    with open(path + '.tmp', 'wb') as f:
        f.write(b'SCN1')
        f.write(struct.pack('<4i', HL, KB, 8, 0))
        for arr in (ftw, ftb, ow, ob, uw, ub):
            f.write(np.ascontiguousarray(arr).tobytes())
    os.replace(path + '.tmp', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='C:/scdata/rec')
    ap.add_argument('--out', default='E:/Chess/Vyuha/nets')
    ap.add_argument('--total', type=float, default=2e9, help='positions to train on')
    ap.add_argument('--batch', type=int, default=16384)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam', type=float, default=0.8)
    ap.add_argument('--export_every', type=float, default=2e8)
    ap.add_argument('--resume', default='')
    ap.add_argument('--tag', default='n1')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    net = Net().to(dev)
    if a.resume:
        net.load_state_dict(torch.load(a.resume))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps_total = int(a.total // a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.01, 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total)))))
    seen, step, next_exp, shard_i = 0, 0, a.export_every, 0
    t0 = time.time(); lv_acc = lu_acc = 0.0; n_acc = 0
    while step < steps_total:
        shards = sorted(glob.glob(a.data + '_*.bin'))
        # skip a shard still being written: only use shards with a successor, or all when converter is done
        done = os.path.exists(a.data + '.done')
        usable = shards if done else shards[:-1]
        if not usable:
            time.sleep(5); continue
        path = usable[shard_i % len(usable)]; shard_i += 1
        arr = np.fromfile(path, dtype=REC)
        sq = torch.from_numpy(arr['sq']).to(dev)
        pc = torch.from_numpy(arr['pc']).to(dev)
        sc = torch.from_numpy(arr['score'].astype(np.float32)).to(dev)
        rs = torch.from_numpy(arr['result'].astype(np.float32)).to(dev)
        st = torch.from_numpy(arr['stm']).to(dev)
        n = len(arr); del arr
        perm = torch.randperm(n, device=dev)
        for bi in range(0, n - a.batch + 1, a.batch):
            idx = perm[bi:bi + a.batch]
            i_s, i_n, bk = batch_features(sq[idx], pc[idx], st[idx])
            v, u = net(i_s, i_n, bk)
            tgt_cp = sc[idx] / 2.0
            target = a.lam * torch.sigmoid(tgt_cp / SCALE) + (1 - a.lam) * (rs[idx] + 1) / 2
            lv = ((torch.sigmoid(v) - target).abs() ** 2.5).mean()
            ut = (tgt_cp / SCALE - v.detach()).abs()
            lu = (torch.relu(u) - ut).abs().mean()
            loss = lv + 0.05 * lu
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); net.clip()
            step += 1; seen += a.batch
            lv_acc += lv.item() if step % 50 == 0 else 0
            if step % 50 == 0:
                lu_acc += lu.item(); n_acc += 1
            if step % 2000 == 0:
                el = time.time() - t0
                print(f'step {step} seen {seen/1e6:.0f}M  lv {lv_acc/max(1,n_acc):.5f} lu {lu_acc/max(1,n_acc):.4f} '
                      f'lr {sched.get_last_lr()[0]:.2e} {seen/el/1e6:.2f}M pos/s  {el/60:.1f} min', flush=True)
                lv_acc = lu_acc = 0.0; n_acc = 0
            if seen >= next_exp or step >= steps_total:
                next_exp += a.export_every
                p = f'{a.out}/{a.tag}-{int(seen/1e6)}M.nnue'
                export(net, p); export(net, f'{a.out}/{a.tag}-latest.nnue')
                torch.save(net.state_dict(), f'{a.out}/{a.tag}-latest.pt')
                print('exported', p, flush=True)
            if step >= steps_total:
                break
        del sq, pc, sc, rs, st, perm
        torch.cuda.empty_cache()
    print('TRAINING DONE', flush=True)


if __name__ == '__main__':
    main()
