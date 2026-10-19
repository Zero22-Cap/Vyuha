"""Mix files of 72-byte training records into shards that each hold an equal slice of every input.

Usage: python mix_records.py <in.bin>[,<in2.bin>,...] <out_prefix> [shards]
train/train.py permutes records only inside a shard and visits the shards in order, so shards cut
straight from consecutive generation files would feed one generation at a time (the last ones at the
lowest learning rate). Here shard k holds part k of every input. Inputs are cut to whole records.
Writes <out_prefix>_000.bin ... and an empty <out_prefix>.done marker.
"""
import os
import sys

REC = 72
CHUNK = 1 << 20  # records per read


def main():
    srcs, prefix = sys.argv[1].split(','), sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sizes = [os.path.getsize(s) // REC for s in srcs]
    total = 0
    for i in range(k):
        with open('%s_%03d.bin' % (prefix, i), 'wb') as out:
            for src, n in zip(srcs, sizes):
                a, b = n * i // k, n * (i + 1) // k
                with open(src, 'rb') as f:
                    f.seek(a * REC)
                    left = b - a
                    while left:
                        m = min(left, CHUNK)
                        out.write(f.read(m * REC))
                        left -= m
                total += b - a
    open(prefix + '.done', 'w').close()
    print('%d records from %d files -> %d mixed shards (%s_*.bin)' % (total, len(srcs), k, prefix))


if __name__ == '__main__':
    main()
