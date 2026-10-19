"""Split files of 72-byte training records into shards for train/train.py.

Usage: python split_records.py <in.bin>[,<in2.bin>,...] <out_prefix> [records_per_shard]
Writes <out_prefix>_000.bin, _001.bin, ... and an empty <out_prefix>.done marker (train.py then
uses every shard). Each input is cut to whole records: a file whose writer was stopped mid-record
ends with a partial record, and joining such files byte by byte would shift every following
record (this broke the overnight run of 2026-10-17).
"""
import os
import sys

REC = 72


def main():
    srcs, prefix = sys.argv[1].split(','), sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
    shard, in_shard, total = 0, 0, 0
    out = open('%s_%03d.bin' % (prefix, shard), 'wb')
    for src in srcs:
        n = os.path.getsize(src) // REC
        total += n
        with open(src, 'rb') as f:
            left = n
            while left:
                if in_shard == per:
                    out.close()
                    shard += 1
                    in_shard = 0
                    out = open('%s_%03d.bin' % (prefix, shard), 'wb')
                k = min(left, per - in_shard, 1 << 20)
                out.write(f.read(k * REC))
                in_shard += k
                left -= k
    out.close()
    open(prefix + '.done', 'w').close()
    print('%d records from %d files -> %d shards (%s_*.bin)' % (total, len(srcs), shard + 1, prefix))


if __name__ == '__main__':
    main()
