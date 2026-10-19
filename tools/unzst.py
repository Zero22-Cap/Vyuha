import sys, zstandard, time
src, dst = sys.argv[1], sys.argv[2]
t = time.time()
with open(src, 'rb') as fi, open(dst, 'wb') as fo:
    zstandard.ZstdDecompressor(max_window_size=2**31).copy_stream(fi, fo, read_size=1 << 24, write_size=1 << 24)
print('done', dst, round(time.time() - t, 1), 's', flush=True)
