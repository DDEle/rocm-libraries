# (      2,      32,  256,  256, 32768,  8192),
# (      4,      32,  256,  256, 16384,  4096),
# (      8,      32,  256,  256,  8192,  2048),
# (      4,      32,  256,  256,  4096,  1024),  # small BATCH, small MEAN

bin/tile_example_fmha_bwd -mode=1 -deterministic=1 -v=0 -b=2 -h=32 \
	-s=$(python -c 'import random; random.seed(42); print(*[int(random.gauss(32768, 8192)) for _ in range(2)], sep=",")')
bin/tile_example_fmha_bwd -mode=1 -deterministic=1 -v=0 -b=4 -h=32 \
	-s=$(python -c 'import random; random.seed(42); print(*[int(random.gauss(16384, 4096)) for _ in range(4)], sep=",")')
bin/tile_example_fmha_bwd -mode=1 -deterministic=1 -v=0 -b=8 -h=32 \
	-s=$(python -c 'import random; random.seed(42); print(*[int(random.gauss(8192, 2048)) for _ in range(8)], sep=",")')
bin/tile_example_fmha_bwd -mode=1 -deterministic=1 -v=0 -b=4 -h=32 \
	-s=$(python -c 'import random; random.seed(42); print(*[int(random.gauss(4096, 1024)) for _ in range(4)], sep=",")')

