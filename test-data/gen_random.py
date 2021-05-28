# Usage:
# `python3 gen_random.py <num-vertices> <num_edges>`

import sys
import random

random.seed(123)

v, e = map(int, sys.argv[1:])

print(f"{v} {v + e} undirected")
for i in range(v):
    edge_start = i
    edge_end   = (i + 1) % v
    print(f"{edge_start} {edge_end}")

for _ in range(e):
    edge_start = random.randint(0, v - 1)
    edge_end   = random.randint(0, v - 1)
    print(f"{edge_start} {edge_end}")
