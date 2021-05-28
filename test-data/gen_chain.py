# Usage:
# `python3 gen_random.py <num-vertices> <num_edges_between_consecutive_vertices>`

import sys
import random

random.seed(123)

v, edges_between_vertices = map(int, sys.argv[1:])

print(f"{v} {v * edges_between_vertices} undirected")
for i in range(v):
    edge_start = i
    edge_end   = (i + 1) % v
    print(f"{edge_start} {edge_end}\n" * edges_between_vertices, end="")
