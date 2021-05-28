# Usage:
# `python3 gen_tree.py <num-levels>`

import sys

num_levels = int(sys.argv[1])

print(f"{2 ** num_levels - 1} {2 ** num_levels - 2} directed")

vertex_idx = 0
for level in range(num_levels - 1):
    for vertex in range(2 ** level - 1, 2 ** (level + 1) - 1):
        edge_start = vertex

        edge_end   = 2 * vertex + 1
        print(f"{edge_start} {edge_end}")

        edge_end   = 2 * vertex + 2
        print(f"{edge_start} {edge_end}")
