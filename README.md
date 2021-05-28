BFS in CUDA
======

This is a CUDA implementation of BFS from the "Large Graph Algorithms for Massively Multithreaded Architectures" paper (Harish et al).

## How to Use

Compile with `$ bash compile.sh` (*sorry, no makefile*). This will create an executable called `./demo`.

Run it like so: `./demo /path/to/graph.txt <start-vertex-idx> (cpu | gpu)`.

Example: `./demo ./test-data/small.txt 0 gpu`.

On `i`-th line of stdout, it will write the distance between vertex `<start-vertex-idx>` and vertex `i`, or a very large number (> 4e9) if `i` isn't reachable from `<start-vertex-idx>`.

#### Graph Representation

```
<number-of-vertices> <number-of-edges> (directed | undirected)
<edge-start-vertex-index> <edge-end-vertex-index>
<edge-start-vertex-index> <edge-end-vertex-index>
...
<edge-start-vertex-index> <edge-end-vertex-index>
```

There's an example at `./test-data/small.txt`.

## Code Structure

* `demo.cpp` has `main()` and is the only executable utility.
* The only public header is `bfs.hpp`.
* The reference CPU version `bfsCPU()` is in `bfs_cpu.cpp`.
* The CUDA version `bfsCUDA()` is in `bfs.cu`.
* It uses
  * `bfs_kernels.cu`, which has `BFSKernel1()` and `BFSKernel2()` (Algorithms 2 and 3 in the paper);
  * `compaction.cu`, which has `compactSIMD()` (Section 3.2 in the paper);
  * `scan.cu` and `scan_kernels.cu`, which have, most notably, `prescanArray()` (an off-the-shelf implementation of scan operation).
