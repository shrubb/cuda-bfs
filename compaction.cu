#include "scan.cu"

__global__
void compactSIMD(size_t N, unsigned *prefixSums, unsigned *activeMask, size_t blockSize) {
    size_t blockStart = blockIdx.x * blockSize;
    // Vertex assigned to current thread
    size_t v = blockStart + threadIdx.x;

    if (v < N) {
        // Can possibly be accelerated by using shared memory
        if (prefixSums[v+1] != prefixSums[v]) {
            activeMask[prefixSums[v]] = v;
        }
    }
}
