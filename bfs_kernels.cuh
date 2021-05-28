#include <assert.h>
#include "errchk.cuh"

#define BLOCK_SIZE 256

#define FALSE 0u
#define  TRUE 1u

__global__
void BFSKernel1(
    size_t graphSize, unsigned *activeMask, unsigned *V, unsigned *E,
    unsigned *F, unsigned *X, unsigned *C, unsigned *Fu);

__global__
void BFSKernel2(size_t graphSize, unsigned *F, unsigned *X, unsigned *Fu);

__global__
void getActiveMaskTemp(size_t graphSize, unsigned *F, unsigned *activeMask);
