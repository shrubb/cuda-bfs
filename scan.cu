#include "scan_kernels.cu"

inline
bool isPowerOfTwo(int n) {
    return (n & (n-1)) == 0;
}

inline
int floorPow2(int n) {
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
}

#define BLOCK_SIZE 256

unsigned **scanBlockSums;
unsigned numEltsAllocated = 0;
unsigned numLevelsAllocated = 0;

__host__
void preallocBlockSums(unsigned maxNumElements) {
    numEltsAllocated = maxNumElements;

    unsigned blockSize = BLOCK_SIZE;
    unsigned numElts = maxNumElements;

    int level = 0;

    do {
        unsigned numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    scanBlockSums = (unsigned**) malloc(level * sizeof(unsigned*));
    numLevelsAllocated = level;

    numElts = maxNumElements;
    level = 0;

    do {
        unsigned numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) {
            gpuErrchk(cudaMalloc(&scanBlockSums[level++], numBlocks * sizeof(unsigned)));
        }
        numElts = numBlocks;
    } while (numElts > 1);
}

__host__
void deallocBlockSums() {
    for (unsigned i = 0; i < numLevelsAllocated; i++) {
        cudaFree(scanBlockSums[i]);
    }

    free(scanBlockSums);

    scanBlockSums = 0;
    numEltsAllocated = 0;
    numLevelsAllocated = 0;
}

__host__
void prescanArrayRecursive(
        unsigned *outArray,
        const unsigned *inArray,
        int numElements,
        int level) {

    unsigned blockSize = BLOCK_SIZE;
    unsigned numBlocks =
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned numEltsPerBlock = numThreads * 2;

    unsigned numEltsLastBlock =
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned np2LastBlock = 0;
    unsigned sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock) {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);

        unsigned extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock =
            sizeof(unsigned) * (2 * numThreadsLastBlock + extraSpace);
    }

    // Avoid shared memory bank conflicts
    unsigned extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned sharedMemSize =
        sizeof(unsigned) * (numEltsPerBlock + extraSpace);

    dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3 threads(numThreads, 1, 1);

    // Main action

    if (numBlocks > 1) {
        prescan<true, false> <<< grid, threads, sharedMemSize >>> (
            outArray, inArray, scanBlockSums[level], numThreads * 2, 0, 0);

        if (np2LastBlock) {
            prescan<true, true> <<< 1, numThreadsLastBlock, sharedMemLastBlock >>> (
                outArray, inArray, scanBlockSums[level], numEltsLastBlock,
                numBlocks - 1, numElements - numEltsLastBlock);
        }

        prescanArrayRecursive(scanBlockSums[level], scanBlockSums[level], numBlocks, level+1);

        uniformAdd <<< grid, threads >>> (
            outArray, scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);

        if (np2LastBlock) {
            uniformAdd <<<1, numThreadsLastBlock>>> (
                outArray, scanBlockSums[level], numEltsLastBlock,
                numBlocks - 1, numElements - numEltsLastBlock);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescan<false, false> <<<grid, threads, sharedMemSize>>> (
            outArray, inArray, 0, numThreads * 2, 0, 0);
    } else {
         prescan<false, true> <<<grid, threads, sharedMemSize>>> (
            outArray, inArray, 0, numElements, 0, 0);
    }
}

__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements) {
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}
