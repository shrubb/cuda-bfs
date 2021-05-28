#include "errchk.cuh"

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)

template <bool isNP2>
__device__ void loadSharedChunkFromMem(
        unsigned *s_data,
        const unsigned *idata,
        int n, int baseIndex,
        int& ai, int& bi,
        int& mem_ai, int& mem_bi,
        int& bankOffsetA, int& bankOffsetB) {

    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    s_data[ai + bankOffsetA] = idata[mem_ai];

    if (isNP2) {
        s_data[bi + bankOffsetB] = (bi < n) ? idata[mem_bi] : 0;
    } else {
        s_data[bi + bankOffsetB] = idata[mem_bi];
    }
}

template <bool isNP2>
__device__
void storeSharedChunkToMem(
        unsigned* odata,
        const unsigned* s_data,
        int n,
        int ai, int bi,
        int mem_ai, int mem_bi,
        int bankOffsetA, int bankOffsetB) {

    __syncthreads();

    odata[mem_ai] = s_data[ai + bankOffsetA];
    if (isNP2) {
        if (bi < n)
            odata[mem_bi] = s_data[bi + bankOffsetB];
    } else {
        odata[mem_bi] = s_data[bi + bankOffsetB];
    }
}

template <bool storeSum>
__device__
void clearLastElement(
        unsigned* s_data,
        unsigned *blockSums,
        int blockIndex) {

    if (threadIdx.x == 0) {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);

        if (storeSum) {
            blockSums[blockIndex] = s_data[index];
        }

        s_data[index] = 0;
    }
}

__device__
unsigned int buildSum(unsigned *s_data) {
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;

    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__
void scanRootToLeaves(unsigned *s_data, unsigned int stride) {
     unsigned int thid = threadIdx.x;

    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;

        __syncthreads();

        if (thid < d)
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            unsigned t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__
void prescanBlock(unsigned *data, int blockIndex, unsigned *blockSums) {
    int stride = buildSum(data);
    clearLastElement<storeSum>(data, blockSums,
                               (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves(data, stride);
}

template <bool storeSum, bool isNP2>
__global__
void prescan(
        unsigned *odata,
        const unsigned *idata,
        unsigned *blockSums,
        int n,
        int blockIndex,
        int baseIndex) {
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ unsigned s_data[];

    loadSharedChunkFromMem<isNP2>(s_data, idata, n,
                                  (baseIndex == 0) ?
                                  __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi,
                                  bankOffsetA, bankOffsetB);

    prescanBlock<storeSum>(s_data, blockIndex, blockSums);

    storeSharedChunkToMem<isNP2>(odata, s_data, n,
                                 ai, bi, mem_ai, mem_bi,
                                 bankOffsetA, bankOffsetB);
}

__global__
void uniformAdd(
        unsigned *data,
        unsigned *uniforms,
        int n,
        int blockOffset,
        int baseIndex) {

    __shared__ unsigned uni;
    if (threadIdx.x == 0)
        uni = uniforms[blockIdx.x + blockOffset];

    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

    __syncthreads();

    data[address]              += uni;
    data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}
