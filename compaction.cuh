__host__
void preallocBlockSums(unsigned maxNumElements);

__host__
void prescanArray(unsigned *outArray, unsigned *inArray, int numElements);

__host__
void deallocBlockSums();

__global__
void compactSIMD(size_t N, unsigned *prefixSums, unsigned *activeMask, size_t blockSize);
