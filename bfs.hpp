#include <vector>

typedef std::vector<std::vector<unsigned>> Graph;

void bfsCUDA(Graph & graph, unsigned sourceVertex, std::vector<unsigned> & distances);

void bfsCPU(Graph & graph, unsigned sourceVertex, std::vector<unsigned> & distances);
