#include <fstream>
#include <queue>
#include <limits>

#include "bfs.hpp"

void bfsCPU(Graph & graph, unsigned sourceVertex, std::vector<unsigned> & distances) {
    distances.resize(graph.size(), std::numeric_limits<unsigned>::max());

    distances[sourceVertex] = 0;

    std::queue<unsigned> q;
    q.push(sourceVertex);

    while (!q.empty()) {
        unsigned curr = q.front();
        q.pop();

        for (int i = 0; i < graph[curr].size(); ++i) {
            unsigned next = graph[curr][i];
            if (distances[next] == std::numeric_limits<unsigned>::max()) {
                q.push(next);
                distances[next] = distances[curr] + 1;
            }
        }
    }
}
