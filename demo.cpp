// Usage:
// ./demo /path/to/graph.txt <start-vertex-idx> (cpu | gpu)

#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>

#include "bfs.hpp"

class ScopedTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::string name;

public:
    ScopedTimer(const std::string & name = "time"):
        start_time(std::chrono::high_resolution_clock::now()),
        name(name) {}

    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cerr << name << ": " << duration << " ms" << std::endl;
    }
};

Graph loadGraphFromFile(std::string filePath) {
    std::ifstream inputFile(filePath);

    int n; inputFile >> n;
    Graph graph(n);
    int m; inputFile >> m;
    std::string isDirectedStr; inputFile >> isDirectedStr;
    assert(isDirectedStr == "directed" or isDirectedStr == "undirected");
    bool isDirected = isDirectedStr == "directed";

    for (int i = 0; i < m; ++i) {
        int a, b;
        inputFile >> a >> b;
        graph[a].push_back(b);
        if (not isDirected) graph[b].push_back(a);
    }

    return graph;
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    assert(argc == 4);

    Graph graph = loadGraphFromFile(argv[1]);
    const int sourceVertex = std::atoi(argv[2]);
    std::string processor = argv[3];
    assert(processor == "cpu" or processor == "gpu");

    std::cerr << "graph size: " << graph.size() << std::endl;

    std::vector<unsigned> distances;

    {
        ScopedTimer _("BFS on " + processor + " took");

        if (processor == "cpu") {
            bfsCPU(graph, sourceVertex, distances);
        } else {
            bfsCUDA(graph, sourceVertex, distances);
        }
    }

    for (int i = 0; i < distances.size(); ++i) {
        std::cout << distances[i] << std::endl;
    }

    return 0;
}
