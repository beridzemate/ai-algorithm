function dijkstra(graph, start) {
    let distances = {};
    let visited = new Set();
    let nodes = Object.keys(graph);

    for (let node of nodes) {
        distances[node] = Infinity;
    }
    distances[start] = 0;

    while (nodes.length > 0) {
        let minNode = nodes.reduce((min, node) => distances[node] < distances[min] ? node : min, nodes[0]);
        nodes = nodes.filter(node => node !== minNode);
        visited.add(minNode);

        for (let neighbor in graph[minNode]) {
            if (!visited.has(neighbor)) {
                let newDist = distances[minNode] + graph[minNode][neighbor];
                if (newDist < distances[neighbor]) {
                    distances[neighbor] = newDist;
                }
            }
        }
    }

    return distances;
}
