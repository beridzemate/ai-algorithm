function dfs(graph, start, visited = new Set()) {
    visited.add(start);
    console.log(start); // Process the node

    for (let neighbor of graph[start]) {
        if (!visited.has(neighbor)) {
            dfs(graph, neighbor, visited);
        }
    }
}
