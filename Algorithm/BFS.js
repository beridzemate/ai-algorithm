function bfs(graph, start) {
    let visited = new Set();
    let queue = [start];
    visited.add(start);

    while (queue.length > 0) {
        let node = queue.shift();
        console.log(node); // Process the node

        for (let neighbor of graph[node]) {
            if (!visited.has(neighbor)) {
                visited.add(neighbor);
                queue.push(neighbor);
            }
        }
    }
}
