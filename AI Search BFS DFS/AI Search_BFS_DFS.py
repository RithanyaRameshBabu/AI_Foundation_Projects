

from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# 0 = free path, 1 = wall
maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

start = (0, 0)
goal = (4, 4)

rows, cols = len(maze), len(maze[0])

def get_neighbors(node):
    x, y = node
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def bfs(start, goal):
    queue = deque([start])
    visited = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    path = []
    while goal:
        path.append(goal)
        goal = visited[goal]
    return path[::-1]

def dfs(start, goal):
    stack = [start]
    visited = {start: None}

    while stack:
        current = stack.pop()
        if current == goal:
            break
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited[neighbor] = current
                stack.append(neighbor)

    path = []
    while goal:
        path.append(goal)
        goal = visited[goal]
    return path[::-1]

bfs_path = bfs(start, goal)
dfs_path = dfs(start, goal)

print("BFS Path:", bfs_path)
print("DFS Path:", dfs_path)

# -----------------------------
# Plotting the maze and paths
# -----------------------------
maze_array = np.array(maze)
plt.figure(figsize=(6,6))
plt.imshow(maze_array, cmap='Greys', origin='upper')

# Plot BFS path
if bfs_path:
    bx, by = zip(*bfs_path)
    plt.plot(by, bx, color='blue', marker='o', label='BFS Path')

# Plot DFS path
if dfs_path:
    dx, dy = zip(*dfs_path)
    plt.plot(dy, dx, color='red', marker='x', label='DFS Path')

# Start and Goal
plt.scatter(start[1], start[0], color='green', s=100, label='Start')
plt.scatter(goal[1], goal[0], color='purple', s=100, label='Goal')

plt.title('Maze BFS and DFS Paths')
plt.legend()
plt.show()