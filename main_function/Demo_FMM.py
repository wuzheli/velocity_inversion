import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_grid_3d(shape, source):
    """
    Initialize the 3D grid with infinite values and set the source to zero.
    """
    grid = np.full(shape, np.inf)
    grid[source] = 0
    return grid

def get_neighbors_3d(pos, shape):
    """
    Get the valid neighbors of a given position in the 3D grid.
    """
    neighbors = []
    for d in [(0, 1, 0), (1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]:
        neighbor = (pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])
        if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1] and 0 <= neighbor[2] < shape[2]:
            neighbors.append(neighbor)
    return neighbors

def update_distance_3d(grid, pos, neighbor):
    """
    Update the distance value of the neighbor using the current position in 3D.
    """
    new_distance = grid[pos] + 1  # Assuming uniform cost, modify as needed for non-uniform grids
    if new_distance < grid[neighbor]:
        grid[neighbor] = new_distance
        return True
    return False

def fast_marching_method_3d(shape, source):
    """
    Implement the Fast Marching Method for 3D grids.
    """
    grid = initialize_grid_3d(shape, source)
    heap = []
    heapq.heappush(heap, (0, source))
    visited = set()

    while heap:
        current_distance, current_pos = heapq.heappop(heap)
        if current_pos in visited:
            continue
        visited.add(current_pos)

        for neighbor in get_neighbors_3d(current_pos, shape):
            if neighbor not in visited:
                if update_distance_3d(grid, current_pos, neighbor):
                    heapq.heappush(heap, (grid[neighbor], neighbor))

    return grid

# Example usage
shape = (10, 10, 10)
source = (0, 0, 0)
distance_grid = fast_marching_method_3d(shape, source)

print("3D Distance grid:")
print(distance_grid)

# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prepare data for plotting
x, y, z = np.indices(distance_grid.shape)
x = x.flatten()
y = y.flatten()
z = z.flatten()
distances = distance_grid.flatten()

# Plotting the scatter plot with color based on distance values
scatter = ax.scatter(x, y, z, c=distances, cmap='viridis', marker='o')

# Setting labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Adding color bar
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('Distance')

# Show plot
plt.show()