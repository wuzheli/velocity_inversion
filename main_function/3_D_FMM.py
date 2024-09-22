import numpy as np
import heapq

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

'''
代码解释：
初始化3D网格： initialize_grid_3d 函数创建一个形状为 shape 的3D网格，并将所有值初始化为无穷大 (np.inf)，同时将源点的值设置为0。

获取邻居： get_neighbors_3d 函数返回给定位置的有效邻居（六个方向：上下左右前后）。

更新距离： update_distance_3d 函数使用当前位置更新邻居的距离值，如果新距离更小，则更新邻居的距离。

快速行进方法： fast_marching_method_3d 函数实现FMM算法，使用最小堆 (heapq) 来存储和处理当前前沿的点。它从源点开始，不断扩展前沿并更新距离，直到处理完所有点。

使用示例：
我们定义了一个形状为 (10, 10, 10) 的3D网格，并以 (0, 0, 0) 作为源点，调用 fast_marching_method_3d 函数来计算距离网格。最终打印出结果。

可以根据具体应用需求进一步修改此代码，如调整更新距离的方式以适应非均匀网格的情况。
'''
