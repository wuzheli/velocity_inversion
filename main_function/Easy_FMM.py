# 2-DFMM
import numpy as np
import heapq

def initialize_grid(shape, source):
    """
    Initialize the grid with infinite values and set the source to zero.
    """
    grid = np.full(shape, np.inf)
    grid[source] = 0
    return grid

def get_neighbors(pos, shape):
    """
    Get the valid neighbors of a given position in the grid.
    """
    neighbors = []
    for d in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        neighbor = (pos[0] + d[0], pos[1] + d[1])
        if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
            neighbors.append(neighbor)
    return neighbors

def update_distance(grid, pos, neighbor):
    """
    Update the distance value of the neighbor using the current position.
    """
    new_distance = grid[pos] + 1  # Assuming uniform cost, modify as needed for non-uniform grids
    if new_distance < grid[neighbor]:
        grid[neighbor] = new_distance
        return True
    return False

def fast_marching_method(shape, source):
    """
    Implement the Fast Marching Method.
    """
    grid = initialize_grid(shape, source)
    heap = []
    heapq.heappush(heap, (0, source))
    visited = set()

    while heap:
        current_distance, current_pos = heapq.heappop(heap)
        if current_pos in visited:
            continue
        visited.add(current_pos)

        for neighbor in get_neighbors(current_pos, shape):
            if neighbor not in visited:
                if update_distance(grid, current_pos, neighbor):
                    heapq.heappush(heap, (grid[neighbor], neighbor))

    return grid

# Example usage
shape = (20, 30)
source = (10, 15)
distance_grid = fast_marching_method(shape, source)

print("Distance grid:")
print(distance_grid)

'''
初始化网格： initialize_grid 函数创建一个形状为 shape 的网格，并将所有值初始化为无穷大 (np.inf)，同时将源点的值设置为0。

获取邻居： get_neighbors 函数返回给定位置的有效邻居（上下左右四个方向）。

更新距离： update_distance 函数使用当前位置更新邻居的距离值，如果新距离更小，则更新邻居的距离。

快速行进方法： fast_marching_method 函数实现FMM算法，使用最小堆 (heapq) 来存储和处理当前前沿的点。它从源点开始，不断扩展前沿并更新距离，直到处理完所有点。

使用示例：
我们定义了一个形状为 (10, 10) 的网格，并以 (0, 0) 作为源点，调用 fast_marching_method 函数来计算距离网格。最终打印出结果。

您可以根据具体应用需求调整此代码，比如修改更新距离的方式以适应非均匀网格的情况。
'''