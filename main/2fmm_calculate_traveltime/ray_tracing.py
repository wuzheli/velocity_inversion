# 路径追踪
import numpy as np

# 路径追踪算法
def trace_path(time_model, start, end):   # 输入理论走时模型，起点以及终点

    path = [end]  # 初始化路径
    current = end # 起点出发
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    # 动态规划
    while True:
        # 设定判定条件
        current_time = time_model[current[0]][current[1]][current[2]]   # 当前位置的理论走时
        for i, j, k in neighbors:
            ni, nj, nk = current[0] + i, current[1] + j, current[2] + k
            if time_model[ni][nj][nk] < next:
                i_min, j_min, k_min = ni, nj, nk                        # 周围最小的理论走时
        path.append([i_min, j_min, k_min])
        current = [i_min, j_min, k_min]

    return path

if __name__ == '__main__':
    # 加载理论走时模型
    time_model=np.load(file='travel_time1.npy')  # 10*12*26       (z,x,y)

    np.set_printoptions(threshold=np.inf)  # 将数组完整输出

    print(time_model)

    # 射线追踪
    source=(0,0,0)   # 定义震源位置              (z,y,x)
    destination=(9,11,25)

    path = trace_path(time_model, source, destination)

    print("Shortest path from source to destination:")
    for point in path:
        print(point)

