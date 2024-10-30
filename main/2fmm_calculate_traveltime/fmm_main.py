# fmm计算理论走时   (z,x,y)
import numpy as np
import heapq
import matplotlib.pyplot as plt
import pandas as pd
# 初始化理论走时模型
def initialize_travel_time(shape, source):


    travel_time = np.full(shape, np.inf)
    travel_time[source] = 0     # 将震源位置走时设为0
    return travel_time
 

# fmm计算理论走时
def fmm(velocity, source):

    shape = velocity.shape
    travel_time = initialize_travel_time(shape, source)

    # 优先队列（小根堆）
    pq = []  # 初始化优先队列，用于存储当前处理的前沿点。
    heapq.heappush(pq, (0, source)) # 将声源点（走时为0）添加到优先队列中。

    visited = np.zeros(shape, dtype=bool)   #

    # 行进方向 上下左右前后
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]   # 行进方向

    while pq:
        current_time, (i, j, k) = heapq.heappop(pq)     # 当前时间，初始化为震源位置的理论走时

        if visited[i, j, k]:
            continue

        visited[i, j, k] = True

        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk

            if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2] and not visited[ni, nj, nk]:
                dt = np.sqrt((di*60) ** 2 + (dj*150) ** 2 + (dk*150) ** 2) / velocity[ni, nj, nk]
                new_time = current_time + dt

                if new_time < travel_time[ni, nj, nk]:
                    travel_time[ni, nj, nk] = new_time
                    heapq.heappush(pq, (new_time, (ni, nj, nk)))

    return travel_time

## 主函数
if __name__ == '__main__':
    # 加载初始速度模型
    vel_model = np.load(file=r'C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\2fmm_calculate_traveltime\initial_model_5200.npy')  #       (z,y,x)
    # 加载传感器（校正到网格点上）
    grid_sensor = pd.read_excel(r'C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\grid_sensor.xlsx', header = None)

    np.set_printoptions(threshold=np.inf)  # 将数组完整输出

    print("变化之前：")
    print(grid_sensor)

    subtraction_values = [5600, 800, 850]

    grid_sensor = grid_sensor - subtraction_values
    print("减之后：")
    print(grid_sensor)

    scale_factors = [150, 150, 100]

    grid_sensor = grid_sensor / scale_factors
    print("缩小之后：")
    print(grid_sensor)

    print(vel_model)   
    print(grid_sensor.iloc[:][1])


    for i in range(1):
        source = np.array(grid_sensor.iloc[i][:]).astype(int)
        print(source)
        travel_time = fmm(vel_model, source)
        print(travel_time)

    # np.save('travel_time_initial_5200', arr=travel_time)   # save travel_time