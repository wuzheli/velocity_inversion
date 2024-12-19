# fmm计算理论走时   (z,x,y)
import numpy as np
import heapq
import matplotlib.pyplot as plt
import pandas as pd
import os
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

    visited = np.zeros(shape, dtype = bool)   #

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
                dt = np.sqrt((di*100) ** 2 + (dj*150) ** 2 + (dk*150) ** 2) / velocity[ni, nj, nk]
                new_time = current_time + dt

                if new_time < travel_time[ni, nj, nk]:
                    travel_time[ni, nj, nk] = new_time
                    heapq.heappush(pq, (new_time, (ni, nj, nk)))

    return travel_time

## 主函数
if __name__ == '__main__':
    # 加载传感器（校正到网格点上）
    grid_sensor = pd.read_excel(r'C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\grid_sensor.xlsx', header = None)

    np.set_printoptions(threshold=np.inf)  # 将数组完整输出

    # 处理传感器数据
    print("变化之前：",grid_sensor)

    subtraction_values = [5600, 800, 850]

    grid_sensor = grid_sensor - subtraction_values
    print("减之后：",grid_sensor)

    scale_factors = [150, 150, 100]

    grid_sensor = grid_sensor / scale_factors
    print("缩小之后：",grid_sensor)

    print("传感器坐标的第二列：", grid_sensor.iloc[:][1])

    grid_sensor.iloc[:,[0,2]] = grid_sensor.iloc[:,[2,0]]

    print("传感器位置：")
    print(grid_sensor)
    save_path = 'C:\\Users\\建设村彭于晏\\Desktop\\python程序\\velocity_inversion\\main\\2fmm_calculate_traveltime\\theoretical_traveltime\\'
    # 加载初始速度模型 
    for i in range(1, 201):
        print("第", i, "个速度模型：")
        vel_model = np.load('C:\\Users\\建设村彭于晏\\Desktop\\python程序\\velocity_inversion\\main\\1velocity_models\\theoretical_model\\vel_model_' + str(i) + '.npy')
        # 计算28个传感器的理论走时
        theoretical_time = []
        for j in range(len(grid_sensor)):
            print("第", j, "个传感器：")
            z = np.array(grid_sensor.iloc[j,0]).astype(int)
            y = np.array(grid_sensor.iloc[j,1]).astype(int)
            x = np.array(grid_sensor.iloc[j,2]).astype(int)
            grid_source = (z, y, x)
            travel_time = fmm(velocity = vel_model, source = grid_source)
            file_name = os.path.join(save_path, 'traveltime_' + str(i) + '_' + str(j+1) + '.npy')
            np.save(file_name, arr=travel_time)
            travel_time = travel_time.T.reshape(-1)
            theoretical_time.append(travel_time)
        print("传感器数据为：", theoretical_time)
        df_data = pd.DataFrame(data = theoretical_time)

        file_name = os.path.join(save_path, 'traveltime_' + str(i) + '.xlsx')

        df_data.to_excel(file_name, index  = False, header = False)