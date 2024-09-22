# fmm计算理论走时   (z,x,y)
import numpy as np
import heapq
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    vel_model=np.load(file='vel_model_initial1.npy')  # 10*12*26       (z,y,x)

    np.set_printoptions(threshold=np.inf)  # 将数组完整输出

    print(vel_model)

    source=(0,0,0)   # 定义震源位置              (z,y,x)

    travel_time = fmm(vel_model, source)

    np.save('travel_time1', arr=travel_time)   # save travel_time

    print(travel_time)