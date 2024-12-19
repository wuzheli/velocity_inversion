import numpy as np
from matplotlib import pyplot as plt
if __name__ == '__main__':
    # 创建随机填充的3D矩阵
    time_model = np.load(r'C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\2fmm_calculate_traveltime\travel_time_initial_5200.npy').T # 矩阵转置成(x,y,z)

    fig = plt.figure()
    # 获取3D坐标轴
    ax = fig.add_subplot(111, projection='3d')

    # 插值补充




    # 绘制3D矩阵
    x, y, z = np.indices((time_model.shape[0], time_model.shape[1], time_model.shape[2]))
    time_show = ax.scatter(x, y, z, c=time_model.flatten(), cmap='viridis',marker='o')
    fig.colorbar(time_show )
    # 显示图形
    plt.show()


