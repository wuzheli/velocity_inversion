import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    #1 读取震源数据
    df_source = pd.read_csv(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\source_relative.csv")
    
    df_source = df_source.iloc[:,2:]
    
    #2 筛选数据范围

    df_source = df_source[(df_source['x'] >= 5600) & (df_source['x'] <= 8000) & (df_source['y'] >= 800) & (df_source['y'] <= 1700) & (df_source['z'] >= 850) & (df_source['z'] <= 1150)]

    #3 计算走时
    for col in range(1, 29):
        if str(col) in df_source.columns:
            df_source[str(col)] = df_source[str(col)] - df_source['t0']

    interpolation_data = []
    for col in range(1, 29):
        if str(col) in df_source.columns:
            #4 筛选无缺失值和走时>0的数据
            df_data = df_source[['x', 'y', 'z', str(col)]].dropna()
            df_data = df_data[df_data[str(col)] > 0]
            points = df_data[['x', 'y', 'z']].values
            t = df_data[str(col)].values
            #5 定义插值网格
            grid_x, grid_y, grid_z = np.mgrid[5600:8000:17j, 800:1700:7j, 850:1150:4j]
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()
            grid_z = grid_z.flatten()
            grid_points = np.array([grid_x, grid_y, grid_z]).T
            
            #6 进行三维插值
            kernel = RBF() + WhiteKernel(noise_level=1e-04, noise_level_bounds=(1e-08, 1e-1))
            gpr = GaussianProcessRegressor(kernel=kernel, optimizer='fmin_l_bfgs_b', random_state=0).fit(points, t)
            grid_t, _ = gpr.predict(grid_points, return_std=True)

            # #7 可视化插值结果
            # fig, axes = plt.subplots(1, 2, figsize=(24, 8), subplot_kw={'projection': '3d'})

            # # 绘制原始数据点
            # ax_left = axes[0]
            # scatter_show_left = ax_left.scatter(points[:, 0], points[:, 1], points[:, 2], c=t, vmin=0, vmax=1, label='Original Data Points')
            # ax_left.set_title('3D Visualization of Source Data ' + 'Sensor ' + str(col))

            # # 绘制原始数据点
            # ax_right = axes[1]
            # grid_show = ax_right.scatter(grid_x, grid_y, grid_z, c=grid_t, cmap='viridis', vmin=0, vmax=1, label='Interpolated Data Points')
            # plt.colorbar(grid_show, ax=ax_right)
            # ax_right.set_title('3D Visualization of Interpolated Data ' + 'Sensor ' + str(col))
            # plt.show()

        interpolation_data.append(grid_t)
    df_interpolation = pd.DataFrame(data = interpolation_data)
    df_interpolation.to_excel(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\interpolation\interpolation_data.xlsx")
    



