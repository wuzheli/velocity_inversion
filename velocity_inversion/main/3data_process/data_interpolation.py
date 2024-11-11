import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata

if __name__ == "__main__":

    ## 读取震源数据
    df_source = pd.read_csv(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\source_relative.csv")
    
    df_source = df_source.iloc[:,2:]
    ## 显示缺失情况
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)

    df_isna = df_source.isna().mean()
    # 缺失比例
    print(df_isna)
    print(df_isna.sort_values())

    ## 震源数据插值
    # 创建三维网格，网格范围 X：5600~8300,Y：800~1700,Z：850～1150
    # 筛选数据
    df_source = df_source.drop(df_source[df_source['x'] < 5600].index)
    df_source = df_source.drop(df_source[df_source['x'] > 8000].index)

    df_source = df_source.drop(df_source[df_source['y'] < 800].index)
    df_source = df_source.drop(df_source[df_source['y'] > 1700].index)

    df_source = df_source.drop(df_source[df_source['z'] < 850].index)
    df_source = df_source.drop(df_source[df_source['z'] > 1150].index)


    # 最邻近值插值
    X = [5600, 5750, 5900, 6050, 6200, 6350, 6500, 6650, 6800, 6950, 7100, 7250, 7400, 7550, 7700, 7850, 8000]
    Y = [800, 950, 1100, 1250, 1400, 1550, 1700]
    Z = [850, 950, 1050, 1150]
    
    data_grid = []
    
    for i in X:
        for j in Y:
            for k in Z:
                data_grid.append([i, j, k])

    # 计算走时
    df_source['1'] = df_source['1'] - df_source['t0'] 
    df_source['2'] = df_source['2'] - df_source['t0'] 
    df_source['3'] = df_source['3'] - df_source['t0'] 
    df_source['4'] = df_source['4'] - df_source['t0'] 
    df_source['5'] = df_source['5'] - df_source['t0'] 
    df_source['6'] = df_source['6'] - df_source['t0'] 
    df_source['7'] = df_source['7'] - df_source['t0'] 
    df_source['8'] = df_source['8'] - df_source['t0'] 
    df_source['9'] = df_source['9'] - df_source['t0']
    df_source['10'] = df_source['10'] - df_source['t0'] 
    df_source['11'] = df_source['11'] - df_source['t0'] 
    df_source['12'] = df_source['12'] - df_source['t0'] 
    df_source['13'] = df_source['13'] - df_source['t0'] 
    df_source['14'] = df_source['14'] - df_source['t0'] 
    df_source['15'] = df_source['15'] - df_source['t0'] 
    df_source['16'] = df_source['16'] - df_source['t0'] 
    df_source['17'] = df_source['17'] - df_source['t0'] 
    df_source['18'] = df_source['18'] - df_source['t0'] 
    df_source['19'] = df_source['19'] - df_source['t0'] 
    df_source['20'] = df_source['20'] - df_source['t0'] 
    df_source['21'] = df_source['21'] - df_source['t0'] 
    df_source['22'] = df_source['22'] - df_source['t0'] 
    df_source['23'] = df_source['23'] - df_source['t0'] 
    df_source['24'] = df_source['24'] - df_source['t0'] 
    df_source['25'] = df_source['25'] - df_source['t0'] 
    df_source['26'] = df_source['26'] - df_source['t0']
    df_source['27'] = df_source['27'] - df_source['t0'] 
    df_source['28'] = df_source['28'] - df_source['t0']

    
    
    df_data = pd.DataFrame(data = data_grid, columns = ['x', 'y', 'z'] )

    print("震源节点数量：", len(df_data))

    for i in range(10):
        print(i)
        arrival_time = []
        x_grid = df_data.iloc[i,0]
        y_grid = df_data.iloc[i,1] 
        z_grid = df_data.iloc[i,2]
        for j in range(len(df_source)):  
            x_source = df_source.iloc[j]['x']
            y_source = df_source.iloc[j]['y']
            z_source = df_source.iloc[j]['z']            
            if x_grid - 150 <= x_source <= x_grid + 150 and y_grid - 150 <= y_source <= y_grid + 150 and z_grid - 150 <= z_source <= z_grid + 150:
                arrival_time.append(df_source.iloc[j][4:])

        if len(arrival_time) != 0:
            time_means = np.mean(arrival_time, axis=0)
            print("每个传感器平均走时：",time_means)






   
    
         


    
            



