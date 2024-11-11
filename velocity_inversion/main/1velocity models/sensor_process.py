## 对传感器坐标进行处理，将其坐标转到最近的网格点
import pandas as pd
import numpy as np 
import math
import openpyxl
if __name__ == "__main__":
    df_sensor = pd.read_csv(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\sensor.csv")
    print(df_sensor)
    X = np.linspace(5600, 8000, 17)
    Y = np.linspace(800, 1700, 7)
    Z = np.linspace(850, 1150, 4)
    grid = [] 
    for i in X:
        for j in Y:
            for k in Z:
                grid.append([i,j,k])
    
    grid_sensor = []
    for i in range(len(df_sensor)):
        x_sensor = df_sensor.iloc[i][0]
        y_sensor = df_sensor.iloc[i][1]
        z_sensor = df_sensor.iloc[i][2]
        x = x_sensor
        y = y_sensor
        z = z_sensor
        x_distance = 10000
        y_distance = 10000
        z_distance = 10000
        for j in range(len(grid)):
            x_grid = grid[j][0]
            y_grid = grid[j][1]
            z_grid = grid[j][2]
            if abs(x_grid - x_sensor) <= x_distance:
                x_distance = abs(x_grid - x_sensor)
                x = x_grid
            if abs(y_grid - y_sensor) <= y_distance:
                y_distance = abs(y_grid - y_sensor)
                y = y_grid
            if abs(z_grid - z_sensor) <= z_distance:
                z_distance = abs(z_grid - z_sensor)
                z = z_grid
        grid_sensor.append([x, y, z])
    df_grid_sensor = pd.DataFrame(data = grid_sensor, columns = ['x', 'y', 'z'])
    df_grid_sensor.to_excel("grid_sensor.xlsx", index = False, header = False)
