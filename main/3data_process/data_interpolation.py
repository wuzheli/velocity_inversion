import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
form scipy.interpolate import 

if __name__ == "__main__":

    ## 读取震源数据
    df_source = pd.read_csv("source_relative.csv")

    ## 显示缺失情况
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)

    df_isna = df_source.isna().mean()

    print(df_isna)
    print(df_isna.sort_values())

    ## 震源数据插值
    # 创建三维网格，网格范围 X：5600~8300,Y：800~1700,Z：850～1150
    df_source = df_source.drop(df_source[df_source['x'] < 5600].index)
    df_source = df_source.drop(df_source[df_source['x'] > 8300].index)

    df_source = df_source.drop(df_source[df_source['y'] < 800].index)
    df_source = df_source.drop(df_source[df_source['y'] > 1700].index)

    df_source = df_source.drop(df_source[df_source['z'] < 850].index)
    df_source = df_source.drop(df_source[df_source['z'] > 1150].index)

    X = np.linspace(5600,8300,19)
    Y = np.linspace(800,1700,7)
    Z = np.linspace(850,1150,6)

    data_grid = []
    for i in X:
        for j in Y:
            for k in Z:
                data_grid.append([i,j,k])
    df_data = pd.DataFrame(data = data_grid, columns = ['x', 'y', 'z'])

    # 三线性插值
    for i in range():




    # 邻近值插值
    f1 = inter




