import numpy as np
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df_source = pd.read_csv(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\source_relative.csv")
    df_sensor = pd.read_csv(r"C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\sensor.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_source["x"], df_source["y"], df_source["z"], s = 5, c = 'r', marker = '*')
    ax.scatter(df_sensor["X"], df_sensor['Y'], df_sensor['Z'], s = 20, c = 'b', marker = '^' )

    plt.show()


