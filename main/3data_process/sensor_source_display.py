import numpy as np
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    df_source = pd.read_csv("source_relative.csv")
    df_sensor = pd.read_csv("sensor.csv")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df_source["x"], df_source["y"], df_source["z"], s = 5, c = 'r', marker = '*')
    ax.scatter(df_sensor["X"], df_sensor['Y'], df_sensor['Z'], s = 20, c = 'b', marker = '^' )

    plt.show()


