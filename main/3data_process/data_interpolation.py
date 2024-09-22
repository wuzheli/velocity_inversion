import numpy as np
import pandas as pd
import numpy as py
import matplotlib.pyplot as plt


if __name__ == "__main__":


    df_source = pd.read_csv("source_relative.csv")
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    print(df_source.head())
    print(df_source.isna().mean())

