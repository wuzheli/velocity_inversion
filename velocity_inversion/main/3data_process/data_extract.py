import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 打开文件，对数据进行处理，划分事件
    file = open(r'C:\Users\建设村彭于晏\Desktop\python程序\velocity_inversion\main\3data_process\data.txt')

    data = file.readlines()
    index = 0
    ans = []
    temp = []
    while index < len(data):
        if data[index][0] == "e":
            ans.append(temp)
            data_split = data[index].split('\t' )
            temp = []
            temp.append(data_split)
        else:
            data_split = data[index].split('\t' )
            temp.append(data_split)
        index += 1
    ans.append(temp)
    source = []
    for i in range(1, len(ans)):

        temp = [None] * 33
        temp[0] = int(ans[i][0][1])
        temp[1] = float(ans[i][0][3]) - 2990000
        temp[2] = float(ans[i][0][4]) - 380000
        temp[3] = float(ans[i][0][5])
        temp[4] = float(ans[i][0][6])
        for j in range(1, len(ans[i])):
            temp[int(ans[i][j][0]) + 4] = float(ans[i][j][1])
        print(temp)
        source.append(temp)

    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)

    df = pd.DataFrame(data = source,
                      columns = ['event', 'x', 'y', 'z', 't0', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                      )

    print(df.head())
    df.to_csv("source.csv")



    # 对事件进行分组，再根据传感器序号进行排序





