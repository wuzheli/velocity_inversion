# 生成波速模型 (z,y,x)
import numpy as np
import pandas as pd
if __name__ == "__main__":
    number = 1        # 生成数量
    vel_0 = 5200        # 初始波速
    f = 0.2             # 波动比例
    # 创建数组（3维）
    for i in range(1, number + 1):
        vel_model = np.random.randint(vel_0*(1-f)/10, vel_0*(1+f)/10, (10,12,26)) * 10  # 生成随机波速模型
        np.save('vel_model_initial'+ str(i), arr=vel_model) # 保存模型