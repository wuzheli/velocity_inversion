# 生成理论波速模型
import numpy as np
import pandas as pd
import os
if __name__ == "__main__":
    number = 200        # 生成数量
    vel_min = 4500    # 最小波速
    vel_max = 6000    # 最大波速
    step = 50         # 步长
    save_path = 'C:\\Users\\建设村彭于晏\\Desktop\python程序\\velocity_inversion\\main\\1velocity_models\\theoretical_model\\'
    # 创建波速范围数组
    vel_range = np.arange(vel_min, vel_max + step, step)

    # 创建数组（3维）
    for i in range(1, number + 1):
        vel_model = np.random.choice(vel_range, size=(4, 7, 17))  # 生成随机波速模型
        file_name = os.path.join(save_path, 'vel_model_' + str(i) + '.npy')
        print(vel_model)
        # np.save(file_name, arr=vel_model)  # 保存模型
        velocity = vel_model.T.reshape(4, 7*17)
        df_vel = pd.DataFrame(data = velocity)
        file_name = os.path.join(save_path, 'vel_model_' + str(i) + '.xlsx')
        # df_vel.to_excel(file_name, index=False, header=False)
        
    
        