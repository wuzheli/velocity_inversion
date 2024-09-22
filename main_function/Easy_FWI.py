import numpy as np

def forward_model(m, source):
    # 简化的前向模型函数，假设一维情况
    # m: 模型参数（速度）
    # source: 震源信号
    return np.convolve(source, m, mode='same')

def misfit(d_obs, d_syn):
    # 计算观测数据和合成数据之间的误差
    return np.sum((d_obs - d_syn)**2)

def gradient(d_obs, d_syn, source):
    # 计算误差对模型参数的梯度
    return np.convolve(d_syn - d_obs, source[::-1], mode='same')

def full_waveform_inversion(d_obs, source, m0, learning_rate, n_iter):
    # FWI 主循环
    m = m0.copy()
    for i in range(n_iter):
        d_syn = forward_model(m, source)
        error = misfit(d_obs, d_syn)
        grad = gradient(d_obs, d_syn, source)
        m -= learning_rate * grad
        print(f"Iteration {i+1}, Misfit: {error}")
    return m

# 示例数据
n_points = 100
true_model = np.zeros(n_points)
true_model[40:60] = 1  # 简单的块状速度模型
source = np.zeros(n_points)
source[50] = 1  # 简单的脉冲震源信号
d_obs = forward_model(true_model, source)

# 初始模型
initial_model = np.zeros(n_points)

# 反演
n_iter = 50
learning_rate = 0.01
recovered_model = full_waveform_inversion(d_obs, source, initial_model, learning_rate, n_iter)

print("True model:")
print(true_model)
print("Recovered model:")
print(recovered_model)

'''
forward_model: 这是一个简化的前向模型函数，它假设一维情况并使用卷积来模拟波传播。

misfit: 这个函数计算观测数据和合成数据之间的误差（失配函数），这里使用的是平方误差。

gradient: 这个函数计算误差对模型参数的梯度，这里使用简单的反卷积来计算。

full_waveform_inversion: FWI 的主循环，在每次迭代中计算合成数据，误差和梯度，并更新模型参数。

示例数据: 使用简单的块状速度模型和脉冲震源信号来生成观测数据，并使用零值的初始模型开始反演
'''