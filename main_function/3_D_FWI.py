import numpy as np


def forward_model_3d(m, source, nx, ny, nz):
    """
    简化的3D前向模型函数，假设使用卷积来模拟波传播。
    参数:
    - m: 3D速度模型
    - source: 3D震源信号
    - nx, ny, nz: 3D网格的维度

    返回值:
    - d_syn: 合成的波场数据
    """
    d_syn = np.zeros_like(source)  # 初始化合成数据数组
    sx, sy, sz = source.shape  # 获取震源信号的形状
    mx, my, mz = m.shape  # 获取速度模型的形状

    # 遍历网格中的每个点，计算波场值
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ix + sx <= mx and iy + sy <= my and iz + sz <= mz:
                    d_syn[ix, iy, iz] = np.sum(source * m[ix:ix + sx, iy:iy + sy, iz:iz + sz])
    return d_syn


def misfit(d_obs, d_syn):
    """
    计算观测数据和合成数据之间的误差。
    参数:
    - d_obs: 观测数据
    - d_syn: 合成数据

    返回值:
    - error: 误差的平方和
    """
    return np.sum((d_obs - d_syn) ** 2)


def gradient_3d(d_obs, d_syn, source, nx, ny, nz):
    """
    计算误差对模型参数的梯度。
    参数:
    - d_obs: 观测数据
    - d_syn: 合成数据
    - source: 3D震源信号
    - nx, ny, nz: 3D网格的维度

    返回值:
    - grad: 误差对模型参数的梯度
    """
    grad = np.zeros_like(source)  # 初始化梯度数组
    sx, sy, sz = source.shape  # 获取震源信号的形状

    # 遍历网格中的每个点，计算梯度值
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if ix + sx <= nx and iy + sy <= ny and iz + sz <= nz:
                    grad[ix, iy, iz] = np.sum((d_syn - d_obs) * source[ix:ix + sx, iy:iy + sy, iz:iz + sz])
    return grad


def full_waveform_inversion_3d(d_obs, source, m0, learning_rate, n_iter, nx, ny, nz):
    """
    全波形反演的主循环。
    参数:
    - d_obs: 观测数据
    - source: 3D震源信号
    - m0: 初始模型
    - learning_rate: 学习率
    - n_iter: 迭代次数
    - nx, ny, nz: 3D网格的维度

    返回值:
    - m: 反演得到的模型
    """
    m = m0.copy()  # 初始化模型为初始模型的副本

    for i in range(n_iter):
        d_syn = forward_model_3d(m, source, nx, ny, nz)  # 生成合成数据
        error = misfit(d_obs, d_syn)  # 计算误差
        grad = gradient_3d(d_obs, d_syn, source, nx, ny, nz)  # 计算梯度
        m -= learning_rate * grad  # 更新模型
        print(f"Iteration {i + 1}, Misfit: {error}")  # 输出当前迭代的误差

    return m


# 示例数据
nx, ny, nz = 50, 50, 50
true_model = np.zeros((nx, ny, nz))
true_model[20:30, 20:30, 20:30] = 1  # 简单的块状速度模型
source = np.zeros((nx, ny, nz))
source[25, 25, 25] = 1  # 简单的脉冲震源信号
d_obs = forward_model_3d(true_model, source, nx, ny, nz)  # 生成观测数据

# 初始模型
initial_model = np.zeros((nx, ny, nz))

# 反演
n_iter = 50
learning_rate = 0.01
recovered_model = full_waveform_inversion_3d(d_obs, source, initial_model, learning_rate, n_iter, nx, ny, nz)

# 输出结果
print("True model:")
print(true_model)
print("Recovered model:")
print(recovered_model)