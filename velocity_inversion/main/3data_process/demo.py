import numpy as np
from scipy.interpolate import griddata

# 构造数据
grid_x, grid_y, grid_z = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j]
points = np.random.rand(1000, 3)
values = np.random.randn(1000)

# 进行插值
grid = griddata(points, values, (grid_x, grid_y, grid_z), method='cubic')
print(grid)