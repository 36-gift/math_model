import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 假设有一组数据
data = np.random.normal(loc=5, scale=2, size=1000)

# 拟合正态分布
mu, sigma = norm.fit(data)
print(f"拟合的均值: {mu}, 拟合的标准差: {sigma}")

# 绘制拟合结果
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
x = np.linspace(min(data), max(data), 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label='拟合的正态分布')
plt.legend()
plt.show()