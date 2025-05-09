import matplotlib.pyplot as plt
import numpy as np

# 生成虚拟数据集
np.random.seed(42)
n = 1000
A = np.random.normal(loc=50, scale=10, size=n)  # 特征 A，正态分布
B = A + np.random.normal(loc=0, scale=5, size=n)  # 特征 B，与 A 相关联
C = np.random.randint(0, 3, size=n)  # 特征 C，离散类别数据 (0, 1, 2)

# 创建一个图形，包含 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Data Analysis: Feature A, B, and C', fontsize=16)

# 子图 1: A 和 B 的折线图
axs[0, 0].plot(A, label='Feature A', color='red', linewidth=2)
axs[0, 0].plot(B, label='Feature B', color='blue', linestyle='--', linewidth=2)
axs[0, 0].set_title('Line Plot of Feature A and B')
axs[0, 0].set_xlabel('Index')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 子图 2: A 和 B 的散点图，C 作为颜色映射
scatter = axs[0, 1].scatter(A, B, c=C, cmap='plasma', s=100, edgecolor='k', alpha=0.8)
axs[0, 1].set_title('Scatter Plot of Feature A vs Feature B (Colored by Feature C)')
axs[0, 1].set_xlabel('Feature A')
axs[0, 1].set_ylabel('Feature B')
# 添加颜色条
cbar = fig.colorbar(scatter, ax=axs[0, 1])
cbar.set_label('Feature C Categories')

# 子图 3: A 和 B 的直方图
axs[1, 0].hist(A, bins=15, color='green', alpha=0.7, label='Feature A')
axs[1, 0].hist(B, bins=15, color='orange', alpha=0.7, label='Feature B')
axs[1, 0].set_title('Histogram of Feature A and B')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 子图 4: 箱线图，展示不同类别 C 的数据分布 (针对 A 和 B)
axs[1, 1].boxplot([A[C == 0], A[C == 1], A[C == 2]], positions=[1, 2, 3], widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='cyan', color='black'),
                  medianprops=dict(color='black'))
axs[1, 1].boxplot([B[C == 0], B[C == 1], B[C == 2]], positions=[4, 5, 6], widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='pink', color='black'),
                  medianprops=dict(color='black'))
axs[1, 1].set_title('Boxplot of Feature A and B Grouped by Feature C')
axs[1, 1].set_xticks([1.5, 4.5])
axs[1, 1].set_xticklabels(['Feature A', 'Feature B'])
axs[1, 1].set_ylabel('Value')

# 调整子图之间的布局
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 显示图形
plt.show()