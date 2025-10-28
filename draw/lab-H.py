import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 原始数据
data = {
    'hidden_size': [512, 256, 128, 64, 512, 256, 128, 64],
    'light_score': [0.7437, 0.7425, 0.7449, 0.7475, 0.7359, 0.7465, 0.7472, 0.7469],
    'wind_score': [0.6594, 0.6547, 0.6518, 0.6567, 0.6591, 0.6573, 0.6547, 0.6508]
}

df = pd.DataFrame(data)

# 对每个 hidden_size 取平均（因为有两个重复实验）
df_mean = df.groupby('hidden_size', as_index=False).mean()

# 排序：hidden_size 从大到小
df_mean = df_mean.sort_values('hidden_size', ascending=False)
df_mean['hidden_size'] = df_mean['hidden_size'].astype(str)  # 转为字符串，避免数值缩放

# 设置学术风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'figure.figsize': [6, 4],
    'axes.spines.top': False,
    'axes.spines.right': False
})

# 创建子图：1 行 2 列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

# --- 左图：Light Score ---
bars1 = ax1.bar(df_mean['hidden_size'], df_mean['light_score'],
                color='steelblue', edgecolor='black', linewidth=0.8, alpha=0.8)
ax1.set_xlabel('Hidden Size')
ax1.set_ylabel('Light Score')
ax1.set_title('(a) Light Score vs Hidden Size', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_ylim(0.73, 0.75)  # 可根据数据微调
# 在柱子上方添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 2), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

# --- 右图：Wind Score ---
bars2 = ax2.bar(df_mean['hidden_size'], df_mean['wind_score'],
                color='orange', edgecolor='black', linewidth=0.8, alpha=0.8)
ax2.set_xlabel('Hidden Size')
ax2.set_ylabel('Wind Score')
ax2.set_title('(b) Wind Score vs Hidden Size', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.set_ylim(0.64, 0.665)
# 在柱子上方添加数值标签
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 2), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9)

# 调整布局
plt.tight_layout(pad=2.0)

# 保存为高分辨率图像（适合论文）
plt.savefig('separate_bar_plots.pdf', dpi=600, bbox_inches='tight')
plt.savefig('separate_bar_plots.png', dpi=600, bbox_inches='tight')

# 显示
plt.show()