import matplotlib.pyplot as plt
import numpy as np

# 数据
batch_sizes = [64, 48, 32, 16]
solar_scores = [0.7465, 0.7430, 0.7408, 0.7434]
wind_score_seed1 = [0.6527, 0.6563, 0.6559, 0.6538]
wind_score_seed2 = [0.6547, 0.6562, 0.6576, 0.6552]

# 设置中文字体和负号显示（可选，若不需要中文可忽略）
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42  # 保存为pdf时字体嵌入
plt.rcParams['ps.fonttype'] = 42

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(15, 6))

# 设置柱子宽度和位置
x = np.arange(len(batch_sizes))
width = 0.25

# 绘制三组风能得分（柱状图）
bars1 = ax1.bar(x - width, wind_score_seed1, width, label='Wind Score (Seed 1)', color='skyblue', edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x, wind_score_seed2, width, label='Wind Score (Seed 2)', color='lightcoral', edgecolor='black', linewidth=0.8)

# 绘制太阳能得分（折线图，更突出）
ax1.plot(x + width, solar_scores, color='goldenrod', marker='o', markersize=8, linewidth=2.5,
         label='Solar Score', linestyle='-', markerfacecolor='yellow', markeredgecolor='darkgoldenrod', markeredgewidth=1.5)

# 设置 x 轴标签
ax1.set_xlabel('Batch Size', fontsize=14, weight='bold')
ax1.set_ylabel('Performance Score', fontsize=14, weight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes, fontsize=12)
ax1.set_ylim(0.64, 0.76)

# 添加图例
ax1.legend(loc='upper left', fontsize=12, frameon=True, fancybox=False, edgecolor='black')

# 在柱子上方添加数值标签（风能）
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, color='black')

# 为折线图添加数值标签（太阳能）
for i, score in enumerate(solar_scores):
    ax1.annotate(f'{score:.4f}',
                 xy=(x[i] + width, score),
                 xytext=(0, 8),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, color='goldenrod', weight='bold')

# 网格（仅 y 方向）
ax1.grid(axis='y', linestyle='--', alpha=0.6, linewidth=0.8)

# 标题（可选）
# plt.title('Model Performance under Different Batch Sizes', fontsize=16, pad=20)

# 调整布局
plt.tight_layout()

# 保存为高分辨率 PDF 和 PNG（适合论文）
plt.savefig('hyperparameter_performance.pdf', dpi=900, bbox_inches='tight')
plt.savefig('hyperparameter_performance.png', dpi=900, bbox_inches='tight')

# 显示图形
plt.show()