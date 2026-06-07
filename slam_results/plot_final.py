import numpy as np, zipfile, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'WenQuanYi Zen Hei'
matplotlib.rcParams['axes.unicode_minus'] = False

def load_ate(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open('stats.json') as f:
            stats = json.load(f)
        with z.open('error_array.npy') as f:
            errors = np.load(f)
    return errors, stats

base_err, base_stats = load_ate('/home/yuan/slam_results/baseline_ate.zip')
sem_err,  sem_stats  = load_ate('/home/yuan/slam_results/semantic_ate.zip')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：ATE误差随帧变化曲线
ax = axes[0]
ax.plot(base_err, color='#ef4444', lw=1,   alpha=0.8, label=f'基线 (均值={base_stats["mean"]:.3f}m)')
ax.plot(sem_err,  color='#2563eb', lw=1,   alpha=0.8, label=f'语义权重 (均值={sem_stats["mean"]:.3f}m)')
ax.set_xlabel('帧编号')
ax.set_ylabel('ATE (m)')
ax.set_title('KITTI 序列00 — ATE误差曲线对比')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2：箱线图对比
ax = axes[1]
bp = ax.boxplot([base_err, sem_err],
                labels=['基线\nKISS-ICP', '语义权重\n（本研究）'],
                patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
bp['boxes'][0].set_facecolor('#ef444480')
bp['boxes'][1].set_facecolor('#2563eb80')
ax.set_ylabel('ATE (m)')
ax.set_title('ATE 分布箱线图')
ax.grid(True, alpha=0.3, axis='y')

# 标注改善幅度
improve = (base_stats['mean'] - sem_stats['mean']) / base_stats['mean'] * 100
ax.text(1.5, max(base_err)*0.9,
        f'ATE均值改善\n↓ {improve:.1f}%',
        ha='center', fontsize=12, color='#16a34a',
        fontweight='bold')

plt.tight_layout()
plt.savefig('/home/yuan/slam_results/ate_final.png', dpi=150)
print(f"图表保存完成！")
print(f"\n最终结果：")
print(f"  基线 ATE 均值:    {base_stats['mean']:.3f} m")
print(f"  语义版 ATE 均值:  {sem_stats['mean']:.3f} m")
print(f"  改善幅度:         {improve:.1f}%")
