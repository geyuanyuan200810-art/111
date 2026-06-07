import numpy as np
import glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

label_dir = "/home/yuan/kitti_data/dataset/sequences/00/labels"
labels = sorted(glob.glob(f"{label_dir}/*.label"))

DYNAMIC_CLASSES = {10,11,13,15,16,18,20}
PARKED_CLASSES  = {30,31,32}

dynamic_ratio = []
parked_ratio  = []

print(f"分析全部 {len(labels)} 帧...")
for lp in labels:
    lb = np.fromfile(lp, dtype=np.uint32) & 0xFFFF
    n  = len(lb)
    dynamic_ratio.append((np.isin(lb, list(DYNAMIC_CLASSES))).sum() / n * 100)
    parked_ratio.append((np.isin(lb, list(PARKED_CLASSES))).sum()  / n * 100)

# 绘制动态点比例曲线
fig, ax = plt.subplots(figsize=(10, 4))
frames = range(len(dynamic_ratio))
ax.fill_between(frames, dynamic_ratio, alpha=0.6, color='#ef4444', label='移动目标（行人/车辆）')
ax.fill_between(frames, parked_ratio,  alpha=0.4, color='#f59e0b', label='停靠车辆')
ax.set_xlabel('帧编号')
ax.set_ylabel('动态点占比 (%)')
ax.set_title('KITTI 序列00 — 各帧动态点比例（语义权重作用范围）')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/home/yuan/kitti_data/dynamic_ratio.png', dpi=150)
print(f"\n统计结果：")
print(f"  平均移动目标占比: {np.mean(dynamic_ratio):.2f}%")
print(f"  最高帧动态占比:   {np.max(dynamic_ratio):.2f}%")
print(f"  平均停靠车辆占比: {np.mean(parked_ratio):.2f}%")
print(f"图表已保存到 ~/kitti_data/dynamic_ratio.png")
