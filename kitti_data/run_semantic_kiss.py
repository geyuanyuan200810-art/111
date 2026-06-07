import numpy as np, os, glob, struct
from pathlib import Path

SEQ = "/home/yuan/kitti_data/dataset/sequences/00"
OUT = "/home/yuan/kitti_data/dataset/sequences/00_semantic/velodyne"
os.makedirs(OUT, exist_ok=True)

DYNAMIC = {10,11,13,15,16,18,20}

vfiles = sorted(glob.glob(f"{SEQ}/velodyne/*.bin"))
lfiles = sorted(glob.glob(f"{SEQ}/labels/*.label"))

for i,(vf,lf) in enumerate(zip(vfiles,lfiles)):
    pts    = np.fromfile(vf, dtype=np.float32).reshape(-1,4)
    labels = np.fromfile(lf, dtype=np.uint32) & 0xFFFF
    # 直接删除动态点（最强版本）
    mask = ~np.isin(labels, list(DYNAMIC))
    pts_filtered = pts[mask]
    pts_filtered.tofile(f"{OUT}/{os.path.basename(vf)}")
    if i % 500 == 0:
        removed = (~mask).sum()
        print(f"{i}/{len(vfiles)} 过滤掉 {removed} 个动态点")

# 复制其他文件
import shutil
for f in ['calib.txt','times.txt','poses.txt']:
    shutil.copy(f"{SEQ}/{f}",
                f"/home/yuan/kitti_data/dataset/sequences/00_semantic/{f}")
print("完成！")
