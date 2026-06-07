import numpy as np
import glob, os
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.datasets.kitti import KITTIDataset

# 临时把00_semantic改名为序列11（序列11没有真值，但可以运行）
import shutil
src = "/home/yuan/kitti_data/dataset/sequences/00_semantic"
dst = "/home/yuan/kitti_data/dataset/sequences/11"
if not os.path.exists(dst):
    shutil.copytree(src, dst)

# 跑KISS-ICP
pipeline = OdometryPipeline(
    dataset=KITTIDataset(
        data_dir="/home/yuan/kitti_data/dataset",
        sequence="11",
    ),
    config=None,
    deskew=True,
    visualize=False,
)
pipeline.run()

# 保存轨迹
results = pipeline.results
poses = results.poses
np.savetxt("/home/yuan/slam_results/kitti_semantic/traj.txt",
           poses.reshape(-1, 16)[:, :12])
print(f"语义版轨迹已保存，共 {len(poses)} 帧")
