import numpy as np, os
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.datasets.kitti import KITTIOdometryDataset

pipeline = OdometryPipeline(
    dataset=KITTIOdometryDataset(
        data_dir="/home/yuan/kitti_data/dataset",
        sequence="00",
    ),
    visualize=False,
)
pipeline.run()

poses = np.array(pipeline.poses)
os.makedirs("/home/yuan/slam_results/kitti_baseline", exist_ok=True)
np.savetxt("/home/yuan/slam_results/kitti_baseline/traj.txt",
           poses.reshape(-1,16)[:,:-4].reshape(-1,12))
print(f"基线完成！共{len(poses)}帧")
