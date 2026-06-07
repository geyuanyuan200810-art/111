import numpy as np, struct, shutil
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import glob, os

typestore = get_typestore(Stores.ROS2_HUMBLE)
PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
Header      = typestore.types['std_msgs/msg/Header']
Time        = typestore.types['builtin_interfaces/msg/Time']
PointField  = typestore.types['sensor_msgs/msg/PointField']

# 语义权重映射
WEIGHT_MAP = {10:0.05,11:0.05,13:0.05,15:0.05,16:0.05,18:0.05,20:0.05,
              30:0.10,31:0.10,32:0.10}

SEQ = "/home/yuan/kitti_data/dataset/sequences/00"
velodyne_files = sorted(glob.glob(f"{SEQ}/velodyne/*.bin"))
label_files    = sorted(glob.glob(f"{SEQ}/labels/*.label"))
times          = np.loadtxt(f"{SEQ}/times.txt")

print(f"共 {len(velodyne_files)} 帧，开始转换...")

# ── 基线版（不加语义权重）──
bag_base = Path("/home/yuan/slam_data/kitti_baseline")
if bag_base.exists(): shutil.rmtree(bag_base)

# ── 语义版（加语义权重）──
bag_sem  = Path("/home/yuan/slam_data/kitti_semantic")
if bag_sem.exists(): shutil.rmtree(bag_sem)

def make_pc_msg(pts_raw, t_ns):
    raw = bytearray()
    for p in pts_raw:
        raw += struct.pack('ffffH2x', p[0], p[1], p[2], p[3], int(p[4]))
    fields = [
        PointField(name='x',         offset=0,  datatype=7, count=1),
        PointField(name='y',         offset=4,  datatype=7, count=1),
        PointField(name='z',         offset=8,  datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7, count=1),
        PointField(name='ring',      offset=16, datatype=4, count=1),
    ]
    n = len(pts_raw)
    return PointCloud2(
        header=Header(stamp=Time(sec=t_ns//10**9, nanosec=t_ns%10**9),
                      frame_id='base_link'),
        height=1, width=n, fields=fields,
        is_bigendian=False, point_step=20, row_step=20*n,
        data=np.frombuffer(bytes(raw), dtype=np.uint8), is_dense=True)

with Writer(bag_base, version=8) as wb, Writer(bag_sem, version=8) as ws:
    cb = wb.add_connection('/points', 'sensor_msgs/msg/PointCloud2', typestore=typestore)
    cs = ws.add_connection('/points', 'sensor_msgs/msg/PointCloud2', typestore=typestore)

    for i, (vf, lf, t) in enumerate(zip(velodyne_files, label_files, times)):
        t_ns = int(t * 1e9)

        # 读取点云
        pts = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)
        # 读取语义标签
        labels = np.fromfile(lf, dtype=np.uint32) & 0xFFFF

        # 简单分配ring（模拟64线雷达）
        n = len(pts)
        rings = (np.arange(n) % 64).astype(np.uint16)

        # 基线：intensity原值
        base_pts = [(p[0],p[1],p[2],p[3]*255,rings[j])
                    for j,p in enumerate(pts)]

        # 语义版：intensity乘权重
        sem_pts = []
        for j,p in enumerate(pts):
            w = WEIGHT_MAP.get(int(labels[j]), 1.0)
            sem_pts.append((p[0],p[1],p[2],p[3]*255*w,rings[j]))

        # 写入两个bag
        base_msg = make_pc_msg(base_pts, t_ns)
        sem_msg  = make_pc_msg(sem_pts,  t_ns)
        wb.write(cb, t_ns, typestore.serialize_cdr(base_msg, PointCloud2.__msgtype__))
        ws.write(cs, t_ns, typestore.serialize_cdr(sem_msg,  PointCloud2.__msgtype__))

        if i % 200 == 0:
            print(f"进度: {i}/{len(velodyne_files)}")

print("两个bag均已生成！")
print(f"  基线: ~/slam_data/kitti_baseline")
print(f"  语义: ~/slam_data/kitti_semantic")
