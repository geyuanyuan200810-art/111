import numpy as np, struct, shutil
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
import glob

typestore = get_typestore(Stores.ROS2_HUMBLE)
PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
Imu         = typestore.types['sensor_msgs/msg/Imu']
Header      = typestore.types['std_msgs/msg/Header']
Time        = typestore.types['builtin_interfaces/msg/Time']
PointField  = typestore.types['sensor_msgs/msg/PointField']
Vec3        = typestore.types['geometry_msgs/msg/Vector3']
Quat        = typestore.types['geometry_msgs/msg/Quaternion']

WEIGHT_MAP = {10:0.05,11:0.05,13:0.05,15:0.05,16:0.05,18:0.05,20:0.05,
              30:0.10,31:0.10,32:0.10}

SEQ = "/home/yuan/kitti_data/dataset/sequences/00"
velodyne_files = sorted(glob.glob(f"{SEQ}/velodyne/*.bin"))
label_files    = sorted(glob.glob(f"{SEQ}/labels/*.label"))
times          = np.loadtxt(f"{SEQ}/times.txt")
poses = []
with open(f"{SEQ}/poses.txt") as f:
    for line in f:
        T = np.array(line.strip().split(), dtype=np.float64).reshape(3,4)
        poses.append(T)

def calc_ring(pts, n_scan=64):
    """按仰角计算每个点的ring编号（HDL-64E: -24.9° 到 +2°）"""
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    dist_xy = np.sqrt(x**2 + y**2)
    elev = np.degrees(np.arctan2(z, dist_xy))
    # HDL-64E 垂直角度范围
    elev_min, elev_max = -24.9, 2.0
    elev_clipped = np.clip(elev, elev_min, elev_max)
    ring = ((elev_clipped - elev_min) / (elev_max - elev_min) * (n_scan-1))
    return ring.astype(np.uint16)

def make_pc(pts, labels, t_ns, use_semantic=False):
    rings = calc_ring(pts)
    raw = bytearray()
    for j in range(len(pts)):
        intensity = float(pts[j,3]) * 255.0
        if use_semantic:
            w = WEIGHT_MAP.get(int(labels[j]), 1.0)
            intensity *= w
        raw += struct.pack('ffffH2x',
            float(pts[j,0]), float(pts[j,1]),
            float(pts[j,2]), intensity, rings[j])
    n = len(pts)
    fields = [
        PointField(name='x',         offset=0,  datatype=7, count=1),
        PointField(name='y',         offset=4,  datatype=7, count=1),
        PointField(name='z',         offset=8,  datatype=7, count=1),
        PointField(name='intensity', offset=12, datatype=7, count=1),
        PointField(name='ring',      offset=16, datatype=4, count=1),
    ]
    return PointCloud2(
        header=Header(stamp=Time(sec=t_ns//10**9, nanosec=t_ns%10**9),
                      frame_id='base_link'),
        height=1, width=n, fields=fields,
        is_bigendian=False, point_step=20, row_step=20*n,
        data=np.frombuffer(bytes(raw), dtype=np.uint8), is_dense=True)

def make_imu(t_ns, gyro, acc):
    return Imu(
        header=Header(stamp=Time(sec=t_ns//10**9, nanosec=t_ns%10**9),
                      frame_id='base_link'),
        orientation=Quat(x=0.0,y=0.0,z=0.0,w=1.0),
        orientation_covariance=np.zeros(9),
        angular_velocity=Vec3(x=float(gyro[0]),y=float(gyro[1]),z=float(gyro[2])),
        angular_velocity_covariance=np.zeros(9),
        linear_acceleration=Vec3(x=float(acc[0]),y=float(acc[1]),z=float(acc[2])),
        linear_acceleration_covariance=np.zeros(9))

for bag_type in ['baseline', 'semantic']:
    bag_path = Path(f"/home/yuan/slam_data/kitti_{bag_type}_v2")
    if bag_path.exists(): shutil.rmtree(bag_path)
    print(f"\n生成 {bag_type} bag（正确ring）...")

    with Writer(bag_path, version=8) as w:
        cc = w.add_connection('/points',  'sensor_msgs/msg/PointCloud2', typestore=typestore)
        ci = w.add_connection('/imu_raw', 'sensor_msgs/msg/Imu',         typestore=typestore)

        for i in range(len(velodyne_files)):
            t_ns = int(times[i] * 1e9)
            pts    = np.fromfile(velodyne_files[i], dtype=np.float32).reshape(-1,4)
            labels = np.fromfile(label_files[i],    dtype=np.uint32) & 0xFFFF

            pc_msg = make_pc(pts, labels, t_ns, use_semantic=(bag_type=='semantic'))
            w.write(cc, t_ns, typestore.serialize_cdr(pc_msg, PointCloud2.__msgtype__))

            # IMU
            dt = (times[i+1]-times[i]) if i < len(times)-1 else 0.1
            if i > 0 and dt > 0:
                R0 = poses[i-1][:3,:3]; R1 = poses[i][:3,:3]
                dR = R1 @ R0.T
                angle = np.arccos(np.clip((np.trace(dR)-1)/2,-1,1))
                ax = np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])
                norm = np.linalg.norm(ax)
                gyro = (ax/norm*angle/dt) if norm>1e-8 else np.zeros(3)
            else:
                gyro = np.zeros(3)
            acc = np.array([0.0, 0.0, 9.81])

            for k in range(10):
                t_imu = t_ns + k*int(dt/10*1e9)
                imu_msg = make_imu(t_imu,
                    gyro + np.random.randn(3)*0.005,
                    acc  + np.random.randn(3)*0.05)
                w.write(ci, t_imu, typestore.serialize_cdr(imu_msg, Imu.__msgtype__))

            if i % 200 == 0:
                print(f"  {i}/{len(velodyne_files)}")

print("\n完成！")
