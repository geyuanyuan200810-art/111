
import numpy as np, struct, shutil
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from pathlib import Path

typestore = get_typestore(Stores.ROS2_HUMBLE)
PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
Imu         = typestore.types["sensor_msgs/msg/Imu"]
Header      = typestore.types["std_msgs/msg/Header"]
Time        = typestore.types["builtin_interfaces/msg/Time"]
Vec3        = typestore.types["geometry_msgs/msg/Vector3"]
Quat        = typestore.types["geometry_msgs/msg/Quaternion"]
PointField  = typestore.types["sensor_msgs/msg/PointField"]

bag_path = Path("/home/yuan/slam_data/sim_lidar_imu2")
if bag_path.exists():
    shutil.rmtree(bag_path)

N_SCAN, N_PTS, N_FRAMES = 16, 1800, 600
point_step = 20

with Writer(bag_path, version=8) as writer:
    lc = writer.add_connection("/points", "sensor_msgs/msg/PointCloud2", typestore=typestore)
    ic = writer.add_connection("/imu/data", "sensor_msgs/msg/Imu", typestore=typestore)

    for i in range(N_FRAMES):
        t_ns = int(i * 0.05 * 1e9)
        ao = i * 0.02
        raw = bytearray()
        for ring in range(N_SCAN):
            elev = np.radians(-15 + ring * 2.0)
            for az_idx in range(N_PTS):
                az = np.radians(az_idx * 360.0 / N_PTS) + ao
                r = 5.0 + 2.0*np.sin(az*3) + np.random.randn()*0.05
                x = r * np.cos(elev) * np.cos(az)
                y = r * np.cos(elev) * np.sin(az)
                z = r * np.sin(elev)
                raw += struct.pack("ffffH2x", x, y, z, 50.0+ring*10, ring)

        fields = [
            PointField(name="x",         offset=0,  datatype=7, count=1),
            PointField(name="y",         offset=4,  datatype=7, count=1),
            PointField(name="z",         offset=8,  datatype=7, count=1),
            PointField(name="intensity", offset=12, datatype=7, count=1),
            PointField(name="ring",      offset=16, datatype=4, count=1),
        ]
        n = N_SCAN * N_PTS
        pc = PointCloud2(
            header=Header(stamp=Time(sec=t_ns//10**9, nanosec=t_ns%10**9), frame_id="base_link"),
            height=1, width=n, fields=fields,
            is_bigendian=False, point_step=point_step, row_step=point_step*n,
            data=np.frombuffer(bytes(raw), dtype=np.uint8), is_dense=True)
        writer.write(lc, t_ns, typestore.serialize_cdr(pc, PointCloud2.__msgtype__))

        for j in range(10):
            ti = int((i*0.05 + j*0.005)*1e9)
            imu = Imu(
                header=Header(stamp=Time(sec=ti//10**9, nanosec=ti%10**9), frame_id="base_link"),
                orientation=Quat(x=0.0,y=0.0,z=float(np.sin(ao/2)),w=float(np.cos(ao/2))),
                orientation_covariance=np.zeros(9),
                angular_velocity=Vec3(x=0.0,y=0.0,z=0.1+float(np.random.randn()*0.005)),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=Vec3(x=float(np.random.randn()*0.05),
                    y=float(np.random.randn()*0.05), z=9.81+float(np.random.randn()*0.02)),
                linear_acceleration_covariance=np.zeros(9))
            writer.write(ic, ti, typestore.serialize_cdr(imu, Imu.__msgtype__))

        if i % 100 == 0:
            print(f"进度 {i}/{N_FRAMES}")

print("完成！")
