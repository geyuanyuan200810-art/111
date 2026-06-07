import numpy as np, struct, shutil
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore
from pathlib import Path

typestore  = get_typestore(Stores.ROS2_HUMBLE)
PC2        = typestore.types["sensor_msgs/msg/PointCloud2"]
Imu        = typestore.types["sensor_msgs/msg/Imu"]
Header     = typestore.types["std_msgs/msg/Header"]
Time       = typestore.types["builtin_interfaces/msg/Time"]
Vec3       = typestore.types["geometry_msgs/msg/Vector3"]
Quat       = typestore.types["geometry_msgs/msg/Quaternion"]
PointField = typestore.types["sensor_msgs/msg/PointField"]

N_SCAN, N_PTS, N_FRAMES = 16, 1800, 300
POINT_STEP = 20

def write_bag(bag_name, n_peds, n_vehs):
    bag_path = Path(f"/home/yuan/slam_data/{bag_name}")
    if bag_path.exists():
        shutil.rmtree(bag_path)
    rng = np.random.default_rng(42)
    ped_pos = rng.uniform(-10, 10, (n_peds, 2))
    ped_vel = rng.uniform(-0.3, 0.3, (n_peds, 2))
    veh_pos = rng.uniform(-18, 18, (n_vehs, 2)) if n_vehs>0 else np.zeros((0,2))
    veh_vel = rng.uniform(-0.8, 0.8, (n_vehs, 2)) if n_vehs>0 else np.zeros((0,2))
    print(f"生成 {bag_name}: {n_peds}行人 {n_vehs}车 {N_FRAMES}帧...")
    with Writer(bag_path, version=8) as writer:
        lc = writer.add_connection("/points","sensor_msgs/msg/PointCloud2",typestore=typestore)
        ic = writer.add_connection("/imu/data","sensor_msgs/msg/Imu",typestore=typestore)
        fields = [
            PointField(name="x",offset=0,datatype=7,count=1),
            PointField(name="y",offset=4,datatype=7,count=1),
            PointField(name="z",offset=8,datatype=7,count=1),
            PointField(name="intensity",offset=12,datatype=7,count=1),
            PointField(name="ring",offset=16,datatype=4,count=1),
        ]
        for i in range(N_FRAMES):
            t_ns = int(i*0.05*1e9)
            ao = i*0.02
            raw = bytearray()
            n_pts = 0
            # 静态环境
            for ring in range(N_SCAN):
                elev = np.radians(-15+ring*2.0)
                for az_idx in range(N_PTS):
                    az = np.radians(az_idx*360.0/N_PTS)+ao
                    r = 8.0+3.0*np.sin(az*2)+2.0*np.cos(az*3)+rng.standard_normal()*0.03
                    x = r*np.cos(elev)*np.cos(az)
                    y = r*np.cos(elev)*np.sin(az)
                    z = r*np.sin(elev)
                    raw += struct.pack("ffffH2x",x,y,z,60.0+ring*8,ring)
                    n_pts += 1
            # 行人点云
            for p in range(n_peds):
                px,py = ped_pos[p]
                dist = np.sqrt(px**2+py**2)
                if 2<dist<25:
                    az_p = np.arctan2(py,px)
                    for ring in range(4,10):
                        elev = np.radians(-15+ring*2.0)
                        z_at = dist*np.tan(elev)
                        if -0.3<z_at<1.8:
                            for daz in np.linspace(-0.025,0.025,3):
                                a = az_p+daz
                                r = dist+rng.standard_normal()*0.02
                                x = r*np.cos(elev)*np.cos(a)
                                y = r*np.cos(elev)*np.sin(a)
                                z = r*np.sin(elev)
                                raw += struct.pack("ffffH2x",x,y,z,80.0,ring)
                                n_pts += 1
                ped_pos[p] += ped_vel[p]*0.05
                for d in range(2):
                    if abs(ped_pos[p,d])>12: ped_vel[p,d]*=-1
            # 车辆点云
            for v in range(n_vehs):
                vx,vy = veh_pos[v]
                dist = np.sqrt(vx**2+vy**2)
                if 3<dist<35:
                    az_v = np.arctan2(vy,vx)
                    for ring in range(3,8):
                        elev = np.radians(-15+ring*2.0)
                        z_at = dist*np.tan(elev)
                        if -0.3<z_at<1.5:
                            for daz in np.linspace(-0.04,0.04,6):
                                a = az_v+daz
                                r = dist+rng.standard_normal()*0.03
                                x = r*np.cos(elev)*np.cos(a)
                                y = r*np.cos(elev)*np.sin(a)
                                z = r*np.sin(elev)
                                raw += struct.pack("ffffH2x",x,y,z,100.0,ring)
                                n_pts += 1
                veh_pos[v] += veh_vel[v]*0.05
                for d in range(2):
                    if abs(veh_pos[v,d])>22: veh_vel[v,d]*=-1
            pc = PC2(
                header=Header(stamp=Time(sec=t_ns//10**9,nanosec=t_ns%10**9),frame_id="base_link"),
                height=1,width=n_pts,fields=fields,
                is_bigendian=False,point_step=POINT_STEP,row_step=POINT_STEP*n_pts,
                data=np.frombuffer(bytes(raw),dtype=np.uint8),is_dense=True)
            writer.write(lc,t_ns,typestore.serialize_cdr(pc,PC2.__msgtype__))
            for j in range(10):
                ti = int((i*0.05+j*0.005)*1e9)
                imu = Imu(
                    header=Header(stamp=Time(sec=ti//10**9,nanosec=ti%10**9),frame_id="base_link"),
                    orientation=Quat(x=0.0,y=0.0,z=float(np.sin(ao/2)),w=float(np.cos(ao/2))),
                    orientation_covariance=np.zeros(9),
                    angular_velocity=Vec3(x=0.0,y=0.0,z=0.4+float(rng.standard_normal()*0.01)),
                    angular_velocity_covariance=np.zeros(9),
                    linear_acceleration=Vec3(x=float(rng.standard_normal()*0.05),
                        y=float(rng.standard_normal()*0.05),z=9.81+float(rng.standard_normal()*0.02)),
                    linear_acceleration_covariance=np.zeros(9))
                writer.write(ic,ti,typestore.serialize_cdr(imu,Imu.__msgtype__))
            if i%100==0: print(f"  帧{i}/{N_FRAMES}")
    print(f"  完成: {bag_path}")

write_bag("sim_scene1_static",  0, 0)
write_bag("sim_scene2_medium",  5, 0)
write_bag("sim_scene3_dynamic", 40, 8)
print("全部完成")
