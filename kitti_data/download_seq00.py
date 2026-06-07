import zipfile, requests, io, os

print("开始流式下载序列00点云...")
url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
out_dir = os.path.expanduser("~/kitti_data/dataset/sequences")

# 先获取zip目录结构（只下前几MB）
resp = requests.get(url, stream=True, timeout=30)
buf = b""
for chunk in resp.iter_content(chunk_size=1024*1024):
    buf += chunk
    if len(buf) > 5*1024*1024:  # 下5MB后检查
        break
resp.close()

print(f"已获取 {len(buf)/1024/1024:.1f} MB，检查文件结构...")
print("请直接用 wget 分段下载，见下方命令")
