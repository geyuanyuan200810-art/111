import numpy as np

# SemanticKITTI 标签→权重映射表
# 完整类别定义见 semantic-kitti.org/dataset.html
WEIGHT_MAP = {
    0:  1.0,   # 未标注 → 保守处理，给全权重
    10: 0.05,  # 车辆(moving car)
    11: 0.05,  # 摩托车(moving)
    13: 0.05,  # 公交车(moving)
    15: 0.05,  # 行人(moving person)
    16: 0.05,  # 骑手(moving)
    18: 0.05,  # 卡车(moving)
    20: 0.05,  # 其他移动物体
    30: 0.10,  # 停靠的车辆（低权重但比移动车高）
    31: 0.10,  # 停靠的摩托车
    32: 0.10,  # 停靠的自行车
    # 静态类别 → 全权重
    40: 1.0,   # 道路
    44: 1.0,   # 停车场
    48: 1.0,   # 人行道
    49: 1.0,   # 其他地面
    50: 1.0,   # 建筑
    51: 1.0,   # 栅栏
    52: 1.0,   # 其他结构
    60: 1.0,   # 车道标线
    70: 1.0,   # 植被
    71: 1.0,   # 树干
    72: 1.0,   # 地形
    80: 1.0,   # 杆
    81: 1.0,   # 交通标志
    99: 1.0,   # 其他静态
}

def get_weight_map(label_path):
    """读取一帧的语义标签，返回每个点的权重"""
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem = labels & 0xFFFF
    weights = np.ones(len(sem), dtype=np.float32)
    for cls_id, w in WEIGHT_MAP.items():
        weights[sem == cls_id] = w
    return weights

def analyze_frame(label_path):
    """分析一帧中动态/静态点的比例"""
    weights = get_weight_map(label_path)
    dynamic = (weights < 0.5).sum()
    total = len(weights)
    print(f"总点数: {total}")
    print(f"动态点: {dynamic} ({100*dynamic/total:.1f}%)")
    print(f"静态点: {total-dynamic} ({100*(total-dynamic)/total:.1f}%)")
    return weights

# 测试第一帧
if __name__ == "__main__":
    import glob, os
    label_dir = "/home/yuan/kitti_data/dataset/sequences/00/labels"
    labels = sorted(glob.glob(f"{label_dir}/*.label"))
    print(f"共找到 {len(labels)} 帧标签\n")
    
    # 分析前5帧
    for lp in labels[:5]:
        fname = os.path.basename(lp)
        print(f"--- 帧 {fname} ---")
        analyze_frame(lp)
        print()
