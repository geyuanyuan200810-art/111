"""
confidence_adaptive_weight.py
=============================
语义置信度自适应权重方案

背景：
  run_semantic_kiss.py 直接将动态点权重设为 0（完全删除），相当于假设语义分割完美无误。
  但真实的语义分割网络（如 RangeNet++）存在误检和漏检：
    - 误检（False Positive）：动态点被误判为静态，5% 概率
    - 漏检（False Negative）：静态点被误判为动态，3% 概率
  本文件引入置信度自适应权重：
    - 置信度高的预测动态点 → 权重更低（我们信任这个判断，压制该点）
    - 置信度低的预测动态点 → 保留更多权重（可能是误检，不敢完全删除）
  这样可以减少因误检导致的有用静态点被错误删除问题。
"""

import numpy as np
import os
import glob
import shutil
import struct
import matplotlib
matplotlib.use('Agg')  # 无显示器环境使用非交互后端
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────── 常量定义 ───────────────────────────

# KITTI Sequence 00 数据路径（与 run_semantic_kiss.py 保持一致）
SEQ_DIR = "/home/yuan/kitti_data/dataset/sequences/00"

# 固定权重输出目录（现有方案：直接删除动态点，w=0）
OUT_FIXED = "/home/yuan/kitti_data/dataset/sequences/00_fixed_weight"

# 自适应权重输出目录（新方案：置信度加权保留低置信度疑似动态点）
OUT_ADAPTIVE = "/home/yuan/kitti_data/dataset/sequences/00_adaptive_weight"

# 最终分析图保存路径
RESULT_FIG = "/home/yuan/slam_results/adaptive_weight_analysis.png"

# 动态类别（SemanticKITTI 定义的移动物体类别）
# 10=car(moving), 11=motorcycle(moving), 13=bus(moving),
# 15=person(moving), 16=bicyclist(moving), 18=truck(moving), 20=other-vehicle(moving)
DYNAMIC_CLASSES = {10, 11, 13, 15, 16, 18, 20}

# 静态类别（道路、建筑、植被等固定结构）
# 40=road, 44=parking, 48=sidewalk, 49=other-ground, 50=building,
# 51=fence, 52=other-structure, 60=lane-marking, 70=vegetation,
# 71=trunk, 72=terrain, 80=pole, 81=traffic-sign
STATIC_CLASSES = {40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81}

# 误检率：动态点被预测为静态的概率（模拟网络漏检）
FALSE_NEGATIVE_RATE = 0.05  # 5%

# 漏检率：静态点被预测为动态的概率（模拟网络误报）
FALSE_POSITIVE_RATE = 0.03  # 3%

# 随机种子（保证结果可复现）
RANDOM_SEED = 42


# ─────────────────────── 置信度模拟函数 ───────────────────────────

def simulate_rangenet_predictions(true_labels: np.ndarray,
                                   rng: np.random.Generator):
    """
    模拟 RangeNet++ 语义分割网络的预测结果和置信度。

    物理意义：
      真实网络的输出是每个类别的 softmax 概率分布，最大概率作为置信度。
      我们用以下方式模拟这种不确定性：
        1. 对正确预测的点：置信度从 Beta(8,2) 分布采样（偏高，均值~0.8），
           模拟网络对正确预测通常比较自信。
        2. 对错误预测的点（误检/漏检）：置信度从 Beta(2,5) 分布采样（偏低，均值~0.29），
           模拟网络在出错时通常置信度较低。
        3. 根据误检率/漏检率翻转部分点的预测标签。

    参数：
      true_labels: 每个点的真值语义标签（uint32）
      rng: NumPy 随机数生成器

    返回：
      pred_is_dynamic: bool 数组，True 表示该点被预测为动态
      confidence:      float32 数组，每个点的预测置信度 ∈ [0, 1]
    """
    n = len(true_labels)

    # 判断真值中哪些是动态点、哪些是静态点
    true_is_dynamic = np.isin(true_labels, list(DYNAMIC_CLASSES))
    true_is_static  = np.isin(true_labels, list(STATIC_CLASSES))
    # 未知类别（既不在动态也不在静态集合中）视为静态处理
    true_is_unknown = ~(true_is_dynamic | true_is_static)

    # 初始化预测标签与置信度
    pred_is_dynamic = true_is_dynamic.copy()  # 先假设预测完全正确
    confidence = np.zeros(n, dtype=np.float32)

    # ── 为"正确预测"的点生成高置信度 ──
    # Beta(8,2)：偏右分布，均值 ≈ 0.80，模拟网络对正确预测较为自信
    confidence[:] = rng.beta(8, 2, size=n).astype(np.float32)

    # ── 模拟漏检：动态点被错误预测为静态（False Negative） ──
    # 物理含义：快速移动的行人/车辆有时被误判为背景，5% 概率
    dyn_indices = np.where(true_is_dynamic)[0]
    if len(dyn_indices) > 0:
        fn_mask = rng.random(len(dyn_indices)) < FALSE_NEGATIVE_RATE
        fn_indices = dyn_indices[fn_mask]
        pred_is_dynamic[fn_indices] = False  # 漏检：动态→静态
        # 漏检点的置信度较低，反映网络不确定性
        confidence[fn_indices] = rng.beta(2, 5, size=len(fn_indices)).astype(np.float32)

    # ── 模拟误检：静态点被错误预测为动态（False Positive） ──
    # 物理含义：路边停放的车辆有时被误判为移动车辆，3% 概率
    sta_indices = np.where(true_is_static | true_is_unknown)[0]
    if len(sta_indices) > 0:
        fp_mask = rng.random(len(sta_indices)) < FALSE_POSITIVE_RATE
        fp_indices = sta_indices[fp_mask]
        pred_is_dynamic[fp_indices] = True   # 误检：静态→动态
        # 误检点的置信度也较低
        confidence[fp_indices] = rng.beta(2, 5, size=len(fp_indices)).astype(np.float32)

    return pred_is_dynamic, confidence


# ─────────────────────── 自适应权重公式 ───────────────────────────

def compute_adaptive_weights(pred_is_dynamic: np.ndarray,
                              confidence: np.ndarray) -> np.ndarray:
    """
    根据预测类别和置信度计算每个点的自适应权重。

    权重公式设计直觉：
      ┌─ 预测为动态的点 ────────────────────────────────────────────────┐
      │  w = conf × 0.05 + (1 - conf) × 1.0                           │
      │  • conf → 1（非常确定是动态）：w → 0.05，几乎完全压制该点         │
      │  • conf → 0（很不确定）：w → 1.0，保留完整权重（可能是误检）       │
      └─────────────────────────────────────────────────────────────────┘
      ┌─ 预测为静态的点 ────────────────────────────────────────────────┐
      │  w = conf × 1.0 + (1 - conf) × 0.3                            │
      │  • conf → 1（确定是静态）：w → 1.0，完整保留                     │
      │  • conf → 0（不确定）：w → 0.3，轻微降权（可能是漏检的动态点）     │
      └─────────────────────────────────────────────────────────────────┘

    参数：
      pred_is_dynamic: 预测是否为动态的布尔数组
      confidence:      置信度数组 ∈ [0, 1]

    返回：
      weights: 每个点的权重数组，范围 ∈ [0.05, 1.0]
    """
    weights = np.ones(len(pred_is_dynamic), dtype=np.float32)

    # 预测为动态的点：高置信度→低权重，低置信度→高权重
    dyn_mask = pred_is_dynamic
    weights[dyn_mask] = (confidence[dyn_mask] * 0.05
                         + (1.0 - confidence[dyn_mask]) * 1.0)

    # 预测为静态的点：高置信度→完整权重，低置信度→轻微降权
    sta_mask = ~pred_is_dynamic
    weights[sta_mask] = (confidence[sta_mask] * 1.0
                         + (1.0 - confidence[sta_mask]) * 0.3)

    return weights


# ─────────────────────── 点云序列生成 ───────────────────────────

def generate_fixed_weight_sequence():
    """
    生成固定权重（现有方案）的点云序列。

    物理意义：
      假设语义分割完美，直接删除所有被标注为动态的点（权重=0）。
      这是 run_semantic_kiss.py 中的当前方案。
      输出目录：00_fixed_weight/velodyne/
    """
    print("\n[固定权重方案] 生成点云序列（直接删除动态点）...")
    out_velo = os.path.join(OUT_FIXED, "velodyne")
    os.makedirs(out_velo, exist_ok=True)

    vfiles = sorted(glob.glob(os.path.join(SEQ_DIR, "velodyne", "*.bin")))
    lfiles = sorted(glob.glob(os.path.join(SEQ_DIR, "labels", "*.label")))

    if not vfiles:
        print(f"  警告：未找到点云文件，跳过（路径：{SEQ_DIR}/velodyne/）")
        return False

    total_removed = 0
    for i, (vf, lf) in enumerate(zip(vfiles, lfiles)):
        pts    = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(lf, dtype=np.uint32) & 0xFFFF

        # 直接删除真值动态点（假设语义分割完美无误）
        keep_mask = ~np.isin(labels, list(DYNAMIC_CLASSES))
        pts_filtered = pts[keep_mask]
        pts_filtered.tofile(os.path.join(out_velo, os.path.basename(vf)))

        removed = (~keep_mask).sum()
        total_removed += removed
        if i % 500 == 0:
            print(f"  帧 {i:04d}/{len(vfiles)}: 删除 {removed} 个动态点")

    print(f"  完成！共删除 {total_removed} 个动态点")

    # 复制辅助文件（标定/时间戳/位姿）
    for fname in ['calib.txt', 'times.txt', 'poses.txt']:
        src = os.path.join(SEQ_DIR, fname)
        dst = os.path.join(OUT_FIXED, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
    return True


def generate_adaptive_weight_sequence():
    """
    生成自适应置信度权重的点云序列（新方案）。

    物理意义：
      模拟真实网络存在误检/漏检的场景。对每个点计算置信度自适应权重：
        - 高置信度的预测动态点：接近删除（权重≈0.05）
        - 低置信度的预测动态点（可能误检）：保留（权重→1.0），不删除
        - 静态点：根据置信度给完整或轻微降权

      关键决策：置信度 < 0.5 的预测动态点不删除（保留进点云），
      通过修改 intensity 通道（pts[:,3]）传递权重信息给后续分析。
      输出目录：00_adaptive_weight/velodyne/
    """
    print("\n[自适应权重方案] 生成点云序列（置信度加权，保留低置信度疑似动态点）...")
    out_velo = os.path.join(OUT_ADAPTIVE, "velodyne")
    os.makedirs(out_velo, exist_ok=True)

    vfiles = sorted(glob.glob(os.path.join(SEQ_DIR, "velodyne", "*.bin")))
    lfiles = sorted(glob.glob(os.path.join(SEQ_DIR, "labels", "*.label")))

    if not vfiles:
        print(f"  警告：未找到点云文件，跳过（路径：{SEQ_DIR}/velodyne/）")
        return False

    rng = np.random.default_rng(RANDOM_SEED)

    total_kept_dynamic = 0   # 本应删除但因低置信度保留的点数
    total_removed      = 0   # 高置信度动态点（被删除）

    for i, (vf, lf) in enumerate(zip(vfiles, lfiles)):
        pts    = np.fromfile(vf, dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(lf, dtype=np.uint32) & 0xFFFF

        # 模拟网络预测：在真值基础上加入误检/漏检噪声
        pred_is_dynamic, confidence = simulate_rangenet_predictions(labels, rng)

        # 计算自适应权重
        weights = compute_adaptive_weights(pred_is_dynamic, confidence)

        # ── 过滤策略 ──
        # 置信度 >= 0.5 的预测动态点：高置信度，直接删除（权重极低，不值得保留）
        # 置信度 <  0.5 的预测动态点：低置信度，保留（可能误检，用权重降权代替删除）
        high_conf_dynamic = pred_is_dynamic & (confidence >= 0.5)
        low_conf_dynamic  = pred_is_dynamic & (confidence <  0.5)

        keep_mask = ~high_conf_dynamic  # 保留所有非高置信度动态点

        pts_out = pts[keep_mask].copy()

        # 将权重信息编码到 intensity 通道（第4列）
        # 后续 SLAM 系统可读取此通道用于加权最小二乘
        pts_out[:, 3] = weights[keep_mask]

        pts_out.tofile(os.path.join(out_velo, os.path.basename(vf)))

        removed = high_conf_dynamic.sum()
        kept_dyn = low_conf_dynamic.sum()
        total_removed      += removed
        total_kept_dynamic += kept_dyn

        if i % 500 == 0:
            print(f"  帧 {i:04d}/{len(vfiles)}: 删除高置信度动态点 {removed}，"
                  f"保留低置信度疑似动态点 {kept_dyn}（降权处理）")

    print(f"  完成！共删除 {total_removed} 个高置信度动态点，"
          f"保留并降权 {total_kept_dynamic} 个低置信度点")

    # 复制辅助文件
    for fname in ['calib.txt', 'times.txt', 'poses.txt']:
        src = os.path.join(SEQ_DIR, fname)
        dst = os.path.join(OUT_ADAPTIVE, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
    return True


# ─────────────────────── ATE 计算函数 ───────────────────────────

def load_poses(poses_file: str) -> np.ndarray:
    """
    读取 KITTI 格式的位姿文件，返回 Nx4x4 的变换矩阵数组。

    物理意义：
      每行 12 个数字，表示 3×4 变换矩阵（旋转+平移），
      最后一行补 [0,0,0,1] 构成齐次变换矩阵。
    """
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(T)
    return np.array(poses)


def compute_ate(gt_poses: np.ndarray, est_poses: np.ndarray) -> float:
    """
    计算绝对轨迹误差（Absolute Trajectory Error, ATE）。

    物理意义：
      ATE 衡量估计轨迹与真值轨迹在每个时刻的平移误差的均方根（RMSE）。
      公式：ATE = sqrt(1/N * Σ ||t_est_i - t_gt_i||²)
      其中 t_est_i 和 t_gt_i 分别是第 i 帧的估计和真值平移向量。

    注意：这里假设位姿已在同一坐标系下对齐（第一帧对齐）。
    """
    n = min(len(gt_poses), len(est_poses))
    # 以第一帧为参考对齐
    T0_gt  = gt_poses[0]
    T0_est = est_poses[0]
    errors = []
    for i in range(n):
        # 相对位姿：从第0帧到第i帧
        rel_gt  = np.linalg.inv(T0_gt)  @ gt_poses[i]
        rel_est = np.linalg.inv(T0_est) @ est_poses[i]
        # 平移误差
        t_err = rel_gt[:3, 3] - rel_est[:3, 3]
        errors.append(np.linalg.norm(t_err))
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def compare_ate():
    """
    对比三组 ATE：基线（无过滤）、固定权重（删除动态点）、自适应权重。

    物理意义：
      加载三个序列各自的 SLAM 估计位姿（poses.txt），
      与真值位姿对比，计算 ATE 并输出对比表格。

    注意：本函数假设 SLAM 已经运行完毕，输出了 poses.txt。
          若文件不存在则跳过并提示用户先运行 SLAM。
    """
    print("\n" + "=" * 55)
    print("ATE 对比结果（需要先运行 SLAM 生成 poses.txt）")
    print("=" * 55)

    gt_file = os.path.join(SEQ_DIR, "poses.txt")
    if not os.path.exists(gt_file):
        print(f"  警告：未找到真值位姿 {gt_file}，跳过 ATE 计算")
        return None

    gt_poses = load_poses(gt_file)

    results = {}
    configs = [
        ("基线（无过滤）",   SEQ_DIR,    "baseline"),
        ("固定权重（w=0）",  OUT_FIXED,   "fixed"),
        ("自适应权重（新）", OUT_ADAPTIVE, "adaptive"),
    ]

    for name, seq_dir, key in configs:
        est_file = os.path.join(seq_dir, "poses_est.txt")  # SLAM 输出
        if not os.path.exists(est_file):
            print(f"  {name}: 未找到 {est_file}，跳过")
            results[key] = None
            continue
        est_poses = load_poses(est_file)
        ate = compute_ate(gt_poses, est_poses)
        results[key] = ate
        print(f"  {name}: ATE = {ate:.4f} m")

    print("=" * 55)

    # 输出结构化表格
    if any(v is not None for v in results.values()):
        print(f"\n{'方案':<20} {'ATE (m)':<12} {'相对基线':<12}")
        print("-" * 44)
        baseline_ate = results.get("baseline")
        for name, _, key in configs:
            ate = results.get(key)
            if ate is None:
                ate_str = "N/A"
                rel_str = "N/A"
            else:
                ate_str = f"{ate:.4f}"
                if baseline_ate is not None:
                    rel_str = f"{(ate - baseline_ate) / baseline_ate * 100:+.1f}%"
                else:
                    rel_str = "N/A"
            print(f"  {name:<18} {ate_str:<12} {rel_str:<12}")
        print("-" * 44)

    return results


# ─────────────────────── 理论特性分析图 ───────────────────────────

def plot_adaptive_weight_theory(save_path: str = None):
    """
    生成置信度自适应权重的理论特性分析图。

    图表内容：
      - 主图：x 轴=置信度(0~1)，y 轴=点权重(0~1)
        · 蓝色曲线：预测为动态的点的权重（w = conf×0.05 + (1-conf)×1.0）
        · 绿色曲线：预测为静态的点的权重（w = conf×1.0 + (1-conf)×0.3）
      - 辅图：模拟置信度分布（正确预测 vs 错误预测）
      - 辅图：不同置信度区间下动态点权重分布箱线图

    物理直觉说明：
      自适应权重的核心思想是"用置信度加权不确定性"：
        1. 高置信度动态点（conf>0.8）：网络非常确定这是动态物体，
           大胆压制（w≈0.1），减少动态物体对 SLAM 的干扰。
        2. 低置信度动态点（conf<0.4）：网络不确定，可能是误检，
           保留较高权重（w≈0.6~1.0），避免把有用的静态点删掉。
        3. 不确定区域（conf∈[0.4,0.8]）：线性插值过渡，
           平滑处理不同置信度的点。
    """
    conf = np.linspace(0, 1, 500)

    # 自适应权重公式
    w_dynamic = conf * 0.05 + (1 - conf) * 1.0   # 预测为动态
    w_static  = conf * 1.0  + (1 - conf) * 0.3   # 预测为静态
    w_fixed   = np.full_like(conf, 0.025)         # 固定权重（参考线，平均≈0）

    # ── 创建图布局 ──
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle('置信度自适应权重方案 — 理论特性分析\n'
                 'Confidence-Adaptive Weighting for Semantic SLAM',
                 fontsize=15, color='white', fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.90, bottom=0.10)

    ax_main  = fig.add_subplot(gs[:, 0:2])   # 左侧大图（主图）
    ax_dist  = fig.add_subplot(gs[0, 2])     # 右上：置信度分布
    ax_box   = fig.add_subplot(gs[1, 2])     # 右下：权重分布

    panel_color = '#16213e'

    # ── 主图：权重 vs 置信度 ──
    ax_main.set_facecolor(panel_color)
    ax_main.plot(conf, w_dynamic, color='#e94560', linewidth=2.5,
                 label='预测为动态的点  w = conf×0.05 + (1-conf)×1.0')
    ax_main.plot(conf, w_static,  color='#0f3460', linewidth=2.5,
                 linestyle='--',
                 label='预测为静态的点  w = conf×1.0 + (1-conf)×0.3',
                 path_effects=None)
    ax_main.plot(conf, w_static,  color='#4ecca3', linewidth=2.5,
                 linestyle='--',
                 label='预测为静态的点  w = conf×1.0 + (1-conf)×0.3')

    # 固定权重参考线（现有方案）
    ax_main.axhline(y=0.05, color='#f5a623', linewidth=1.5, linestyle=':',
                    alpha=0.8, label='现有固定权重下界（w=0.05）')

    # 标注关键区域
    ax_main.axvspan(0, 0.5, alpha=0.08, color='#e94560',
                    label='低置信度区（保留疑似误检点）')
    ax_main.axvspan(0.5, 1.0, alpha=0.06, color='#4ecca3',
                    label='高置信度区（信任网络判断）')

    # 标注关键点
    ax_main.annotate('conf=0.5\n动态点权重=0.525\n(中间过渡)',
                     xy=(0.5, 0.525), xytext=(0.35, 0.72),
                     fontsize=8.5, color='#e94560',
                     arrowprops=dict(arrowstyle='->', color='#e94560', lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', ec='#e94560', alpha=0.8))
    ax_main.annotate('conf→1\n动态点权重→0.05\n(几乎完全压制)',
                     xy=(0.95, 0.098), xytext=(0.72, 0.22),
                     fontsize=8.5, color='#e94560',
                     arrowprops=dict(arrowstyle='->', color='#e94560', lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', ec='#e94560', alpha=0.8))
    ax_main.annotate('conf→0\n动态点权重→1.0\n(完整保留，防止误删)',
                     xy=(0.05, 0.952), xytext=(0.12, 0.75),
                     fontsize=8.5, color='#ff6b6b',
                     arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.2),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', ec='#ff6b6b', alpha=0.8))

    ax_main.set_xlabel('置信度 (Confidence)', color='white', fontsize=12)
    ax_main.set_ylabel('点权重 (Point Weight)', color='white', fontsize=12)
    ax_main.set_title('置信度 → 权重映射曲线', color='white', fontsize=12, pad=10)
    ax_main.tick_params(colors='white')
    ax_main.spines[:].set_color('#444')
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(-0.02, 1.08)
    ax_main.grid(alpha=0.2, color='white')
    ax_main.legend(fontsize=8.5, loc='center right',
                   facecolor='#1a1a2e', edgecolor='#444',
                   labelcolor='white', framealpha=0.9)

    # 添加设计直觉文字框
    intuition_text = (
        "设计直觉：\n"
        "• 高置信度动态点（conf→1）：\n"
        "  网络非常确定→大胆压制（w→0.05）\n\n"
        "• 低置信度动态点（conf→0）：\n"
        "  可能是误检→保留权重（w→1.0）\n\n"
        "• 静态点始终保持高权重（0.3~1.0）\n"
        "  防止漏检动态点干扰SLAM"
    )
    ax_main.text(0.01, 0.02, intuition_text, transform=ax_main.transAxes,
                 fontsize=8, color='#aaaaaa', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='#0d0d1a', ec='#333', alpha=0.9))

    # ── 右上图：模拟置信度分布 ──
    ax_dist.set_facecolor(panel_color)
    rng_plot = np.random.default_rng(42)
    conf_correct = rng_plot.beta(8, 2, 3000)   # 正确预测：高置信度
    conf_wrong   = rng_plot.beta(2, 5, 300)    # 错误预测：低置信度

    ax_dist.hist(conf_correct, bins=40, alpha=0.7, color='#4ecca3',
                 density=True, label=f'正确预测 (n≈{len(conf_correct)})')
    ax_dist.hist(conf_wrong,   bins=40, alpha=0.7, color='#e94560',
                 density=True, label=f'误检/漏检 (n≈{len(conf_wrong)})')
    ax_dist.axvline(0.5, color='#f5a623', linestyle='--', linewidth=1.5,
                    label='决策阈值 conf=0.5')
    ax_dist.set_xlabel('置信度', color='white', fontsize=9)
    ax_dist.set_ylabel('概率密度', color='white', fontsize=9)
    ax_dist.set_title('模拟置信度分布\n(RangeNet++ 网络误差模拟)',
                      color='white', fontsize=9, pad=6)
    ax_dist.tick_params(colors='white', labelsize=8)
    ax_dist.spines[:].set_color('#444')
    ax_dist.legend(fontsize=7.5, facecolor='#1a1a2e', edgecolor='#444',
                   labelcolor='white', framealpha=0.8)
    ax_dist.grid(alpha=0.2, color='white')

    # ── 右下图：不同置信度区间的动态点权重分布 ──
    ax_box.set_facecolor(panel_color)
    bins_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels  = ['0~0.2', '0.2~0.4', '0.4~0.6', '0.6~0.8', '0.8~1.0']
    colors_box  = ['#ff4d4d', '#ff9933', '#ffdd57', '#66bb6a', '#42a5f5']

    box_data = []
    for lo, hi in zip(bins_edges[:-1], bins_edges[1:]):
        c_sample = np.random.default_rng(42).uniform(lo, hi, 200)
        w_sample = c_sample * 0.05 + (1 - c_sample) * 1.0
        box_data.append(w_sample)

    bp = ax_box.boxplot(box_data, patch_artist=True, labels=bin_labels,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#aaa'),
                        capprops=dict(color='#aaa'),
                        flierprops=dict(marker='o', color='#aaa', markersize=2))
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax_box.set_xlabel('置信度区间', color='white', fontsize=9)
    ax_box.set_ylabel('动态点权重', color='white', fontsize=9)
    ax_box.set_title('各置信度区间的动态点权重分布',
                      color='white', fontsize=9, pad=6)
    ax_box.tick_params(colors='white', labelsize=7.5)
    ax_box.spines[:].set_color('#444')
    ax_box.grid(alpha=0.2, color='white', axis='y')

    # ── 保存 ──
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"\n分析图已保存至：{save_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig


# ─────────────────────── 主入口 ───────────────────────────

if __name__ == "__main__":
    """
    主入口：
      1. 首先只生成理论分析图（不依赖 KITTI 数据），展示置信度自适应权重的理论特性。
      2. 如果 KITTI 数据存在，则额外生成两个点云序列并对比 ATE。
    """
    import sys

    print("=" * 60)
    print("  置信度自适应权重方案 — Confidence Adaptive Weight")
    print("=" * 60)

    # ── 步骤1：生成理论特性分析图（不依赖 KITTI 数据）──
    print("\n[步骤 1] 生成置信度自适应权重理论特性分析图...")
    plot_adaptive_weight_theory(save_path=RESULT_FIG)

    # ── 步骤2：检查 KITTI 数据是否存在 ──
    velo_dir = os.path.join(SEQ_DIR, "velodyne")
    label_dir = os.path.join(SEQ_DIR, "labels")

    if not os.path.isdir(velo_dir) or not os.path.isdir(label_dir):
        print(f"\n[步骤 2] 未找到 KITTI 数据（{SEQ_DIR}），跳过点云序列生成和 ATE 对比。")
        print("  若需完整对比，请先下载 KITTI Sequence 00 数据并运行 SLAM。")
        print("\n完成！（仅生成了理论分析图）")
        sys.exit(0)

    # ── 步骤3：生成固定权重点云序列（现有方案）──
    print("\n[步骤 2] 生成固定权重点云序列（现有方案：直接删除动态点）...")
    ok_fixed = generate_fixed_weight_sequence()

    # ── 步骤4：生成自适应权重点云序列（新方案）──
    print("\n[步骤 3] 生成自适应置信度权重点云序列（新方案）...")
    ok_adaptive = generate_adaptive_weight_sequence()

    # ── 步骤5：对比 ATE ──
    print("\n[步骤 4] 对比三组 ATE（需先运行 SLAM 生成 poses_est.txt）...")
    ate_results = compare_ate()

    print("\n" + "=" * 60)
    print("  全部完成！")
    print(f"  · 理论分析图：{RESULT_FIG}")
    print(f"  · 固定权重序列：{OUT_FIXED}/velodyne/")
    print(f"  · 自适应权重序列：{OUT_ADAPTIVE}/velodyne/")
    print("=" * 60)
