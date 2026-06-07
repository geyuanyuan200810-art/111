"""
semantic_octomap.py
===================
语义八叉树地图（用纯 numpy + 字典实现，不依赖 octomap 库）

核心思想
--------
传统点云地图把所有点直接堆叠，导致动态物体（行人、车辆）在地图中
留下"鬼影"（ghost artifacts）。本模块用 log-odds 概率框架解决这个问题：

  log-odds(p) = log(p / (1-p))

每次观测到某体素被占据，就给它加一个正值；被射线穿过（空闲），就减去一个值。
对动态点（语义标签属于可移动类别）使用：
  - 命中时加更小的正值（置信度低，动态物体不该留在地图里）
  - 未命中时减更大的负值（快速衰减，消除鬼影）

这样，动态物体经过后，对应体素的 log-odds 会迅速跌到阈值以下，
从占据状态变为空闲，鬼影自然消除。
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from collections import defaultdict

# ─────────────────────────── 数据结构 ───────────────────────────

@dataclass
class VoxelCell:
    """
    单个体素的状态。

    log_odds: float
        占据概率的 log-odds 表示。
        > 0.5  → 认为该体素被占据（有障碍物）
        < -0.5 → 认为该体素空闲
        介于两者之间 → 不确定

    semantic_label: int
        该体素中观测频率最高的语义类别 ID。
        KITTI 语义标签中动态类别（行人=11,两轮车=13,小车=26-28等）会触发快速衰减。

    hit_count: int
        被点云命中的总次数（用于统计）

    dynamic_hit_count: int
        被动态点命中的次数（动态物体经过次数）

    last_update_time: int
        最近一次更新的帧序号，用于计算"多少帧未更新"，
        超过阈值的动态体素会额外衰减。
    """
    log_odds: float = 0.0
    semantic_label: int = 0
    hit_count: int = 0
    dynamic_hit_count: int = 0
    last_update_time: int = 0
    # 用于统计语义标签众数
    _label_counts: Dict[int, int] = field(default_factory=dict)


# ─────────────────────────── 核心类 ───────────────────────────

# KITTI 语义中属于动态类别的标签 ID
# 参考 SemanticKITTI 标注定义
DYNAMIC_LABELS = {
    11,  # person / pedestrian
    13,  # bicyclist
    15,  # motorcyclist
    18,  # truck（通常停着，但语义上可移动）
    20,  # other-vehicle
    252, # moving-car
    253, # moving-bicyclist
    254, # moving-person
    255, # moving-motorcyclist
    256, # moving-on-rails
    257, # moving-bus
    258, # moving-truck
    259, # moving-other-vehicle
    # 以小车为主体的 label（SemanticKITTI compact id）
    26, 27, 28, 29, 30, 31,
}


class SemanticVoxelMap:
    """
    语义体素占据地图。

    用字典存储体素，key = (ix, iy, iz)（整数体素坐标），
    value = VoxelCell。

    Parameters
    ----------
    resolution : float
        体素边长（米），默认 0.2 m
    log_odds_hit_static : float
        静态点命中时的 log-odds 增量（正值）
    log_odds_miss_static : float
        静态点射线穿越时的 log-odds 减量（负值，填入正数后内部取反）
    log_odds_hit_dynamic : float
        动态点命中时的 log-odds 增量（小正值，体现低置信度）
    log_odds_miss_dynamic : float
        动态点未命中时的 log-odds 减量（大负值，快速衰减）
    occ_threshold : float
        log-odds 超过此值才认为体素被占据
    free_threshold : float
        log-odds 低于此值认为体素空闲（负值）
    dynamic_decay_interval : int
        超过多少帧未更新的动态体素，每帧额外衰减
    dynamic_decay_extra : float
        额外衰减量（负值，填入正数后内部取反）
    """

    def __init__(
        self,
        resolution: float = 0.2,
        log_odds_hit_static: float = 0.85,
        log_odds_miss_static: float = 0.4,
        log_odds_hit_dynamic: float = 0.3,
        log_odds_miss_dynamic: float = 0.85,
        occ_threshold: float = 0.5,
        free_threshold: float = -0.5,
        dynamic_decay_interval: int = 20,
        dynamic_decay_extra: float = 0.1,
        use_dynamic_decay: bool = True,  # False → baseline 模式
    ):
        self.resolution = resolution
        self.lo_hit_s = log_odds_hit_static
        self.lo_miss_s = -log_odds_miss_static
        self.lo_hit_d = log_odds_hit_dynamic
        self.lo_miss_d = -log_odds_miss_dynamic
        self.occ_threshold = occ_threshold
        self.free_threshold = free_threshold
        self.dynamic_decay_interval = dynamic_decay_interval
        self.dynamic_decay_extra = -dynamic_decay_extra
        self.use_dynamic_decay = use_dynamic_decay

        # 体素字典：(ix,iy,iz) → VoxelCell
        self.voxels: Dict[Tuple[int, int, int], VoxelCell] = {}

        # 每帧动态体素统计（用于绘制衰减曲线）
        self.frame_dynamic_counts: List[int] = []

        # log-odds 上下限（防止数值溢出）
        self.lo_max = 3.5
        self.lo_min = -3.5

    # ── 坐标转换 ──────────────────────────────────────────────

    def _to_voxel_key(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """将世界坐标转换为体素整数索引"""
        ix = int(np.floor(x / self.resolution))
        iy = int(np.floor(y / self.resolution))
        iz = int(np.floor(z / self.resolution))
        return (ix, iy, iz)

    def _get_or_create(self, key: Tuple[int, int, int]) -> VoxelCell:
        if key not in self.voxels:
            self.voxels[key] = VoxelCell()
        return self.voxels[key]

    # ── log-odds 更新 ──────────────────────────────────────────

    def _update_cell(self, key, delta_lo: float, label: int, frame_id: int):
        """更新单个体素的 log-odds 和语义标签"""
        cell = self._get_or_create(key)
        cell.log_odds = np.clip(cell.log_odds + delta_lo, self.lo_min, self.lo_max)
        cell.last_update_time = frame_id
        # 更新语义标签众数
        cell._label_counts[label] = cell._label_counts.get(label, 0) + 1
        cell.semantic_label = max(cell._label_counts, key=cell._label_counts.get)

    # ── 射线投射（Bresenham 3D 简化版）──────────────────────────

    def _raycast_miss(self, origin: np.ndarray, endpoint: np.ndarray,
                      is_dynamic: bool, frame_id: int, label: int):
        """
        从传感器原点到观测点之间，沿射线方向标记"空闲"体素。
        这是 OctoMap 的核心机制：被射线穿过的体素 → 空闲概率增加。
        为节省计算，只取射线上均匀采样的几个点。
        """
        delta = endpoint - origin
        length = np.linalg.norm(delta)
        if length < 1e-6:
            return
        # 沿射线每隔 resolution 取一个采样点（最多取到终点前一个体素）
        n_steps = max(1, int(length / self.resolution) - 1)
        if n_steps == 0:
            return
        lo_miss = self.lo_miss_d if is_dynamic else self.lo_miss_s
        for i in range(1, n_steps + 1):
            t = i / (n_steps + 1)
            pt = origin + t * delta
            key = self._to_voxel_key(*pt)
            self._update_cell(key, lo_miss, label, frame_id)

    # ── 插入一帧点云 ──────────────────────────────────────────

    def insert_frame(
        self,
        points: np.ndarray,       # (N, 3) 世界坐标
        labels: np.ndarray,       # (N,)   语义标签
        sensor_origin: np.ndarray, # (3,)  传感器位置（世界坐标）
        frame_id: int,
        do_raycast: bool = True,
    ):
        """
        插入一帧点云并更新地图。

        对每个点：
        1. 判断是否属于动态类别
        2. 用对应的 log-odds 增量更新命中体素
        3. （可选）沿射线更新空闲体素

        Parameters
        ----------
        do_raycast : bool
            是否执行射线投射。启用时精度更高但更慢。
        """
        if points.shape[0] == 0:
            return

        for i in range(len(points)):
            pt = points[i]
            lbl = int(labels[i]) if labels is not None else 0
            is_dyn = lbl in DYNAMIC_LABELS

            key = self._to_voxel_key(*pt)
            cell = self._get_or_create(key)

            if is_dyn:
                # 动态点：命中置信度低（只加 0.3）
                delta_hit = self.lo_hit_d
                cell.dynamic_hit_count += 1
            else:
                # 静态点：命中置信度高（加 0.85）
                delta_hit = self.lo_hit_s

            cell.hit_count += 1
            self._update_cell(key, delta_hit, lbl, frame_id)

            # 射线投射：沿光路标记空闲（计算量较大，可按需开启）
            if do_raycast:
                self._raycast_miss(sensor_origin, pt, is_dyn, frame_id, lbl)

        # ── 帧结束后：对长时间未更新的动态体素额外衰减 ──────────
        if self.use_dynamic_decay:
            self._apply_dynamic_decay(frame_id)

        # 记录本帧占据动态体素数量（用于绘图）
        self.frame_dynamic_counts.append(self._count_dynamic_occupied())

    def _apply_dynamic_decay(self, current_frame: int):
        """
        额外衰减：超过 dynamic_decay_interval 帧未被更新的动态体素，
        log-odds 再减 dynamic_decay_extra。

        物理意义：如果一个体素之前主要由动态点命中，但很久没有再
        被观测到，说明那个动态物体已经离开，体素应当恢复空闲状态。
        """
        for cell in self.voxels.values():
            age = current_frame - cell.last_update_time
            if age > self.dynamic_decay_interval and cell.dynamic_hit_count > 0:
                cell.log_odds = np.clip(
                    cell.log_odds + self.dynamic_decay_extra,
                    self.lo_min, self.lo_max
                )

    def _count_dynamic_occupied(self) -> int:
        """统计当前地图中仍处于占据状态的动态体素数量（即鬼影数量）"""
        count = 0
        for cell in self.voxels.values():
            if cell.log_odds > self.occ_threshold and cell.dynamic_hit_count > 0:
                count += 1
        return count

    # ── 查询接口 ──────────────────────────────────────────────

    def get_occupied_voxels(self, z_min: float = None, z_max: float = None):
        """
        返回所有占据体素的中心坐标和标签。

        Returns
        -------
        coords : np.ndarray, shape (M, 3)
        labels : np.ndarray, shape (M,)
        is_dynamic : np.ndarray, shape (M,), bool
        """
        coords, labels, is_dyn = [], [], []
        for (ix, iy, iz), cell in self.voxels.items():
            if cell.log_odds > self.occ_threshold:
                cx = (ix + 0.5) * self.resolution
                cy = (iy + 0.5) * self.resolution
                cz = (iz + 0.5) * self.resolution
                if z_min is not None and cz < z_min:
                    continue
                if z_max is not None and cz > z_max:
                    continue
                coords.append([cx, cy, cz])
                labels.append(cell.semantic_label)
                is_dyn.append(cell.dynamic_hit_count > 0)
        if not coords:
            return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, bool)
        return (np.array(coords), np.array(labels, int),
                np.array(is_dyn, bool))

    def summary(self) -> dict:
        """返回地图统计摘要"""
        total = 0
        occ = 0
        dyn_occ = 0
        static_occ = 0
        for cell in self.voxels.values():
            total += 1
            if cell.log_odds > self.occ_threshold:
                occ += 1
                if cell.dynamic_hit_count > 0:
                    dyn_occ += 1
                else:
                    static_occ += 1
        return {
            'total_voxels': total,
            'occupied_voxels': occ,
            'dynamic_ghost_voxels': dyn_occ,
            'static_occupied_voxels': static_occ,
        }


# ─────────────────────────── KITTI 数据加载 ───────────────────────────

def load_kitti_poses(traj_path: str) -> List[np.ndarray]:
    """
    读取轨迹文件（KITTI 格式：每行 12 个数，3×4 变换矩阵）。
    返回 list of 4×4 齐次变换矩阵。
    """
    poses = []
    with open(traj_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


def load_velodyne_bin(bin_path: str) -> np.ndarray:
    """读取 KITTI velodyne .bin 文件，返回 (N, 4) [x,y,z,intensity]"""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pts


def load_semantic_label(label_path: str) -> np.ndarray:
    """
    读取 SemanticKITTI .label 文件。
    每个点对应 uint32，低 16 位是语义标签，高 16 位是实例 ID。
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    semantic = labels & 0xFFFF   # 取低 16 位
    return semantic.astype(np.int32)


def transform_points(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """将局部坐标系点云变换到世界坐标系"""
    N = pts_xyz.shape[0]
    pts_h = np.hstack([pts_xyz, np.ones((N, 1))])   # 齐次坐标
    world = (T @ pts_h.T).T                           # (N, 4)
    return world[:, :3]


# ─────────────────────────── 主流程（KITTI 数据）───────────────────────────

SEQ_PATH = "/home/yuan/kitti_data/dataset/sequences/00"
TRAJ_PATH = "/home/yuan/slam_results/kitti_semantic/traj.txt"
OUT_DIR = "/home/yuan/slam_results"
OUT_PNG = os.path.join(OUT_DIR, "semantic_octomap.png")

def run_kitti_experiment():
    """
    读取 KITTI 序列 00，每隔 5 帧采样，共 100 帧，
    同时建 baseline_map 和 semantic_map，对比鬼影消除效果。
    """
    velodyne_dir = os.path.join(SEQ_PATH, "velodyne")
    label_dir = os.path.join(SEQ_PATH, "labels")

    if not os.path.isdir(velodyne_dir):
        print(f"[ERROR] 找不到 KITTI velodyne 目录: {velodyne_dir}")
        return None, None

    poses = load_kitti_poses(TRAJ_PATH)
    bin_files = sorted([
        f for f in os.listdir(velodyne_dir) if f.endswith('.bin')
    ])

    STEP = 5
    MAX_FRAME = 500
    frames = list(range(0, min(MAX_FRAME, len(bin_files)), STEP))
    print(f"[INFO] 将处理 {len(frames)} 帧（每 {STEP} 帧采样一次，共 {MAX_FRAME} 帧）")

    # baseline：不区分动静，统一用静态参数
    baseline_map = SemanticVoxelMap(
        resolution=0.2,
        use_dynamic_decay=False,   # 不做动态衰减
    )
    # semantic：区分动静，动态点加速衰减
    semantic_map = SemanticVoxelMap(
        resolution=0.2,
        use_dynamic_decay=True,
    )

    for fi, frame_idx in enumerate(frames):
        bin_path = os.path.join(velodyne_dir, bin_files[frame_idx])
        label_path = os.path.join(
            label_dir,
            bin_files[frame_idx].replace('.bin', '.label')
        )

        pts_raw = load_velodyne_bin(bin_path)
        pts_xyz = pts_raw[:, :3]

        if os.path.isfile(label_path):
            labels = load_semantic_label(label_path)
        else:
            labels = np.zeros(len(pts_xyz), dtype=np.int32)

        if frame_idx < len(poses):
            T = poses[frame_idx]
        else:
            T = np.eye(4)

        pts_world = transform_points(pts_xyz, T)
        sensor_origin = T[:3, 3]

        # 距离过滤：只保留 50m 以内的点（减少计算量）
        dist = np.linalg.norm(pts_world - sensor_origin, axis=1)
        mask = dist < 50.0
        pts_world = pts_world[mask]
        labels_f = labels[mask]

        baseline_map.insert_frame(
            pts_world, labels_f, sensor_origin, fi, do_raycast=False
        )
        semantic_map.insert_frame(
            pts_world, labels_f, sensor_origin, fi, do_raycast=False
        )

        if (fi + 1) % 10 == 0:
            print(f"  帧 {fi+1}/{len(frames)} 处理完毕")

    return baseline_map, semantic_map


# ─────────────────────────── 可视化 ───────────────────────────

def visualize_maps(baseline_map: SemanticVoxelMap,
                   semantic_map: SemanticVoxelMap,
                   out_path: str):
    """
    生成 3 子图对比图：
      左图：基线地图俯视图（XY 平面，Z > -0.5），红色标注动态鬼影体素
      中图：语义地图俯视图，动态体素已消除
      右图：动态体素残留数量随帧数的变化曲线
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('语义八叉树地图：动态鬼影消除对比', fontsize=14, fontweight='bold')

    # ── 左图：基线地图 ──────────────────────────────────────
    ax = axes[0]
    coords_b, labels_b, is_dyn_b = baseline_map.get_occupied_voxels(z_min=-0.5)
    if len(coords_b) > 0:
        static_mask = ~is_dyn_b
        dyn_mask = is_dyn_b
        if static_mask.any():
            ax.scatter(coords_b[static_mask, 0], coords_b[static_mask, 1],
                       s=0.3, c='steelblue', alpha=0.5, rasterized=True)
        if dyn_mask.any():
            ax.scatter(coords_b[dyn_mask, 0], coords_b[dyn_mask, 1],
                       s=1.5, c='red', alpha=0.8, rasterized=True,
                       label=f'动态鬼影 ({dyn_mask.sum()}个)')
    ax.set_title('基线地图（鬼影未消除）', fontsize=11)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── 中图：语义地图 ──────────────────────────────────────
    ax = axes[1]
    coords_s, labels_s, is_dyn_s = semantic_map.get_occupied_voxels(z_min=-0.5)
    if len(coords_s) > 0:
        static_mask_s = ~is_dyn_s
        dyn_mask_s = is_dyn_s
        if static_mask_s.any():
            ax.scatter(coords_s[static_mask_s, 0], coords_s[static_mask_s, 1],
                       s=0.3, c='steelblue', alpha=0.5, rasterized=True)
        if dyn_mask_s.any():
            ax.scatter(coords_s[dyn_mask_s, 0], coords_s[dyn_mask_s, 1],
                       s=1.5, c='orange', alpha=0.8, rasterized=True,
                       label=f'残留动态 ({dyn_mask_s.sum()}个)')
    ax.set_title('语义地图（动态鬼影消除后）', fontsize=11)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── 右图：动态体素衰减曲线 ──────────────────────────────
    ax = axes[2]
    x_frames = list(range(len(baseline_map.frame_dynamic_counts)))
    ax.plot(x_frames, baseline_map.frame_dynamic_counts,
            color='red', label='基线地图（无衰减）', linewidth=1.5)
    ax.plot(list(range(len(semantic_map.frame_dynamic_counts))),
            semantic_map.frame_dynamic_counts,
            color='green', label='语义地图（动态衰减）', linewidth=1.5)
    ax.set_title('动态鬼影体素残留数量随帧数变化', fontsize=11)
    ax.set_xlabel('帧序号')
    ax.set_ylabel('动态体素数量（鬼影）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] 可视化图表已保存到: {out_path}")
    plt.close()


def print_summary(baseline_map: SemanticVoxelMap, semantic_map: SemanticVoxelMap):
    """打印定量汇总表"""
    bs = baseline_map.summary()
    ss = semantic_map.summary()

    ghost_b = bs['dynamic_ghost_voxels']
    ghost_s = ss['dynamic_ghost_voxels']

    elim_rate = (ghost_b - ghost_s) / max(ghost_b, 1) * 100
    static_b = bs['static_occupied_voxels']
    static_s = ss['static_occupied_voxels']
    retain_rate = static_s / max(static_b, 1) * 100

    print("\n" + "=" * 60)
    print("          语义八叉树地图定量汇总")
    print("=" * 60)
    print(f"{'指标':<28} {'基线地图':>12} {'语义地图':>12}")
    print("-" * 60)
    print(f"{'总体素数':<28} {bs['total_voxels']:>12,} {ss['total_voxels']:>12,}")
    print(f"{'占据体素数':<28} {bs['occupied_voxels']:>12,} {ss['occupied_voxels']:>12,}")
    print(f"{'动态鬼影体素数':<28} {ghost_b:>12,} {ghost_s:>12,}")
    print(f"{'静态占据体素数':<28} {static_b:>12,} {static_s:>12,}")
    print("-" * 60)
    print(f"{'动态鬼影消除率 (%)':<28} {'—':>12} {elim_rate:>11.1f}%")
    print(f"{'静态体素保留率 (%)':<28} {'—':>12} {retain_rate:>11.1f}%")
    print("=" * 60)


# ─────────────────────────── 演示模式（无 KITTI 数据）───────────────────────────

def run_demo():
    """
    演示模式：不需要 KITTI 数据，直接模拟 log-odds 随时间的变化，
    展示静态体素与动态体素的衰减速度对比。

    场景描述
    --------
    假设某体素在前 30 帧内被频繁命中（有物体在此处），
    之后该物体离开，后续帧只有射线穿过（未命中）。

    对比：
    - 静态模式：命中 +0.85，未命中 -0.4
    - 动态模式：命中 +0.3，未命中 -0.85

    通过这条曲线可以直观看到：动态体素的 log-odds 在物体离开后
    迅速跌到阈值 -0.5 以下，而静态体素则缓慢下降，很难被清除。
    这正是"鬼影消除"的核心机制。
    """

    n_frames = 80
    frames = np.arange(n_frames)

    # ── 模拟 log-odds 序列 ──────────────────────────────────
    LO_MAX, LO_MIN = 3.5, -3.5
    OCC_THRESH = 0.5
    FREE_THRESH = -0.5

    def simulate(hit_frames, miss_frames, lo_hit, lo_miss,
                 extra_decay_frames=None, extra_decay=-0.1):
        lo = 0.0
        history = []
        for t in frames:
            if t in hit_frames:
                lo = np.clip(lo + lo_hit, LO_MIN, LO_MAX)
            elif t in miss_frames:
                lo = np.clip(lo + lo_miss, LO_MIN, LO_MAX)
            if extra_decay_frames and t in extra_decay_frames:
                lo = np.clip(lo + extra_decay, LO_MIN, LO_MAX)
            history.append(lo)
        return np.array(history)

    # 前 30 帧命中，之后每帧未命中（物体离开后射线穿越）
    hit_set = set(range(0, 30))
    miss_set = set(range(30, n_frames))
    # 动态额外衰减：30 帧之后，每帧多减 0.1
    extra_set = set(range(50, n_frames))   # 超过 20 帧未更新后触发

    lo_static  = simulate(hit_set, miss_set, lo_hit=+0.85, lo_miss=-0.4)
    lo_dynamic = simulate(hit_set, miss_set, lo_hit=+0.3,  lo_miss=-0.85,
                          extra_decay_frames=extra_set, extra_decay=-0.1)

    # ── 绘图 ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Log-Odds 动态体素衰减演示（无需 KITTI 数据）',
                 fontsize=13, fontweight='bold')

    # 左图：log-odds 随时间变化
    ax = axes[0]
    ax.plot(frames, lo_static,  color='steelblue', linewidth=2,
            label='静态体素（hit+0.85 / miss-0.40）')
    ax.plot(frames, lo_dynamic, color='tomato', linewidth=2,
            label='动态体素（hit+0.30 / miss-0.85 + 额外-0.10）')
    ax.axhline(OCC_THRESH,  color='gray', linestyle='--', alpha=0.7,
               label=f'占据阈值 ({OCC_THRESH})')
    ax.axhline(FREE_THRESH, color='gray', linestyle=':',  alpha=0.7,
               label=f'空闲阈值 ({FREE_THRESH})')
    ax.axvline(30, color='orange', linestyle='-.',  alpha=0.7,
               label='物体离开（帧 30）')
    ax.fill_between(frames, FREE_THRESH, OCC_THRESH,
                    alpha=0.05, color='yellow', label='不确定区间')
    ax.set_xlabel('帧序号')
    ax.set_ylabel('Log-Odds')
    ax.set_title('log-odds 随时间变化对比')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_frames - 1)

    # 右图：对应的占据概率 p = sigmoid(log_odds)
    ax = axes[1]
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    ax.plot(frames, sigmoid(lo_static),  color='steelblue', linewidth=2,
            label='静态体素占据概率')
    ax.plot(frames, sigmoid(lo_dynamic), color='tomato', linewidth=2,
            label='动态体素占据概率')
    ax.axhline(sigmoid(OCC_THRESH),  color='gray', linestyle='--', alpha=0.7,
               label=f'占据阈值 (p≈{sigmoid(OCC_THRESH):.2f})')
    ax.axhline(sigmoid(FREE_THRESH), color='gray', linestyle=':',  alpha=0.7,
               label=f'空闲阈值 (p≈{sigmoid(FREE_THRESH):.2f})')
    ax.axvline(30, color='orange', linestyle='-.', alpha=0.7,
               label='物体离开（帧 30）')

    # 标注动态体素清除时刻
    for t in range(n_frames):
        if lo_dynamic[t] < FREE_THRESH:
            ax.annotate(f'动态体素\n已清除(帧{t})',
                        xy=(t, sigmoid(lo_dynamic[t])),
                        xytext=(t + 3, 0.35),
                        arrowprops=dict(arrowstyle='->', color='tomato'),
                        fontsize=8, color='tomato')
            break

    ax.set_xlabel('帧序号')
    ax.set_ylabel('占据概率 P(occupied)')
    ax.set_title('占据概率随时间变化（sigmoid 转换）')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    demo_path = "/home/user/111/slam_results/semantic_octomap_demo.png"
    plt.savefig(demo_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] 演示图已保存到: {demo_path}")
    plt.close()

    # 文字说明
    print("\n" + "=" * 65)
    print("  Log-Odds 演示说明")
    print("=" * 65)
    print("场景：某体素在帧 0~29 被频繁命中（有物体），帧 30 后物体离开。")
    print()
    print("静态体素（蓝线）：")
    print("  命中 +0.85 → 快速升至上限；未命中仅 -0.4 → 缓慢下降。")
    print("  物体离开后很长时间内 log-odds 仍 > 0.5，体素保持'占据'")
    print("  状态 → 这就是'鬼影'。")
    print()
    print("动态体素（红线）：")
    print("  命中 +0.3 → 积累较慢，峰值较低；未命中 -0.85 → 快速下降。")
    print("  超过 20 帧未更新后额外 -0.1/帧。")
    print("  物体离开后 log-odds 迅速跌破 -0.5，体素恢复'空闲'。")
    print("  → 鬼影自动消除。")
    print("=" * 65)


# ─────────────────────────── 入口 ───────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="语义八叉树地图演示与实验")
    parser.add_argument(
        '--mode', choices=['demo', 'kitti', 'both'], default='demo',
        help='demo: 仅演示log-odds曲线（无需KITTI数据）\n'
             'kitti: 运行KITTI实验\n'
             'both: 先demo后kitti'
    )
    args = parser.parse_args()

    if args.mode in ('demo', 'both'):
        print("[INFO] 运行演示模式（log-odds 衰减对比）...")
        run_demo()

    if args.mode in ('kitti', 'both'):
        print("\n[INFO] 运行 KITTI 实验（序列 00，前 500 帧，STEP=5）...")
        baseline_map, semantic_map = run_kitti_experiment()
        if baseline_map is not None and semantic_map is not None:
            print_summary(baseline_map, semantic_map)
            visualize_maps(baseline_map, semantic_map, OUT_PNG)
        else:
            print("[WARN] KITTI 数据不可用，跳过实验。")

    if args.mode == 'demo':
        # 默认 demo 模式：同时展示一个用合成数据模拟的小地图对比图
        print("\n[INFO] 生成合成地图对比图（模拟场景）...")

        def make_synthetic_maps():
            """用随机合成数据快速演示两张地图的差异"""
            rng = np.random.default_rng(42)
            bmap = SemanticVoxelMap(resolution=0.5, use_dynamic_decay=False)
            smap = SemanticVoxelMap(resolution=0.5, use_dynamic_decay=True)

            # 静态障碍物（标签 0=road, 50=building 等静态类别）
            static_pts = rng.uniform(-20, 20, (2000, 3))
            static_pts[:, 2] = rng.uniform(-0.5, 1.5, 2000)
            static_labels = np.zeros(2000, dtype=np.int32)

            # 动态物体（标签 252=moving-car）在地图中央移动
            dyn_labels_val = 252
            sensor = np.array([0.0, 0.0, 0.5])

            for fi in range(60):
                # 静态点每帧都在
                bmap.insert_frame(static_pts, static_labels, sensor, fi, do_raycast=False)
                smap.insert_frame(static_pts, static_labels, sensor, fi, do_raycast=False)

                if fi < 20:
                    # 动态物体只在前 20 帧出现
                    dyn_pts = rng.normal([5.0, 3.0, 0.5], 0.5, (100, 3))
                    dyn_labels = np.full(100, dyn_labels_val, dtype=np.int32)
                    bmap.insert_frame(dyn_pts, dyn_labels, sensor, fi, do_raycast=False)
                    smap.insert_frame(dyn_pts, dyn_labels, sensor, fi, do_raycast=False)

            return bmap, smap

        bmap, smap = make_synthetic_maps()
        print_summary(bmap, smap)
        synth_out = "/home/user/111/slam_results/semantic_octomap_synth.png"
        visualize_maps(bmap, smap, synth_out)
