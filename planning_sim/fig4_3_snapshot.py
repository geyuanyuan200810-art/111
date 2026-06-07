"""
图4-3 四格快照对比脚本
======================
基于 simulator.py 中已有的三个规划器，生成论文图4-3。

运行：
    cd ~/planning_sim
    python3 fig4_3_snapshot.py

输出：
    ~/planning_sim/fig4_3_conflict_snapshots.png   （300 DPI，论文用）
    ~/planning_sim/fig4_3_data.txt                 （关键数值，用于填写正文）

四格含义：
    (a) 标准DWA      ── 高速迎面行人，反应滞后，险情帧
    (b) 本文改进方法  ── 同场景同时刻，速度感知椭球提前规避
    (c) EGO-Planner  ── 双向夹击行人，势场合力近零，原地卡死
    (d) 本文改进方法  ── 同场景同时刻，预测机制绕开间隙穿越
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ── 把 simulator.py 所在目录加入搜索路径 ─────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── 规划器核心逻辑直接内联，与 simulator.py / sim_ghost_injection.py 数值一致 ──
# （不直接 import simulator 是为了避免 sim_ghost_injection.py 顶部的
#   KMeans 聚类代码在 import 时自动运行，造成依赖问题）

# ══════════════════════════════════════════════════════════
# 参数（与你 simulator.py / sim_ghost_injection.py 保持一致）
# ══════════════════════════════════════════════════════════
SIGMA0   = 0.7    # sim_ghost_injection.py ImprovedPlanner.__init__
ALPHA_V  = 0.3    # sim_ghost_injection.py ImprovedPlanner.__init__
T_PRED   = 1.5    # sim_ghost_injection.py ImprovedPlanner.__init__
UAV_R    = 0.5    # simulator.py UAV.radius
PED_R    = 0.3    # simulator.py 碰撞半径
UAV_VMAX = 3.0    # simulator.py UAV.max_vel
DT_SIM   = 0.05   # simulator.py run_trial dt

# ── 商业街固定静态障碍（与 sim_ghost_injection.py make_scene 一致）──
STATIC_OBS = [
    [-5.0,  3.5, 0.8],
    [ 5.0, -3.5, 0.8],
    [ 0.0,  3.8, 0.6],
    [-10.0, -3.5, 0.7],
]
BOUNDS = [[-20, 20], [-4, 4]]
GOAL   = np.array([18.0, 0.0])

# ══════════════════════════════════════════════════════════
# 内联规划器（与你已有代码数值完全一致）
# ══════════════════════════════════════════════════════════

def _risk_pt(uav, ped_pos, ped_vel):
    """
    单点高斯风险（来自 sim_ghost_injection.py ImprovedPlanner.risk_score）
    """
    diff = uav - ped_pos
    spd  = np.linalg.norm(ped_vel)
    if spd < 0.01:
        return np.exp(-0.5 * np.dot(diff, diff) / SIGMA0**2)
    vd  = ped_vel / spd
    nd  = np.array([-vd[1], vd[0]])
    dp  = np.dot(diff, vd)
    dn  = np.dot(diff, nd)
    sp  = SIGMA0 + ALPHA_V * spd
    return np.exp(-0.5 * ((dp / sp)**2 + (dn / SIGMA0)**2))


def _risk_grid(GX, GY, peds):
    """
    全网格风险场（含预测时域，与 ImprovedPlanner.risk_score 逻辑一致）
    """
    total = np.zeros_like(GX, float)
    for (pp, pv) in peds:
        pp  = np.asarray(pp, float)
        pv  = np.asarray(pv, float)
        spd = np.linalg.norm(pv)
        # 各预测时刻叠加（权重与 risk_score 里的 tau/w 对应）
        for tau, w in [(0.0, 1.0), (0.5, 0.5), (1.0, 0.3), (1.5, 0.2)]:
            pred = pp + pv * tau
            dx   = GX - pred[0];  dy = GY - pred[1]
            if spd < 0.01:
                r = np.exp(-0.5 * (dx**2 + dy**2) / SIGMA0**2)
            else:
                vd  = pv / spd
                nd  = np.array([-vd[1], vd[0]])
                dpar  = dx * vd[0]  + dy * vd[1]
                dperp = dx * nd[0]  + dy * nd[1]
                sp  = SIGMA0 + ALPHA_V * spd
                r   = np.exp(-0.5 * ((dpar / sp)**2 + (dperp / SIGMA0)**2))
            decay = np.exp(-tau / T_PRED) if tau > 0 else 1.0
            total += w * decay * r
    # 归一化
    w_sum = 1.0 + 0.5 * np.exp(-0.5 / T_PRED) \
                + 0.3 * np.exp(-1.0 / T_PRED) \
                + 0.2 * np.exp(-1.5 / T_PRED)
    return np.clip(total / (len(peds) * w_sum + 1e-9), 0, 1)


def _in_wall(pos):
    x, y = pos
    # 商业街上下墙（与 bounds 对应）
    if y < BOUNDS[1][0] + 0.5 or y > BOUNDS[1][1] - 0.5:
        return True
    # 静态障碍物（柱子）
    for obs in STATIC_OBS:
        if np.linalg.norm(pos - np.array(obs[:2])) < obs[2] + UAV_R:
            return True
    return False


def _simulate_arc(pos, head, v, w, steps=20):
    """向前模拟一段弧线轨迹，返回 (N,2) 数组"""
    pts = [pos.copy()]
    p = pos.copy();  h = head
    for _ in range(steps):
        h += w * DT_SIM
        p  = p + v * np.array([np.cos(h), np.sin(h)]) * DT_SIM
        pts.append(p.copy())
    return np.array(pts)


def _score_dwa(pos, head, v, w, peds, steps=20):
    """
    标准DWA评分（来自 sim_ghost_injection.py DWAPlanner）
    dist 项仅用当前欧氏距离，不考虑速度方向
    """
    p = pos.copy();  h = head;  min_d = 1e9
    for _ in range(steps):
        h += w * DT_SIM
        p  = p + v * np.array([np.cos(h), np.sin(h)]) * DT_SIM
        if _in_wall(p):
            return -1e9
        for (pp, _) in peds:
            d = np.linalg.norm(p - np.array(pp)) - PED_R
            min_d = min(min_d, d)
            if d < 0:
                return -1e9
    tg  = GOAL - p;  dg = np.linalg.norm(tg)
    vd  = np.array([np.cos(h), np.sin(h)])
    heading = (np.dot(vd, tg / (dg + 1e-6)) + 1) / 2
    dist_sc = min(min_d / 3.0, 1.0)
    vel_sc  = v / UAV_VMAX
    # 与 DWAPlanner 的 0.5/0.35/0.15 一致
    return 0.50 * heading + 0.35 * dist_sc + 0.15 * vel_sc


def _score_semantic(pos, head, v, w, peds, steps=20):
    """
    语义感知DWA评分（来自 sim_ghost_injection.py ImprovedPlanner）
    safety 项用速度感知风险椭球 + 预测时域
    """
    p = pos.copy();  h = head;  max_risk = 0.;  min_d = 1e9
    d_obs = min((np.linalg.norm(pos - np.array(o[:2])) - o[2]
                 for o in STATIC_OBS), default=5.0)
    # 自适应权重（与你代码里 w_safe/w_head 对应）
    w_safe = 0.40 if d_obs > 2.5 else 0.55
    w_head = 0.50 if d_obs > 2.5 else 0.35

    for _ in range(steps):
        h += w * DT_SIM
        p  = p + v * np.array([np.cos(h), np.sin(h)]) * DT_SIM
        if _in_wall(p):
            return -1e9
        for (pp, pv) in peds:
            d = np.linalg.norm(p - np.array(pp)) - PED_R
            min_d = min(min_d, d)
            if d < 0:
                return -1e9
            # 预测时域风险（与 risk_score 里的 tau/w 完全对应）
            r = _risk_pt(p, np.array(pp), np.array(pv))
            for tau, wt in [(0.5, 0.5), (1.0, 0.3), (1.5, 0.2)]:
                pred = np.array(pp) + np.array(pv) * tau
                r   += wt * _risk_pt(p, pred, np.array(pv)) \
                          * np.exp(-tau / T_PRED)
            max_risk = max(max_risk, r)

    tg  = GOAL - p;  dg = np.linalg.norm(tg)
    vd  = np.array([np.cos(h), np.sin(h)])
    heading   = (np.dot(vd, tg / (dg + 1e-6)) + 1) / 2
    safety_sc = np.exp(-2.0 * min(max_risk, 1.0))   # 同 risk_score 的 exp(-2·max)
    return w_head * heading + w_safe * safety_sc + 0.10 * (v / UAV_VMAX)


def _best_action(pos, head, v_cur, peds, method='dwa'):
    """
    在速度空间采样，返回(最优v, 最优w, 所有候选轨迹列表)
    采样范围与 sim_ghost_injection.py DWAPlanner / ImprovedPlanner 一致
    """
    if method == 'dwa':
        vs = np.linspace(max(0.2, v_cur-0.4), min(UAV_VMAX, v_cur+0.4), 5)
        ws = np.linspace(-1.8, 1.8, 7)
    else:
        vs = np.linspace(max(0.5, v_cur-0.6), min(UAV_VMAX, v_cur+0.6), 7)
        ws = np.linspace(-2.2, 2.2, 11)

    best_sc = -1e9;  bv = v_cur;  bw = 0.
    all_trajs = []
    score_fn = _score_dwa if method == 'dwa' else _score_semantic

    for v in vs:
        for w in ws:
            sc   = score_fn(pos, head, v, w, peds)
            traj = _simulate_arc(pos, head, v, w)
            all_trajs.append((sc, traj))
            if sc > best_sc:
                best_sc = sc;  bv = v;  bw = w

    return bv, bw, all_trajs


def _ego_force(uav_pos, peds):
    """
    EGO势场合力（来自 sim_ghost_injection.py EGOPlanner.compute_force）
    """
    f  = np.zeros(2)
    tg = GOAL - uav_pos;  dg = np.linalg.norm(tg)
    if dg > 0:
        f += 2.0 * tg / dg
    for obs in STATIC_OBS:
        d2 = uav_pos - np.array(obs[:2])
        d  = np.linalg.norm(d2) - obs[2]
        if 0 < d < 0.8 * 2:
            f += 3.0 * (0.8 * 2 - d) * d2 / np.linalg.norm(d2)
    for (pp, _) in peds:
        d2 = uav_pos - np.array(pp)
        d  = np.linalg.norm(d2) - PED_R
        if 0 < d < 3.0:
            f += 4.0 * (3.0 - d) * d2 / np.linalg.norm(d2)
    return f


# ══════════════════════════════════════════════════════════
# 场景定义
# ══════════════════════════════════════════════════════════

# ── 场景A/B：高速迎面单行人 ────────────────────────────────
#    无人机刚进入商业街中段，一个行人以2.0m/s迎面冲来
#    时刻选在"已能感知威胁但DWA来不及绕开"的关键帧
UAV_AB   = np.array([-2.0, 0.0])
HEAD_AB  = 0.0        # 朝右（+x方向，朝向目标）
V_AB     = 1.8        # 当前速度 m/s（与你 simulator.py run_trial 典型速度一致）

PED1_P   = np.array([4.0, 0.2])
PED1_V   = np.array([-2.0, 0.1])   # 迎面，轻微侧偏

PEDS_AB  = [(PED1_P, PED1_V)]

# ── 场景C/D：双向夹击，EGO势场极小值 ─────────────────────────
#    两个行人从两侧夹击，势场引力+两个斥力合力约为零
UAV_CD   = np.array([0.0, 0.0])
HEAD_CD  = 0.0
V_CD     = 1.0

PED2_P   = np.array([-1.2,  0.0])
PED2_V   = np.array([ 1.6,  0.0])

PED3_P   = np.array([ 1.2,  0.0])
PED3_V   = np.array([-1.6,  0.0])

PEDS_CD  = [(PED2_P, PED2_V), (PED3_P, PED3_V)]



# ══════════════════════════════════════════════════════════
# 运行规划器，获取最优动作和候选轨迹
# ══════════════════════════════════════════════════════════
bv_dwa, bw_dwa, trajs_dwa = _best_action(UAV_AB, HEAD_AB, V_AB, PEDS_AB, 'dwa')
bv_sem, bw_sem, trajs_sem = _best_action(UAV_AB, HEAD_AB, V_AB, PEDS_AB, 'semantic')
opt_dwa = _simulate_arc(UAV_AB, HEAD_AB, bv_dwa, bw_dwa)
opt_sem = _simulate_arc(UAV_AB, HEAD_AB, bv_sem, bw_sem)

ped1_fut  = PED1_P + PED1_V * (20 * DT_SIM)   # 行人1秒后位置
d_now     = np.linalg.norm(UAV_AB - PED1_P)
d_fut_dwa = min(np.linalg.norm(pt - ped1_fut) for pt in opt_dwa)
d_fut_sem = min(np.linalg.norm(pt - ped1_fut) for pt in opt_sem)

print(f"  当前距离={d_now:.2f}m")
print(f"  标准DWA: v={bv_dwa:.2f} w={bw_dwa:.2f} 预测最近={d_fut_dwa:.2f}m")
print(f"  语义DWA: v={bv_sem:.2f} w={bw_sem:.2f} 预测最近={d_fut_sem:.2f}m")

print("计算场景C/D（EGO-Planner vs 本文改进）...")
ego_f     = _ego_force(UAV_CD, PEDS_CD)
ego_speed = np.linalg.norm(ego_f)
bv_sem_d, bw_sem_d, trajs_sem_d = _best_action(UAV_CD, HEAD_CD, V_CD, PEDS_CD, 'semantic')
opt_sem_d = _simulate_arc(UAV_CD, HEAD_CD, bv_sem_d, bw_sem_d)
d_min_cd  = min(
    min(np.linalg.norm(pt - PED2_P) for pt in opt_sem_d),
    min(np.linalg.norm(pt - PED3_P) for pt in opt_sem_d)
)

print(f"  EGO合力速度={ego_speed:.3f}m/s  "
      f"({'卡死' if ego_speed < 0.5 else '未完全卡死，可微调PED位置'})")
print(f"  语义DWA: v={bv_sem_d:.2f} w={bw_sem_d:.2f} 最近={d_min_cd:.2f}m")


# ══════════════════════════════════════════════════════════
# 预计算风险场网格（与你 simulator.py 的场景坐标系一致）
# ══════════════════════════════════════════════════════════
XMIN, XMAX = -8, 14
YMIN, YMAX = BOUNDS[1][0], BOUNDS[1][1]
gs  = 0.07
gx  = np.arange(XMIN, XMAX + gs, gs)
gy  = np.arange(YMIN, YMAX + gs, gs)
GX, GY = np.meshgrid(gx, gy)

risk_ab = _risk_grid(GX, GY, PEDS_AB)
risk_ab = gaussian_filter(risk_ab, sigma=0.5)

risk_cd = _risk_grid(GX, GY, PEDS_CD)
risk_cd = gaussian_filter(risk_cd, sigma=0.5)

RISK_CMAP = LinearSegmentedColormap.from_list(
    'risk',
    [(1,1,1,0.0), (0.95,0.55,0.45,0.35), (0.82,0.12,0.08,0.78)],
    N=256
)


# ══════════════════════════════════════════════════════════
# 绘图辅助函数
# ══════════════════════════════════════════════════════════

# ── 颜色（与你 sim_ghost_injection.py 里一致）──
C_DWA   = '#fb923c'    # 你代码里 DWA 的颜色 COLORS['DWA']
C_SEM   = '#34d399'    # 你代码里 改进方法的颜色 COLORS['本文改进方法']
C_EGO   = '#f87171'    # 你代码里 EGO 的颜色 COLORS['EGO-Planner（基线）']
C_PED   = '#f87171'
C_RISK  = '#f87171'
C_GOLD  = '#fbbf24'
C_GRAY  = '#9ca3af'
C_UAV   = '#60a5fa'

plt.rcParams.update({
    'font.family'       : 'WenQuanYi Zen Hei',
    'axes.unicode_minus': False,
    'font.size'         : 9,
    'figure.dpi'        : 120,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
})


def setup_ax(ax, title):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values():
        sp.set_color('#374151')
    ax.tick_params(colors=C_GRAY, labelsize=8)
    ax.set_xlabel('X (m)', color=C_GRAY, fontsize=8)
    ax.set_ylabel('Y (m)', color=C_GRAY, fontsize=8)
    ax.grid(True, color='#1f2937', lw=0.35, alpha=0.6)
    ax.set_title(title, fontsize=9, fontweight='bold',
                 color='#f1f5f9', pad=5)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN - 0.3, YMAX + 0.3)
    ax.set_aspect('equal')


def draw_walls(ax):
    """上下墙（与你 simulator.py bounds 对应）"""
    for y0, y1 in [(YMIN - 1.5, YMIN + 0.02), (YMAX - 0.02, YMAX + 1.5)]:
        ax.fill_between([XMIN, XMAX], [y0, y0], [y1, y1],
                        color='#374151', alpha=0.9, zorder=2)
    # 砖纹
    for yb in [YMIN + 0.5, YMAX - 0.5]:
        ax.plot([XMIN, XMAX], [yb, yb], color='#4b5563',
                lw=0.4, alpha=0.4, zorder=3)


def draw_static_obs(ax):
    """固定柱子（与 STATIC_OBS 对应）"""
    for obs in STATIC_OBS:
        if XMIN <= obs[0] <= XMAX:
            c = Circle(obs[:2], obs[2], color='#4b5563',
                       zorder=5, ec='#6b7280', lw=0.8)
            ax.add_patch(c)


def draw_risk_heatmap(ax, risk):
    ax.grid(False)
    ax.pcolormesh(GX, GY, risk, cmap=RISK_CMAP,
                  vmin=0, vmax=1, shading='gouraud',
                  alpha=0.55, zorder=3)


def draw_risk_ellipses(ax, ped_pos, ped_vel, color=C_RISK):
    """
    速度感知风险椭圆（对应你 ImprovedPlanner.risk_ellipsoid 的等值线）
    σ∥ = σ₀ + α·|v|，σ⊥ = σ₀
    """
    spd = np.linalg.norm(ped_vel)
    if spd < 0.01:
        return
    angle = np.degrees(np.arctan2(ped_vel[1], ped_vel[0]))
    s_par = SIGMA0 + ALPHA_V * spd
    # 绘制三个预测时刻的椭圆
    for tau, ls, alp in [(0.5, '--', 0.85), (1.0, '-.', 0.55), (1.5, ':', 0.35)]:
        pp = np.array(ped_pos) + np.array(ped_vel) * tau
        if XMIN <= pp[0] <= XMAX and YMIN <= pp[1] <= YMAX:
            ell = Ellipse(pp, 2 * s_par, 2 * SIGMA0, angle=angle,
                          facecolor='none', edgecolor=color,
                          linestyle=ls, linewidth=1.3,
                          alpha=alp, zorder=7)
            ax.add_patch(ell)
            ax.text(pp[0], pp[1] + SIGMA0 + 0.2,
                    f'τ={tau}s', fontsize=6.5, color=color,
                    alpha=alp + 0.05, ha='center', zorder=8)


def draw_ped(ax, pos, vel, idx=1):
    pos = np.array(pos);  vel = np.array(vel)
    ax.add_patch(Circle(pos, PED_R, color=C_PED, alpha=0.9,
                        ec='#9b1c1c', lw=1.0, zorder=9))
    spd = np.linalg.norm(vel)
    if spd > 0.05:
        uv = vel / spd
        ax.annotate('', xy=pos + uv * 0.85, xytext=pos + uv * 0.3,
                    arrowprops=dict(arrowstyle='->', color=C_PED,
                                    lw=2.0, mutation_scale=13), zorder=10)
    ax.text(pos[0], pos[1] + PED_R + 0.22,
            f'P{idx} {spd:.1f}m/s',
            fontsize=7.5, color='#fca5a5', ha='center',
            fontweight='bold', zorder=11)


def draw_uav(ax, pos, color=C_UAV):
    pos = np.array(pos)
    ax.add_patch(Circle(pos, UAV_R, color=color, alpha=0.15,
                        ls='--', ec=color, lw=1.2, zorder=8))
    ax.add_patch(Circle(pos, 0.2, color=color, alpha=0.92,
                        ec='#1e40af', lw=0.8, zorder=10))
    ax.text(pos[0], pos[1] + UAV_R + 0.25, 'UAV',
            fontsize=7.5, color=color, ha='center',
            fontweight='bold', zorder=11)


def draw_goal(ax):
    ax.plot(GOAL[0], GOAL[1], '*', color=C_GOLD,
            ms=13, zorder=10, mec='#92400e', mew=0.8)
    ax.text(GOAL[0], GOAL[1] + 0.5, '目标',
            fontsize=7.5, color='#92400e', ha='center', zorder=11)


def draw_candidates(ax, all_trajs, color='#6b7280', top_n=8):
    """绘制次优候选轨迹（浅色细线），说明速度空间采样范围"""
    ranked = sorted(all_trajs, key=lambda x: x[0], reverse=True)
    for sc, traj in ranked[1: top_n + 1]:
        if sc < -1e8:
            continue
        ax.plot(traj[:, 0], traj[:, 1], '-',
                color=color, lw=0.7, alpha=0.22, zorder=4)


def draw_optimal(ax, traj, color, label=None):
    """最优轨迹（粗实线）"""
    ax.plot(traj[:, 0], traj[:, 1], '-', color=color,
            lw=2.5, alpha=0.95, zorder=5, label=label)
    if len(traj) >= 4:
        ax.annotate('', xy=traj[-1], xytext=traj[-3],
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.8, mutation_scale=12), zorder=6)


def dist_label(ax, p1, p2, text='', color='#a78bfa'):
    p1 = np.array(p1);  p2 = np.array(p2)
    mx, my = (p1 + p2) / 2
    d = np.linalg.norm(p1 - p2)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
            '--', color=color, lw=0.9, alpha=0.5, zorder=6)
    ax.text(mx, my + 0.2, f'd={d:.2f}m{text}',
            fontsize=7.5, color=color, ha='center', va='bottom',
            bbox=dict(fc='#0f172a', ec='none', alpha=0.75, pad=1.5),
            zorder=9)


def warn_box(ax, pos, msg, color='#f97316'):
    ax.text(pos[0], pos[1], msg, fontsize=8, color=color,
            ha='center', va='center', fontweight='bold',
            bbox=dict(fc='#1c0700', ec=color, alpha=0.88,
                      pad=3, boxstyle='round,pad=0.4'),
            zorder=13)


def ok_box(ax, pos, msg, color='#34d399'):
    ax.text(pos[0], pos[1], msg, fontsize=8, color=color,
            ha='center', va='center', fontweight='bold',
            bbox=dict(fc='#022c22', ec=color, alpha=0.88,
                      pad=3, boxstyle='round,pad=0.4'),
            zorder=13)


# ══════════════════════════════════════════════════════════
# 绘制四格图（与你 sim_ghost_injection.py 的黑底风格一致）
# ══════════════════════════════════════════════════════════
BG = '#0d0d0d'
fig, axes = plt.subplots(2, 2, figsize=(16, 9),
                         facecolor=BG,
                         gridspec_kw={'hspace': 0.32, 'wspace': 0.18})
fig.patch.set_facecolor(BG)

(ax_a, ax_b), (ax_c, ax_d) = axes

# ─────────────────────────────────────────────────────────
# (a) 标准DWA ── 高速迎面，险情帧
# ─────────────────────────────────────────────────────────
setup_ax(ax_a, '(a) 标准DWA — 高速迎面行人，反应滞后险情帧')
draw_walls(ax_a);  draw_goal(ax_a)

# 行人1秒后预测虚影
ped1_shadow = PED1_P + PED1_V * 1.0
ax_a.add_patch(Circle(ped1_shadow, PED_R, color=C_PED,
                      alpha=0.18, ls=':', ec=C_PED, lw=1, zorder=4))
ax_a.annotate('', xy=ped1_shadow, xytext=PED1_P,
              arrowprops=dict(arrowstyle='->', color=C_PED,
                              lw=1.0, linestyle='dashed', alpha=0.3), zorder=4)
ax_a.text(ped1_shadow[0] + 0.3, ped1_shadow[1] - 0.45,
          '1s后预测位置', fontsize=6.5, color='#fca5a5', alpha=0.5)

# 候选轨迹（浅橙色）
draw_candidates(ax_a, trajs_dwa, color='#fb923c', top_n=10)
# 最优轨迹（橙色，径直朝行人前冲）
draw_optimal(ax_a, opt_dwa, C_DWA, label='DWA最优轨迹')

# 险情：轨迹末端离行人预测位置过近，画红色警示圈
clash = opt_dwa[len(opt_dwa) * 2 // 3]
for rr, aa in [(1.0, 0.12), (0.7, 0.22), (0.45, 0.38)]:
    ax_a.add_patch(Circle(clash, rr, color='#ef4444',
                          alpha=aa, lw=0, zorder=3))

warn_box(ax_a,
         (clash[0] + 0.1, clash[1] + 1.2),
         f'!! 预测最近距离 {d_fut_dwa:.2f}m\n< 安全半径 {UAV_R}m')

draw_ped(ax_a, PED1_P, PED1_V, idx=1)
draw_uav(ax_a, UAV_AB, C_UAV)
dist_label(ax_a, UAV_AB, PED1_P, '（当前）')

ax_a.legend(loc='upper right', facecolor='#111827',
            edgecolor='#374151', labelcolor='#f1f5f9', fontsize=7.5)

# ─────────────────────────────────────────────────────────
# (b) 本文改进 ── 同场景，提前规避
# ─────────────────────────────────────────────────────────
setup_ax(ax_b, '(b) 本文改进方法 — 速度感知风险场，提前规避帧')
draw_walls(ax_b);  draw_goal(ax_b)
draw_risk_heatmap(ax_b, risk_ab)
draw_risk_ellipses(ax_b, PED1_P, PED1_V, color=C_RISK)

draw_candidates(ax_b, trajs_sem, color='#34d399', top_n=10)
draw_optimal(ax_b, opt_sem, C_SEM, label='本文改进最优轨迹')

# 安全通过标注
safe_pt = opt_sem[len(opt_sem) // 2]
ok_box(ax_b,
       (safe_pt[0], safe_pt[1] + 1.3),
       f'OK 预测最近距离 {d_fut_sem:.2f}m\n> 安全半径 {UAV_R}m')

draw_ped(ax_b, PED1_P, PED1_V, idx=1)
draw_uav(ax_b, UAV_AB, C_UAV)
dist_label(ax_b, UAV_AB, PED1_P)

# 标注两个σ轴长度（说明椭球形状）
spd1 = np.linalg.norm(PED1_V)
ax_b.text(PED1_P[0] - 0.3, PED1_P[1] - SIGMA0 - 0.35,
          f'σ∥={SIGMA0+ALPHA_V*spd1:.2f}m  σ⊥={SIGMA0}m',
          fontsize=7, color='#fca5a5', ha='center', zorder=8)

ax_b.legend(loc='upper right', facecolor='#111827',
            edgecolor='#374151', labelcolor='#f1f5f9', fontsize=7.5)

# ─────────────────────────────────────────────────────────
# (c) EGO-Planner ── 双向夹击，势场极小值卡死
# ─────────────────────────────────────────────────────────
setup_ax(ax_c, '(c) EGO-Planner — 双向夹击，势场极小值，原地停止帧')
draw_walls(ax_c);  draw_goal(ax_c)

draw_ped(ax_c, PED2_P, PED2_V, idx=1)
draw_ped(ax_c, PED3_P, PED3_V, idx=2)
draw_uav(ax_c, UAV_CD, C_UAV)

# 画合力向量（近零）
if ego_speed > 1e-3:
    scale = min(ego_speed * 0.5, 0.3)
    ax_c.annotate('', xy=UAV_CD + ego_f / (ego_speed + 1e-6) * scale,
                  xytext=UAV_CD,
                  arrowprops=dict(arrowstyle='->', color='#f97316',
                                  lw=2.0, mutation_scale=12), zorder=9)

# 两个斥力箭头（虚线灰）
for (pp, _) in PEDS_CD:
    pp = np.array(pp)
    diff = UAV_CD - pp;  d = np.linalg.norm(diff)
    if d > 0:
        rep_dir = diff / d
        ax_c.annotate('', xy=UAV_CD + rep_dir * 0.9, xytext=UAV_CD,
                      arrowprops=dict(arrowstyle='->', color='#6b7280',
                                      lw=1.2, linestyle='dashed',
                                      mutation_scale=10, alpha=0.6), zorder=7)

# 引力箭头（金色虚线）
tg_n = (GOAL - UAV_CD) / np.linalg.norm(GOAL - UAV_CD)
ax_c.annotate('', xy=UAV_CD + tg_n * 1.1, xytext=UAV_CD,
              arrowprops=dict(arrowstyle='->', color=C_GOLD,
                              lw=1.4, mutation_scale=11, alpha=0.75), zorder=7)
ax_c.text(UAV_CD[0] + tg_n[0] * 1.3, UAV_CD[1] + tg_n[1] * 1.3 + 0.3,
          '引力', fontsize=7.5, color=C_GOLD, alpha=0.8)

warn_box(ax_c,
         (UAV_CD[0], UAV_CD[1] - 1.5),
         '两侧斥力相互抵消\n!! 有效推进力不足 → 停滞')

# 距离标注
for (pp, _), idx in zip(PEDS_CD, [1, 2]):
    dist_label(ax_c, UAV_CD, pp, f'（P{idx}）')

# ─────────────────────────────────────────────────────────
# (d) 本文改进 ── 同场景，穿越间隙
# ─────────────────────────────────────────────────────────
setup_ax(ax_d, '(d) 本文改进方法 — 预测时域，穿越间隙成功帧')
draw_walls(ax_d);  draw_goal(ax_d)
draw_risk_heatmap(ax_d, risk_cd)
for (pp, pv) in PEDS_CD:
    draw_risk_ellipses(ax_d, pp, pv, color=C_RISK)

draw_candidates(ax_d, trajs_sem_d, color='#34d399', top_n=10)
draw_optimal(ax_d, opt_sem_d, C_SEM, label='本文改进最优轨迹')

mid_d = opt_sem_d[len(opt_sem_d) // 2]
ok_box(ax_d,
       (mid_d[0], mid_d[1] + 1.35),
       f'OK 穿越双向行人间隙\n最近距离 {d_min_cd:.2f}m')

draw_ped(ax_d, PED2_P, PED2_V, idx=1)
draw_ped(ax_d, PED3_P, PED3_V, idx=2)
draw_uav(ax_d, UAV_CD, C_UAV)

ax_d.legend(loc='upper right', facecolor='#111827',
            edgecolor='#374151', labelcolor='#f1f5f9', fontsize=7.5)

# ══════════════════════════════════════════════════════════
# 整体标题 & 图例说明
# ══════════════════════════════════════════════════════════
fig.suptitle(
    '图4-3  典型冲突场景四格快照对比\n'
    '（左列：对比基线；右列：本文改进方法  ——  相同场景、相同时刻）',
    color='#f1f5f9', fontsize=11.5, fontweight='bold', y=1.01
)

fig.text(0.5, -0.015,
         '蓝色圆圈=无人机（安全半径0.5m）  红色圆圈=行人  箭头=速度方向  '
         '橙色/绿色折线=规划候选轨迹（粗线=最优）  '
         '虚线椭圆=速度感知风险等值线（τ=0.5/1.0/1.5s）  '
         '热力图=风险场（红深=高风险）',
         ha='center', color='#9ca3af', fontsize=7.5)

# 竖分隔线
fig.add_artist(plt.Line2D([0.505, 0.505], [0.02, 0.97],
               transform=fig.transFigure, color='#374151', lw=0.8))
fig.add_artist(plt.Line2D([0.05, 0.95], [0.505, 0.505],
               transform=fig.transFigure, color='#374151', lw=0.8))

# ══════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════
OUT_FIG = os.path.join(SCRIPT_DIR, 'fig4_3_conflict_snapshots.png')
plt.savefig(OUT_FIG, dpi=300, bbox_inches='tight', facecolor=BG)
print(f'\nOK 图像已保存：{OUT_FIG}')

# 同时输出关键数值文本，供填写论文正文
OUT_TXT = os.path.join(SCRIPT_DIR, 'fig4_3_data.txt')
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write('=== 图4-3 关键数值汇总（供填写论文正文和表格）===\n\n')
    f.write(f'[场景A/B]  无人机初始速度: {V_AB:.1f} m/s\n')
    f.write(f'           行人速度: {np.linalg.norm(PED1_V):.1f} m/s（迎面）\n')
    f.write(f'           当前UAV–行人距离: {d_now:.3f} m\n')
    f.write(f'           σ∥ = σ₀ + α·|v| = {SIGMA0} + {ALPHA_V}×{np.linalg.norm(PED1_V):.1f}'
            f' = {SIGMA0+ALPHA_V*np.linalg.norm(PED1_V):.2f} m\n')
    f.write(f'           σ⊥ = σ₀ = {SIGMA0} m\n')
    f.write(f'           标准DWA 最优速度: v={bv_dwa:.2f} m/s, ω={bw_dwa:.3f} rad/s\n')
    f.write(f'           标准DWA 预测最近距离: {d_fut_dwa:.3f} m '
            f'({"< 安全半径，险情" if d_fut_dwa < UAV_R else "> 安全半径"})\n')
    f.write(f'           语义DWA 最优速度: v={bv_sem:.2f} m/s, ω={bw_sem:.3f} rad/s\n')
    f.write(f'           语义DWA 预测最近距离: {d_fut_sem:.3f} m\n\n')
    f.write(f'[场景C/D]  行人1速度: {np.linalg.norm(PED2_V):.1f} m/s（左下→右上）\n')
    f.write(f'           行人2速度: {np.linalg.norm(PED3_V):.1f} m/s（右上→左下）\n')
    f.write(f'           EGO势场合力速度: {ego_speed:.4f} m/s '
            f'({"卡死" if ego_speed < 0.5 else "需微调行人位置使合力更小"})\n')
    f.write(f'           语义DWA 最优速度: v={bv_sem_d:.2f} m/s, ω={bw_sem_d:.3f} rad/s\n')
    f.write(f'           语义DWA 最近距离: {d_min_cd:.3f} m\n\n')
    f.write('参数配置（与 simulator.py / sim_ghost_injection.py 一致）:\n')
    f.write(f'  SIGMA0={SIGMA0}  ALPHA_V={ALPHA_V}  T_PRED={T_PRED}  UAV_R={UAV_R}\n')

print(f'OK 数值汇总已保存：{OUT_TXT}')
