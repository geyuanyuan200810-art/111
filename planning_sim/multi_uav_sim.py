"""
multi_uav_sim.py
多无人机场景仿真：验证速度感知风险椭球对高速他机（UAV-to-UAV）避障的适用性。

实验设计：
  配置1：纯行人（对照组）
  配置2：行人 + 1架他机（5m/s穿越）
  配置3：行人 + 2架他机（5~8m/s，不同方向）
每种配置运行50次，对比EGOPlanner和ImprovedPlanner的成功率、最小安全间距。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
#  基础类（精简版，从simulator.py复制）
# ============================================================

class Pedestrian:
    """行人模型：低速（1.2~2.2 m/s），边界反弹+随机扰动"""
    def __init__(self, x, y, vx, vy):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.base_vel = np.array([vx, vy], dtype=float)
        self.radius = 0.3  # 碰撞半径 m

    def update(self, dt, bounds):
        self.pos += self.vel * dt
        # 边界反弹
        for i in range(2):
            if self.pos[i] < bounds[i][0] or self.pos[i] > bounds[i][1]:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], bounds[i][0], bounds[i][1])
        # 轻微随机扰动（模拟真实行人步态）
        self.vel += np.random.randn(2) * 0.05
        speed = np.linalg.norm(self.vel)
        base_speed = np.linalg.norm(self.base_vel)
        if speed > 0:
            self.vel = self.vel / speed * np.clip(speed, 0.5, base_speed * 1.5)


class OtherUAV:
    """
    他机（OtherUAV）模型：高速直线穿越场景
    - 速度范围：5~12 m/s
    - 碰撞半径：0.4 m（比行人略大）
    - 运动模型：匀速直线，偶有 ±15° 随机偏转
    """
    def __init__(self, x, y, vx, vy):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.radius = 0.4  # 碰撞半径 m（比行人0.3m略大）
        # 记录初始速度大小，用于保持匀速
        self.speed_mag = np.linalg.norm(self.vel)
        # 随机偏转计时器
        self._deflect_timer = np.random.uniform(2.0, 5.0)

    def update(self, dt, bounds):
        # 偶发随机偏转（模拟他机小幅机动）
        self._deflect_timer -= dt
        if self._deflect_timer <= 0:
            angle_delta = np.random.uniform(-15, 15) * np.pi / 180  # ±15°
            c, s = np.cos(angle_delta), np.sin(angle_delta)
            vx_new = c * self.vel[0] - s * self.vel[1]
            vy_new = s * self.vel[0] + c * self.vel[1]
            self.vel = np.array([vx_new, vy_new])
            self._deflect_timer = np.random.uniform(2.0, 5.0)

        # 保持匀速大小
        speed = np.linalg.norm(self.vel)
        if speed > 0:
            self.vel = self.vel / speed * self.speed_mag

        self.pos += self.vel * dt
        # 边界反弹（他机在场景内循环）
        for i in range(2):
            if self.pos[i] < bounds[i][0] or self.pos[i] > bounds[i][1]:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], bounds[i][0], bounds[i][1])


class UAV:
    """自主无人机（被规划的ego UAV）"""
    def __init__(self, start, goal):
        self.pos = np.array(start, dtype=float)
        self.vel = np.zeros(2)
        self.goal = np.array(goal, dtype=float)
        self.max_vel = 3.0      # m/s
        self.max_acc = 2.0      # m/s²
        self.radius = 0.5       # 安全半径 m
        self.trajectory = [self.pos.copy()]
        self.min_dist_to_obs = float('inf')

    def reached_goal(self, tol=1.5):
        return np.linalg.norm(self.pos - self.goal) < tol


# ============================================================
#  规划器（精简版，从simulator.py复制）
# ============================================================

class EGOPlanner:
    """
    EGO-Planner 基线：
    - 静态障碍物排斥力
    - 动态障碍（行人/他机）只用当前位置，不预测速度方向
    """
    def __init__(self, d_safe=0.8):
        self.d_safe = d_safe

    def compute_force(self, uav, obstacles, static_obs):
        """
        obstacles: 行人 + 他机的混合列表，均有 .pos / .vel / .radius 属性
        static_obs: [(x, y, r), ...]
        """
        force = np.zeros(2)

        # 目标吸引力
        to_goal = uav.goal - uav.pos
        dist_goal = np.linalg.norm(to_goal)
        if dist_goal > 0:
            force += 2.0 * to_goal / dist_goal

        # 静态障碍物排斥力
        for obs in static_obs:
            to_uav = uav.pos - np.array(obs[:2])
            dist = np.linalg.norm(to_uav) - obs[2]
            if dist < self.d_safe * 2:
                if dist > 0:
                    force += 3.0 * (self.d_safe * 2 - dist) * to_uav / np.linalg.norm(to_uav)

        # 动态障碍排斥力（仅当前位置，不区分速度高低）
        for obs in obstacles:
            to_uav = uav.pos - obs.pos
            dist = np.linalg.norm(to_uav) - obs.radius
            if dist < self.d_safe:
                if np.linalg.norm(to_uav) > 0:
                    force += 2.0 * (self.d_safe - dist) * to_uav / np.linalg.norm(to_uav)

        return force


class ImprovedPlanner(EGOPlanner):
    """
    本文改进方法：速度感知风险椭球 + 自适应安全距离
    对高速他机（5~8 m/s），σ∥自动扩大，预警距离更远，无需手动调参。

    σ∥ = σ₀ + α × speed
      行人1.5m/s：σ∥ = 0.7 + 0.3×1.5 = 1.15m
      他机5.0m/s：σ∥ = 0.7 + 0.3×5.0 = 2.20m
      他机8.0m/s：σ∥ = 0.7 + 0.3×8.0 = 3.10m
    """
    def __init__(self, d_safe=1.0, sigma0=0.7, alpha=0.3, T_pred=1.5,
                 beta=0.65):
        super().__init__(d_safe)
        self.sigma0 = sigma0    # 基础风险半径 m
        self.alpha = alpha      # 速度扩张系数
        self.T_pred = T_pred    # 预测时域 s
        # 三权重（beta=安全权重）
        self.beta = beta
        self.alpha_w = (1 - beta) * 0.55   # 目标方向权重
        self.gamma_w = (1 - beta) * 0.45   # 速度平滑权重

    def risk_ellipsoid(self, uav_pos, obs_pos, obs_vel):
        """
        速度感知高斯风险场（核心创新）
        返回：(风险值, 梯度向量)
        """
        speed = np.linalg.norm(obs_vel)

        # 纵向风险半径（沿速度方向）随速度自动扩大
        sigma_parallel = self.sigma0 + self.alpha * speed
        sigma_perp = self.sigma0  # 横向保持基础值

        # 速度方向单位向量
        if speed > 0.01:
            v_dir = obs_vel / speed
        else:
            v_dir = np.array([1.0, 0.0])
        perp_dir = np.array([-v_dir[1], v_dir[0]])

        diff = uav_pos - obs_pos
        d_parallel = np.dot(diff, v_dir)
        d_perp = np.dot(diff, perp_dir)

        risk = np.exp(-0.5 * (
            (d_parallel / sigma_parallel) ** 2 +
            (d_perp / sigma_perp) ** 2
        ))

        # 解析梯度，用于计算排斥力方向
        grad = -risk * np.array([
            d_parallel / sigma_parallel ** 2 * v_dir[0] +
            d_perp / sigma_perp ** 2 * perp_dir[0],
            d_parallel / sigma_parallel ** 2 * v_dir[1] +
            d_perp / sigma_perp ** 2 * perp_dir[1]
        ])

        return risk, grad

    def compute_force(self, uav, obstacles, static_obs):
        force = np.zeros(2)

        # 目标吸引力
        to_goal = uav.goal - uav.pos
        dist_goal = np.linalg.norm(to_goal)
        if dist_goal > 0:
            force += 2.0 * to_goal / dist_goal

        # 静态障碍物排斥力
        for obs in static_obs:
            to_uav = uav.pos - np.array(obs[:2])
            dist = np.linalg.norm(to_uav) - obs[2]
            if dist < self.d_safe * 2:
                if dist > 0:
                    force += 3.0 * (self.d_safe * 2 - dist) * to_uav / np.linalg.norm(to_uav)

        # 自适应安全权重（障碍密集时增强安全优先）
        lambda_d = self._adaptive_lambda(uav, static_obs)

        # 动态障碍：速度感知风险椭球（行人+他机统一处理）
        for obs in obstacles:
            # 当前位置风险
            risk, grad = self.risk_ellipsoid(uav.pos, obs.pos, obs.vel)

            # 相对速度（接近速度越快，权重越高）
            rel_vel = uav.vel - obs.vel
            to_uav = uav.pos - obs.pos
            dist = max(np.linalg.norm(to_uav), 0.01)
            v_rel_approach = max(0, -np.dot(rel_vel, to_uav / dist))
            weight = 1.0 + 0.5 * v_rel_approach

            force -= lambda_d * weight * grad * 5.0

            # CV预测：未来位置的预期风险（时间衰减）
            for tau in [0.5, 1.0, 1.5]:
                pred_pos = obs.pos + obs.vel * tau
                risk_pred, grad_pred = self.risk_ellipsoid(uav.pos, pred_pos, obs.vel)
                decay = np.exp(-tau / self.T_pred)
                force -= lambda_d * weight * grad_pred * 3.0 * decay

        return force

    def _adaptive_lambda(self, uav, static_obs):
        """自适应效率权重：障碍物密集时安全优先"""
        min_dist = float('inf')
        for obs in static_obs:
            d = np.linalg.norm(uav.pos - np.array(obs[:2])) - obs[2]
            min_dist = min(min_dist, d)
        d_ref = 3.0
        lambda_max = 4.0
        return lambda_max * np.exp(-min_dist / d_ref)


# ============================================================
#  商业街场景参数（与simulator.py保持一致）
# ============================================================

SCENE_PARAMS = {
    'start': [-18.0, 0.0],
    'goal': [18.0, 0.0],
    'bounds': [[-20, 20], [-4, 4]],
    'static_obs': [
        [-5.0, 3.5, 0.8],
        [5.0, -3.5, 0.8],
        [0.0, 3.8, 0.6],
        [-10.0, -3.5, 0.7],
    ]
}


def make_pedestrians(n_min=3, n_max=5):
    """生成行人列表（商业街：双向，1.2~2.2 m/s）"""
    n = np.random.randint(n_min, n_max + 1)
    peds = []
    for _ in range(n):
        x = np.random.uniform(-15, 15)
        y = np.random.uniform(-3, 3)
        # 双向行走，速度1.2~2.2 m/s
        vx = np.random.choice([-1, 1]) * np.random.uniform(1.2, 2.2)
        vy = np.random.uniform(-0.2, 0.2)
        peds.append(Pedestrian(x, y, vx, vy))
    return peds


def make_other_uavs(config_id):
    """
    根据配置生成他机列表
    config_id=1: 无他机（纯行人对照）
    config_id=2: 1架5m/s穿越
    config_id=3: 2架5~8m/s，不同方向穿越
    """
    if config_id == 1:
        return []
    elif config_id == 2:
        # 1架他机从右向左穿越，速度5m/s
        uav = OtherUAV(15.0, np.random.uniform(-2, 2), -5.0, 0.0)
        return [uav]
    else:  # config_id == 3
        # 他机1：从右向左，5 m/s
        uav1 = OtherUAV(15.0, np.random.uniform(-1, 1), -5.0, 0.0)
        # 他机2：从上向下斜穿，8 m/s
        speed2 = np.random.uniform(5.0, 8.0)
        uav2 = OtherUAV(
            np.random.uniform(-5, 5), 3.5,
            np.random.uniform(-1, 1) * speed2 * 0.3,
            -speed2 * 0.95
        )
        return [uav1, uav2]


# ============================================================
#  单次仿真试验
# ============================================================

def run_trial(planner, config_id, max_time=60.0, dt=0.05, seed=None,
              record_traj=False):
    """
    运行一次仿真试验。
    返回 dict: success, time, min_dist, trajectory(可选)
    """
    if seed is not None:
        np.random.seed(seed)

    p = SCENE_PARAMS
    uav = UAV(p['start'], p['goal'])
    static_obs = p['static_obs']
    bounds = p['bounds']

    # 生成行人（3~5人，对照组也有行人）
    pedestrians = make_pedestrians(3, 5)
    # 生成他机（配置决定数量/速度）
    other_uavs = make_other_uavs(config_id)

    # 所有动态障碍合并（行人+他机）
    all_obstacles = pedestrians + other_uavs

    t = 0.0
    success = False
    traj = [uav.pos.copy()] if record_traj else None

    while t < max_time:
        # 更新所有动态障碍
        for ped in pedestrians:
            ped.update(dt, bounds)
        for ouav in other_uavs:
            ouav.update(dt, bounds)

        # 规划力
        force = planner.compute_force(uav, all_obstacles, static_obs)

        # 更新ego无人机动力学
        acc = np.clip(force, -uav.max_acc, uav.max_acc)
        uav.vel += acc * dt
        speed = np.linalg.norm(uav.vel)
        if speed > uav.max_vel:
            uav.vel = uav.vel / speed * uav.max_vel
        uav.pos += uav.vel * dt
        if record_traj:
            traj.append(uav.pos.copy())

        # 记录最小安全间距（与所有动态/静态障碍）
        for obs in all_obstacles:
            d = np.linalg.norm(uav.pos - obs.pos) - obs.radius - uav.radius
            uav.min_dist_to_obs = min(uav.min_dist_to_obs, d)
        for obs in static_obs:
            d = np.linalg.norm(uav.pos - np.array(obs[:2])) - obs[2] - uav.radius
            uav.min_dist_to_obs = min(uav.min_dist_to_obs, d)

        # 碰撞检测
        collision = False
        for obs in all_obstacles:
            if np.linalg.norm(uav.pos - obs.pos) < uav.radius + obs.radius:
                collision = True
                break
        if not collision:
            for obs in static_obs:
                if np.linalg.norm(uav.pos - np.array(obs[:2])) < uav.radius + obs[2]:
                    collision = True
                    break

        if collision:
            break

        if uav.reached_goal():
            success = True
            break

        t += dt

    result = {
        'success': success,
        'time': round(t, 2),
        'min_dist': round(max(0.0, uav.min_dist_to_obs), 3),
    }
    if record_traj:
        result['trajectory'] = traj
    return result


# ============================================================
#  批量实验
# ============================================================

CONFIG_NAMES = {
    1: '纯行人\n（对照组）',
    2: '行人+1架\n他机5m/s',
    3: '行人+2架\n他机5~8m/s',
}

PLANNER_NAMES = {
    'ego': 'EGO-Planner\n（基线）',
    'improved': '本文改进\n（速度感知）',
}


def run_all_experiments(n_trials=50):
    """
    3配置 × 2规划器 = 6组，每组n_trials次。
    返回 results[config_id][planner_key] = {success_rate, avg_min_dist, avg_time}
    """
    planners = {
        'ego': EGOPlanner(d_safe=0.8),
        'improved': ImprovedPlanner(d_safe=1.0, sigma0=0.7, alpha=0.3, T_pred=1.5),
    }

    results = {}
    for cfg in [1, 2, 3]:
        results[cfg] = {}
        for pk, planner in planners.items():
            print(f"  配置{cfg}（{CONFIG_NAMES[cfg].replace(chr(10),' ')}）"
                  f" × {pk} — {n_trials}次仿真...", end=' ', flush=True)
            trials = []
            for i in range(n_trials):
                r = run_trial(planner, cfg, seed=i * 37 + cfg * 100)
                trials.append(r)

            n_succ = sum(r['success'] for r in trials)
            sr = n_succ / n_trials * 100
            dists = [r['min_dist'] for r in trials]
            times = [r['time'] for r in trials if r['success']]
            results[cfg][pk] = {
                'success_rate': round(sr, 1),
                'avg_min_dist': round(float(np.mean(dists)), 3),
                'avg_time': round(float(np.mean(times)), 2) if times else None,
                'trials': trials,
            }
            print(f"成功率={sr:.1f}%  最小间距={results[cfg][pk]['avg_min_dist']:.3f}m")

    return results


# ============================================================
#  打印结果表
# ============================================================

def print_results_table(results):
    """终端打印 3×2 结果表（3场景 × 2算法）"""
    header = f"{'配置':<20} {'算法':<14} {'成功率%':>8} {'最小间距m':>10} {'平均时间s':>10}"
    print("\n" + "=" * 65)
    print("多无人机场景仿真结果汇总（每组50次）")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for cfg in [1, 2, 3]:
        cfg_label = CONFIG_NAMES[cfg].replace('\n', ' ')
        for pk in ['ego', 'improved']:
            r = results[cfg][pk]
            planner_label = PLANNER_NAMES[pk].replace('\n', ' ')
            t_str = f"{r['avg_time']:.2f}" if r['avg_time'] is not None else "N/A"
            print(f"{cfg_label:<20} {planner_label:<14} "
                  f"{r['success_rate']:>8.1f} {r['avg_min_dist']:>10.3f} {t_str:>10}")
        print("-" * 65)

    # 打印σ∥计算说明（论文素材）
    print("\n速度感知风险椭球σ∥计算（σ₀=0.7, α=0.3）：")
    print(f"  行人 1.5 m/s → σ∥ = 0.7 + 0.3×1.5 = {0.7+0.3*1.5:.2f} m")
    print(f"  他机 5.0 m/s → σ∥ = 0.7 + 0.3×5.0 = {0.7+0.3*5.0:.2f} m")
    print(f"  他机 8.0 m/s → σ∥ = 0.7 + 0.3×8.0 = {0.7+0.3*8.0:.2f} m")
    print("  → 高速他机预警距离自动扩大，无需手动调参。")


# ============================================================
#  绘图：柱状图 + 典型轨迹
# ============================================================

def plot_main_results(results, save_path='/home/yuan/planning_sim/multi_uav_results.png'):
    """
    保存图表：左侧柱状图（成功率 + 最小间距），右侧典型轨迹对比
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('多无人机场景仿真结果\n速度感知风险椭球对高速他机的适用性验证',
                 fontsize=14, fontweight='bold')

    configs = [1, 2, 3]
    cfg_labels = [CONFIG_NAMES[c].replace('\n', '\n') for c in configs]
    x = np.arange(len(configs))
    width = 0.35

    # --- 子图1：成功率柱状图 ---
    ax1 = fig.add_subplot(2, 3, 1)
    sr_ego = [results[c]['ego']['success_rate'] for c in configs]
    sr_imp = [results[c]['improved']['success_rate'] for c in configs]
    bars1 = ax1.bar(x - width / 2, sr_ego, width, label='EGO-Planner（基线）',
                    color='#5B9BD5', alpha=0.85, edgecolor='white')
    bars2 = ax1.bar(x + width / 2, sr_imp, width, label='本文改进（速度感知）',
                    color='#ED7D31', alpha=0.85, edgecolor='white')
    ax1.set_ylabel('成功率 (%)', fontsize=11)
    ax1.set_title('任务成功率对比', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfg_labels, fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    # 标注数值
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)

    # --- 子图2：最小安全间距柱状图 ---
    ax2 = fig.add_subplot(2, 3, 2)
    md_ego = [results[c]['ego']['avg_min_dist'] for c in configs]
    md_imp = [results[c]['improved']['avg_min_dist'] for c in configs]
    ax2.bar(x - width / 2, md_ego, width, label='EGO-Planner（基线）',
            color='#5B9BD5', alpha=0.85, edgecolor='white')
    ax2.bar(x + width / 2, md_imp, width, label='本文改进（速度感知）',
            color='#ED7D31', alpha=0.85, edgecolor='white')
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.2, label='安全阈值0.5m')
    ax2.set_ylabel('平均最小安全间距 (m)', fontsize=11)
    ax2.set_title('最小安全间距对比', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfg_labels, fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # --- 子图3：完成时间柱状图 ---
    ax3 = fig.add_subplot(2, 3, 3)
    t_ego = [results[c]['ego']['avg_time'] or 0 for c in configs]
    t_imp = [results[c]['improved']['avg_time'] or 0 for c in configs]
    ax3.bar(x - width / 2, t_ego, width, label='EGO-Planner（基线）',
            color='#5B9BD5', alpha=0.85, edgecolor='white')
    ax3.bar(x + width / 2, t_imp, width, label='本文改进（速度感知）',
            color='#ED7D31', alpha=0.85, edgecolor='white')
    ax3.set_ylabel('平均完成时间 (s)', fontsize=11)
    ax3.set_title('任务完成时间对比', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cfg_labels, fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # --- 子图4~6：配置2（行人+1架他机）典型轨迹对比 ---
    for col, pk in enumerate(['ego', 'improved']):
        ax = fig.add_subplot(2, 3, 4 + col)

        # 重新跑一次记录轨迹（固定seed=999）
        planner = (EGOPlanner(d_safe=0.8) if pk == 'ego'
                   else ImprovedPlanner(d_safe=1.0, sigma0=0.7, alpha=0.3))
        r = run_trial(planner, config_id=2, seed=999, record_traj=True)

        traj = np.array(r['trajectory'])
        ax.plot(traj[:, 0], traj[:, 1], '-', color='#ED7D31' if pk == 'improved' else '#5B9BD5',
                linewidth=1.8, label='UAV轨迹', zorder=5)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='起点', zorder=6)
        ax.plot(SCENE_PARAMS['goal'][0], SCENE_PARAMS['goal'][1],
                'r*', markersize=12, label='终点', zorder=6)

        # 绘制静态障碍物
        for obs in SCENE_PARAMS['static_obs']:
            circle = plt.Circle((obs[0], obs[1]), obs[2],
                                 color='gray', alpha=0.5, zorder=3)
            ax.add_patch(circle)

        # 标注他机初始位置和方向
        ax.annotate('', xy=(10.0, 0.0), xytext=(15.0, 0.0),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax.text(15.0, 0.3, '他机 5m/s', fontsize=8, color='purple')

        ax.set_xlim(-21, 21)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        status = '成功' if r['success'] else '失败'
        alg_name = '本文改进' if pk == 'improved' else 'EGO-Planner'
        ax.set_title(f'典型轨迹：配置2 × {alg_name}\n({status}, 最小间距={r["min_dist"]:.2f}m)',
                     fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.2)

    # --- 子图6：σ∥随速度变化曲线 ---
    ax6 = fig.add_subplot(2, 3, 6)
    speeds = np.linspace(0, 10, 200)
    sigma0, alpha = 0.7, 0.3
    sigma_parallel = sigma0 + alpha * speeds
    sigma_perp = np.full_like(speeds, sigma0)

    ax6.plot(speeds, sigma_parallel, 'r-', linewidth=2, label='σ∥（纵向，速度感知）')
    ax6.plot(speeds, sigma_perp, 'b--', linewidth=2, label='σ⊥（横向，固定）')
    ax6.axvline(x=1.5, color='green', linestyle=':', linewidth=1.5)
    ax6.axvline(x=5.0, color='orange', linestyle=':', linewidth=1.5)
    ax6.axvline(x=8.0, color='red', linestyle=':', linewidth=1.5)
    ax6.text(1.5, 3.5, f'行人\n1.5m/s\nσ∥={0.7+0.3*1.5:.2f}m', ha='center',
             fontsize=7.5, color='green')
    ax6.text(5.0, 3.5, f'他机\n5m/s\nσ∥={0.7+0.3*5.0:.2f}m', ha='center',
             fontsize=7.5, color='orange')
    ax6.text(8.0, 3.5, f'他机\n8m/s\nσ∥={0.7+0.3*8.0:.2f}m', ha='center',
             fontsize=7.5, color='red')
    ax6.set_xlabel('障碍物速度 (m/s)', fontsize=11)
    ax6.set_ylabel('风险椭球半径 (m)', fontsize=11)
    ax6.set_title('速度感知风险椭球半径变化\n（σ∥=σ₀+α×speed）', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_ylim(0, 4.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n主结果图已保存至：{save_path}")


# ============================================================
#  绘图：风险椭球示意图
# ============================================================

def plot_risk_ellipsoid_demo(save_path='/home/yuan/planning_sim/risk_ellipsoid_demo.png'):
    """
    风险椭球示意图：行人（1.5m/s）和他机（5m/s/8m/s）的风险场等高线对比
    x轴y轴为空间坐标，colormap显示风险值，箭头标注速度方向。
    """
    sigma0, alpha = 0.7, 0.3

    def gaussian_risk_field(X, Y, obs_pos, obs_vel):
        """计算网格上每点的高斯风险值"""
        speed = np.linalg.norm(obs_vel)
        sigma_parallel = sigma0 + alpha * speed
        sigma_perp = sigma0

        if speed > 0.01:
            v_dir = obs_vel / speed
        else:
            v_dir = np.array([1.0, 0.0])
        perp_dir = np.array([-v_dir[1], v_dir[0]])

        dx = X - obs_pos[0]
        dy = Y - obs_pos[1]
        d_par = dx * v_dir[0] + dy * v_dir[1]
        d_perp = dx * perp_dir[0] + dy * perp_dir[1]

        risk = np.exp(-0.5 * (
            (d_par / sigma_parallel) ** 2 +
            (d_perp / sigma_perp) ** 2
        ))
        return risk

    # 网格范围
    xr = np.linspace(-5, 5, 300)
    yr = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(xr, yr)

    # 三类障碍物：位置均在原点，速度方向沿+x
    cases = [
        {'label': '行人 1.5 m/s', 'vel': np.array([1.5, 0.0]),
         'color': 'Blues', 'arrow_color': '#1f77b4'},
        {'label': '他机 5.0 m/s', 'vel': np.array([5.0, 0.0]),
         'color': 'Oranges', 'arrow_color': '#ff7f0e'},
        {'label': '他机 8.0 m/s', 'vel': np.array([8.0, 0.0]),
         'color': 'Reds', 'arrow_color': '#d62728'},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('速度感知风险椭球等高线示意图\n（障碍物位于原点，箭头表示速度方向）',
                 fontsize=14, fontweight='bold')

    levels = np.linspace(0.05, 1.0, 16)

    for ax, case in zip(axes, cases):
        speed = np.linalg.norm(case['vel'])
        sigma_par = sigma0 + alpha * speed

        Z = gaussian_risk_field(X, Y, np.array([0.0, 0.0]), case['vel'])

        # 填充等高线（colormap显示风险值）
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=case['color'], alpha=0.85)
        # 等高线轮廓线
        cs = ax.contour(X, Y, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                        colors='white', linewidths=0.8, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

        plt.colorbar(cf, ax=ax, label='风险值', shrink=0.85)

        # 障碍物位置（黑点）
        ax.plot(0, 0, 'ko', markersize=8, zorder=10, label='障碍物位置')

        # 速度方向箭头（归一化长度）
        arrow_len = 2.0
        ax.annotate('',
                    xy=(arrow_len, 0),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=case['arrow_color'],
                                   lw=2.5))
        ax.text(arrow_len + 0.1, 0.15,
                f'v={speed:.1f}m/s', fontsize=9, color=case['arrow_color'],
                fontweight='bold')

        # 标注σ∥和σ⊥
        ax.annotate('', xy=(sigma_par, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5,
                                   linestyle='dashed'))
        ax.text(sigma_par / 2, 0.25, f'σ∥={sigma_par:.2f}m',
                ha='center', fontsize=8.5, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.annotate('', xy=(0, sigma0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5,
                                   linestyle='dashed'))
        ax.text(0.3, sigma0 / 2, f'σ⊥={sigma0}m',
                ha='left', fontsize=8.5, color='black',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', fontsize=11)
        ax.set_ylabel('y (m)', fontsize=11)
        ax.set_title(f'{case["label"]}\nσ∥={sigma_par:.2f}m, σ⊥={sigma0:.2f}m',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.2, color='white')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"风险椭球示意图已保存至：{save_path}")


# ============================================================
#  主入口
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("多无人机场景仿真")
    print("验证速度感知风险椭球对高速他机（UAV-to-UAV）避障的适用性")
    print("=" * 60)
    print("\n核心参数说明（σ∥=σ₀+α×speed, σ₀=0.7, α=0.3）：")
    print(f"  行人 1.5m/s → σ∥ = {0.7+0.3*1.5:.2f}m（预警距离较小）")
    print(f"  他机 5.0m/s → σ∥ = {0.7+0.3*5.0:.2f}m（预警距离自动扩大）")
    print(f"  他机 8.0m/s → σ∥ = {0.7+0.3*8.0:.2f}m（预警距离进一步扩大）\n")

    print("开始批量仿真（每组50次）...")
    results = run_all_experiments(n_trials=50)

    print_results_table(results)

    print("\n生成图表...")
    import os
    os.makedirs('/home/yuan/planning_sim', exist_ok=True)
    plot_main_results(results, save_path='/home/yuan/planning_sim/multi_uav_results.png')
    plot_risk_ellipsoid_demo(save_path='/home/yuan/planning_sim/risk_ellipsoid_demo.png')

    print("\n所有任务完成！")
