import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import CubicSpline
import json, os, time

class Pedestrian:
    def __init__(self, x, y, vx, vy, scene='commercial'):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.scene = scene
        self.base_vel = np.array([vx, vy], dtype=float)
        
    def update(self, dt, bounds):
        self.pos += self.vel * dt
        # 边界反弹
        for i in range(2):
            if self.pos[i] < bounds[i][0] or self.pos[i] > bounds[i][1]:
                self.vel[i] *= -1
                self.pos[i] = np.clip(self.pos[i], bounds[i][0], bounds[i][1])
        # 随机轻微扰动（模拟真实行人）
        self.vel += np.random.randn(2) * 0.05
        speed = np.linalg.norm(self.vel)
        base_speed = np.linalg.norm(self.base_vel)
        if speed > 0:
            self.vel = self.vel / speed * np.clip(speed, 0.5, base_speed * 1.5)

class UAV:
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

class EGOPlanner:
    """EGO-Planner 基线：只用静态障碍物代价"""
    def __init__(self, d_safe=0.8):
        self.d_safe = d_safe
        
    def compute_force(self, uav, pedestrians, static_obs):
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
        
        # 行人排斥力（只用当前位置，不预测）
        for ped in pedestrians:
            to_uav = uav.pos - ped.pos
            dist = np.linalg.norm(to_uav) - 0.3
            if dist < self.d_safe:
                if np.linalg.norm(to_uav) > 0:
                    force += 2.0 * (self.d_safe - dist) * to_uav / np.linalg.norm(to_uav)
        
        return force

class ImprovedPlanner(EGOPlanner):
    """本文改进：速度感知风险椭球 + 自适应安全距离"""
    def __init__(self, d_safe=1.0, sigma0=0.7, alpha=0.3, T_pred=1.5):
        super().__init__(d_safe)
        self.sigma0 = sigma0    # 基础风险半径
        self.alpha = alpha      # 速度扩张系数
        self.T_pred = T_pred    # 预测时域
        
    def risk_ellipsoid(self, uav_pos, ped_pos, ped_vel):
        """计算速度感知高斯风险场"""
        speed = np.linalg.norm(ped_vel)
        
        # 纵向（速度方向）风险更大
        sigma_parallel = self.sigma0 + self.alpha * speed
        sigma_perp = self.sigma0
        
        # 速度方向单位向量
        if speed > 0.01:
            v_dir = ped_vel / speed
        else:
            v_dir = np.array([1.0, 0.0])
        perp_dir = np.array([-v_dir[1], v_dir[0]])
        
        # 当前位置风险
        diff = uav_pos - ped_pos
        d_parallel = np.dot(diff, v_dir)
        d_perp = np.dot(diff, perp_dir)
        
        risk = np.exp(-0.5 * (
            (d_parallel / sigma_parallel)**2 +
            (d_perp / sigma_perp)**2
        ))
        
        # 梯度（解析形式）
        grad = -risk * np.array([
            d_parallel / sigma_parallel**2 * v_dir[0] + d_perp / sigma_perp**2 * perp_dir[0],
            d_parallel / sigma_parallel**2 * v_dir[1] + d_perp / sigma_perp**2 * perp_dir[1]
        ])
        
        return risk, grad
    
    def compute_force(self, uav, pedestrians, static_obs):
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
        
        # 改进：速度感知风险椭球
        lambda_d = self.adaptive_lambda(uav, static_obs)
        
        for ped in pedestrians:
            # 当前位置风险
            risk, grad = self.risk_ellipsoid(uav.pos, ped.pos, ped.vel)
            
            # 相对速度（接近方向权重）
            rel_vel = uav.vel - ped.vel
            to_uav = uav.pos - ped.pos
            dist = max(np.linalg.norm(to_uav), 0.01)
            v_rel_approach = max(0, -np.dot(rel_vel, to_uav/dist))
            
            weight = 1.0 + 0.5 * v_rel_approach
            force -= lambda_d * weight * grad * 5.0
            
            # CV 预测：未来位置风险
            for tau in [0.5, 1.0, 1.5]:
                pred_pos = ped.pos + ped.vel * tau
                risk_pred, grad_pred = self.risk_ellipsoid(uav.pos, pred_pos, ped.vel)
                decay = np.exp(-tau / self.T_pred)
                force -= lambda_d * weight * grad_pred * 3.0 * decay
        
        return force
    
    def adaptive_lambda(self, uav, static_obs):
        """自适应效率权重：障碍物密集时安全优先"""
        min_dist = float('inf')
        for obs in static_obs:
            d = np.linalg.norm(uav.pos - np.array(obs[:2])) - obs[2]
            min_dist = min(min_dist, d)
        d_ref = 3.0
        lambda_max = 4.0
        return lambda_max * np.exp(-min_dist / d_ref)

def run_trial(scene, planner, max_time=60.0, dt=0.05, seed=None):
    """运行一次仿真试验"""
    if seed is not None:
        np.random.seed(seed)
    
    # 场景设置
    if scene == 'commercial':
        # 商业街：窄通道，双向行人
        start = [-18.0, 0.0]
        goal  = [18.0, 0.0]
        bounds = [[-20, 20], [-4, 4]]
        static_obs = [
            [-5.0, 3.5, 0.8], [5.0, -3.5, 0.8],
            [0.0, 3.8, 0.6],  [-10.0, -3.5, 0.7],
        ]
        n_peds = np.random.randint(3, 7)
        pedestrians = []
        for i in range(n_peds):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(-3, 3)
            vx = np.random.choice([-1, 1]) * np.random.uniform(0.8, 1.5)
            vy = np.random.uniform(-0.2, 0.2)
            pedestrians.append(Pedestrian(x, y, vx, vy, scene))
            
    else:  # residential
        # 居民区：开阔广场，随机散步行人
        start = [-25.0, 0.0]
        goal  = [25.0, 0.0]
        bounds = [[-28, 28], [-20, 20]]
        static_obs = [
            [15.0, 15.0, 5.0], [-15.0, 15.0, 5.0],
            [15.0, -15.0, 5.0], [-15.0, -15.0, 5.0],
            [0.0, 18.0, 3.0],
        ]
        n_peds = np.random.randint(8, 15)
        pedestrians = []
        for i in range(n_peds):
            x = np.random.uniform(-20, 20)
            y = np.random.uniform(-15, 15)
            speed = np.random.uniform(0.6, 1.5)
            angle = np.random.uniform(0, 2*np.pi)
            pedestrians.append(Pedestrian(x, y, 
                speed*np.cos(angle), speed*np.sin(angle), scene))
    
    uav = UAV(start, goal)
    t = 0.0
    success = False
    
    while t < max_time:
        # 更新行人
        for ped in pedestrians:
            ped.update(dt, bounds)
        
        # 规划器计算力
        force = planner.compute_force(uav, pedestrians, static_obs)
        
        # 更新无人机
        acc = np.clip(force, -uav.max_acc, uav.max_acc)
        uav.vel += acc * dt
        speed = np.linalg.norm(uav.vel)
        if speed > uav.max_vel:
            uav.vel = uav.vel / speed * uav.max_vel
        uav.pos += uav.vel * dt
        uav.trajectory.append(uav.pos.copy())
        
        # 记录最小安全间距
        for ped in pedestrians:
            d = np.linalg.norm(uav.pos - ped.pos) - 0.3
            uav.min_dist_to_obs = min(uav.min_dist_to_obs, d)
        for obs in static_obs:
            d = np.linalg.norm(uav.pos - np.array(obs[:2])) - obs[2]
            uav.min_dist_to_obs = min(uav.min_dist_to_obs, d)
        
        # 碰撞检测
        collision = False
        for ped in pedestrians:
            if np.linalg.norm(uav.pos - ped.pos) < uav.radius + 0.3:
                collision = True
                break
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
    
    return {
        'success': success,
        'time': round(t, 2),
        'min_dist': round(max(0, uav.min_dist_to_obs), 3),
        'trajectory': [p.tolist() for p in uav.trajectory]
    }

def run_experiments(n_trials=100):
    """跑完所有实验组"""
    configs = [
        ('commercial',  'baseline',    EGOPlanner()),
        ('commercial',  'improved',    ImprovedPlanner()),
        ('residential', 'baseline',    EGOPlanner()),
        ('residential', 'improved',    ImprovedPlanner()),
    ]
    
    all_results = {}
    
    for scene, method, planner in configs:
        key = f"{scene}_{method}"
        print(f"\n>>> 场景:{scene}  方法:{method}  共{n_trials}次")
        results = []
        
        for i in range(n_trials):
            r = run_trial(scene, planner, seed=i*42)
            results.append(r)
            status = "成功" if r['success'] else "失败"
            print(f"  [{i+1:3d}/{n_trials}] {status}  t={r['time']:.1f}s  "
                  f"min_dist={r['min_dist']:.2f}m", end='\r')
        
        n_success = sum(r['success'] for r in results)
        sr = n_success / n_trials * 100
        valid_times = [r['time'] for r in results if r['success']]
        valid_dists = [r['min_dist'] for r in results]
        
        summary = {
            'scene': scene,
            'method': method,
            'n_trials': n_trials,
            'success_rate': round(sr, 1),
            'avg_time': round(np.mean(valid_times), 2) if valid_times else None,
            'avg_min_dist': round(np.mean(valid_dists), 3),
        }
        all_results[key] = summary
        print(f"\n  成功率:{sr:.1f}%  平均时间:{summary['avg_time']}s  "
              f"平均最小间距:{summary['avg_min_dist']}m")
        
        with open(f'/home/yuan/results_{key}.json', 'w') as f:
            json.dump({'results': results, 'summary': summary}, f, indent=2)
    
    return all_results

if __name__ == '__main__':
    print("=== 无人机路径规划仿真实验 ===")
    print("场景：城镇物流配送（商业街 + 居民区）")
    print("方法：EGO-Planner 基线 vs 本文改进方法")
    print("="*40)
    
    results = run_experiments(n_trials=100)
    
    with open('/home/yuan/all_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n所有实验完成！结果已保存。")
