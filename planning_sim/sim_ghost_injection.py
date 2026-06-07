"""
鬼影注入实验：验证SLAM建图精度对路径规划的影响
基线SLAM地图：静态障碍 + 鬼影虚假障碍
语义SLAM地图：仅静态障碍（鬼影已过滤）
3种规划方法 × 2种地图 = 6组实验，每组100次
"""
import numpy as np, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']='WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus']=False

# ── 从KITTI地图提取鬼影格位置 ─────────────────────────────────────────
occ_b=np.load('/home/yuan/planning_sim/map_baseline.npy')
occ_s=np.load('/home/yuan/planning_sim/map_semantic.npy')
with open('/home/yuan/planning_sim/map_params.json') as f: mp=json.load(f)
RES=mp['resolution']; X_MIN=mp['x_min']; Y_MIN=mp['y_min']

ghost=(occ_b.astype(int)-occ_s.astype(int)).clip(0).astype(bool)
rows,cols=np.where(ghost)
ghost_wx=cols*RES+X_MIN+RES/2
ghost_wy=rows*RES+Y_MIN+RES/2

# 只取X=200-250（鬼影最密集区域）的鬼影格
mask=(ghost_wx>=200)&(ghost_wx<=250)&(np.abs(ghost_wy)<=30)
gx=ghost_wx[mask]; gy=ghost_wy[mask]
print(f"选取鬼影格: {len(gx)}个（X=200-250区段）")

# 缩放到商业街坐标系（X:-20~20, Y:-3.5~3.5）
# 把鬼影格的相对分布保留，映射到商业街通道
gx_norm=(gx-gx.min())/(gx.max()-gx.min()+1e-9)  # 0~1
gy_norm=(gy-gy.min())/(gy.max()-gy.min()+1e-9)  # 0~1
# 映射到X=-12到12，Y=-3到3（商业街中段）
ghost_sim_x = gx_norm * 24 - 12
ghost_sim_y = gy_norm * 6  - 3

# 聚类降采样，保留有代表性的鬼影簇（避免太密集导致完全堵路）
from sklearn.cluster import KMeans
if len(ghost_sim_x) > 30:
    coords=np.stack([ghost_sim_x,ghost_sim_y],1)
    km=KMeans(n_clusters=20, random_state=42, n_init=5)
    km.fit(coords)
    ghost_clusters=km.cluster_centers_  # 20个鬼影簇中心
else:
    ghost_clusters=np.stack([ghost_sim_x,ghost_sim_y],1)

print(f"鬼影簇数量: {len(ghost_clusters)}个")
print(f"鬼影簇位置示例: {ghost_clusters[:3].round(2)}")

# ── 商业街仿真场景 ────────────────────────────────────────────────────
class Pedestrian:
    def __init__(self,x,y,vx,vy):
        self.pos=np.array([x,y],dtype=float)
        self.vel=np.array([vx,vy],dtype=float)
        self.base_vel=self.vel.copy()
    def update(self,dt,bounds):
        self.pos+=self.vel*dt
        for i in range(2):
            if self.pos[i]<bounds[i][0] or self.pos[i]>bounds[i][1]:
                self.vel[i]*=-1
                self.pos[i]=np.clip(self.pos[i],bounds[i][0],bounds[i][1])
        self.vel+=np.random.randn(2)*0.08
        s=np.linalg.norm(self.vel); bs=np.linalg.norm(self.base_vel)
        if s>0: self.vel=self.vel/s*np.clip(s,0.8,bs*1.5)

def make_scene(seed, ghost_obs):
    """生成商业街场景：固定静态障碍 + 可选鬼影障碍 + 随机行人"""
    np.random.seed(seed)
    start=[-18.0,0.0]; goal=[18.0,0.0]
    bounds=[[-20,20],[-4,4]]

    # 固定静态障碍（真实存在的建筑柱子）
    static_real=[
        [-5.0, 3.5, 0.8],[5.0,-3.5,0.8],
        [0.0,  3.8, 0.6],[-10.0,-3.5,0.7],
    ]

    # 鬼影虚假障碍（只在基线地图场景中存在）
    static_all = static_real.copy()
    if ghost_obs is not None:
        for gp in ghost_obs:
            # 鬼影格作为小障碍（半径0.3m，比真实障碍小）
            static_all.append([float(gp[0]), float(gp[1]), 0.3])

    # 随机行人
    n_peds=np.random.randint(5,9)  # 增加行人数量
    peds=[]
    for _ in range(n_peds):
        x=np.random.uniform(-15,15); y=np.random.uniform(-3,3)
        # 提高行人速度（1.2-2.2m/s），让方向性更强
        vx=np.random.choice([-1,1])*np.random.uniform(1.2,2.2)
        vy=np.random.uniform(-0.5,0.5)
        peds.append(Pedestrian(x,y,vx,vy))

    return start, goal, bounds, static_all, static_real, peds

# ── 三种规划方法 ──────────────────────────────────────────────────────
class EGOPlanner:
    name='EGO-Planner（基线）'
    def __init__(self): self.d_safe=0.8
    def compute_force(self,uav_pos,uav_vel,goal,peds,static_obs):
        f=np.zeros(2)
        tg=np.array(goal)-uav_pos; dg=np.linalg.norm(tg)
        if dg>0: f+=2.0*tg/dg
        for obs in static_obs:
            d2=uav_pos-np.array(obs[:2])
            d=np.linalg.norm(d2)-obs[2]
            if 0<d<self.d_safe*2:
                f+=3.0*(self.d_safe*2-d)*d2/np.linalg.norm(d2)
        for ped in peds:
            d2=uav_pos-ped.pos; d=np.linalg.norm(d2)-0.3
            if 0<d<self.d_safe:
                f+=2.0*(self.d_safe-d)*d2/np.linalg.norm(d2)
        return f

class DWAPlanner:
    name='DWA'
    def compute_force(self,uav_pos,uav_vel,goal,peds,static_obs):
        s=np.linalg.norm(uav_vel)
        ha=np.arctan2(uav_vel[1],uav_vel[0]) if s>0.1 else            np.arctan2(goal[1]-uav_pos[1],goal[0]-uav_pos[0])
        vmin=max(0.2,s-0.4); vmax=min(3.0,s+0.4); dt=0.05; steps=15
        VS=np.linspace(vmin,vmax,5); WS=np.linspace(-1.8,1.8,7)
        VV,WW=np.meshgrid(VS,WS); VV=VV.ravel(); WW=WW.ravel()
        N=len(VV)
        # 向量化模拟所有轨迹
        px=np.full(N,uav_pos[0]); py=np.full(N,uav_pos[1])
        ang=np.full(N,ha); valid=np.ones(N,bool); mind=np.full(N,1e9)
        obs_pos=np.array([[o[0],o[1]] for o in static_obs])
        obs_r  =np.array([o[2]+0.5 for o in static_obs])
        ped_pos=np.array([p.pos for p in peds]) if peds else np.zeros((0,2))
        for _ in range(steps):
            ang+=WW*dt; px+=VV*np.cos(ang)*dt; py+=VV*np.sin(ang)*dt
            pts=np.stack([px,py],1)  # (N,2)
            # 障碍物碰撞
            if len(obs_pos):
                d_obs=np.linalg.norm(pts[:,None]-obs_pos[None],axis=2) # (N,nobs)
                hit=(d_obs<obs_r[None]).any(axis=1)
                valid&=~hit
            # 行人碰撞
            if len(ped_pos):
                d_ped=np.linalg.norm(pts[:,None]-ped_pos[None],axis=2) # (N,nped)
                hit_ped=(d_ped<0.8).any(axis=1)
                valid&=~hit_ped
                mind=np.minimum(mind,d_ped.min(axis=1) if len(ped_pos) else mind)
        if not valid.any():
            # 无可行轨迹：直接转向目标
            tg=np.array(goal)-uav_pos; dg=np.linalg.norm(tg)
            return 2.0*tg/dg if dg>0 else np.zeros(2)
        goal_arr=np.array(goal)
        end=np.stack([px,py],1)
        h=np.cos(np.arctan2(goal_arr[1]-py,goal_arr[0]-px)-ang)
        d=np.minimum(mind,3.0)/3.0; v=VV/3.0
        score=np.where(valid, 0.5*h+0.35*d+0.15*v, -1e9)
        best=np.argmax(score)
        best_vel=np.array([VV[best]*np.cos(ang[best]),
                           VV[best]*np.sin(ang[best])])
        return (best_vel-uav_vel)/dt

class ImprovedPlanner:
    """语义感知DWA：DWA轨迹采样 + 速度感知风险椭球评分"""
    name="本文改进方法"
    def __init__(self): self.T_pred=1.5; self.sigma0=0.7; self.alpha=0.3

    def risk_score(self, traj_pts, peds):
        """向量化风险评分：同时计算所有轨迹点对所有行人的风险"""
        if not peds or not traj_pts: return 1.0
        pts  = np.array(traj_pts)                        # (T,2)
        ppos = np.array([p.pos for p in peds])           # (P,2)
        pvel = np.array([p.vel for p in peds])           # (P,2)
        spd  = np.linalg.norm(pvel,axis=1).clip(0.01)   # (P,)
        sp   = self.sigma0 + self.alpha*spd              # (P,)
        sn   = np.full(len(peds), self.sigma0)
        vd   = pvel / spd[:,None]                        # (P,2)
        nd   = np.stack([-vd[:,1], vd[:,0]],axis=1)     # (P,2)
        # 当前位置风险 (T,P)
        df   = pts[:,None,:] - ppos[None,:,:]            # (T,P,2)
        dp   = (df * vd[None]).sum(-1)                   # (T,P)
        dn   = (df * nd[None]).sum(-1)
        risk = np.exp(-0.5*((dp/sp)**2+(dn/sn)**2))     # (T,P)
        # 预测位置风险
        for tau,w in [(0.5,0.5),(1.0,0.3),(1.5,0.2)]:
            pred = ppos + pvel*tau                       # (P,2)
            df2  = pts[:,None,:] - pred[None]
            dp2  = (df2*vd[None]).sum(-1)
            dn2  = (df2*nd[None]).sum(-1)
            risk += np.exp(-0.5*((dp2/sp)**2+(dn2/sn)**2))                     * np.exp(-tau/self.T_pred) * w
        # 最危险点的安全分
        max_risk = risk.max()
        return float(np.exp(-2.0*max_risk))

    def compute_force(self,uav_pos,uav_vel,goal,peds,static_obs):
        s=np.linalg.norm(uav_vel)
        ha=np.arctan2(uav_vel[1],uav_vel[0]) if s>0.1 else            np.arctan2(goal[1]-uav_pos[1],goal[0]-uav_pos[0])
        min_obs_d=min((np.linalg.norm(uav_pos-np.array(o[:2]))-o[2]
                       for o in static_obs),default=5.0)
        # 自适应权重：障碍物近时安全优先，远时目标导向优先
        w_safe   = 0.40 if min_obs_d > 2.5 else 0.55
        w_head   = 0.50 if min_obs_d > 2.5 else 0.35
        w_vel    = 0.10
        # 扩大速度采样范围，增加样本数
        vmin=max(0.5, s-0.6); vmax=min(3.0, s+0.6); dt=0.05; steps=18
        VS=np.linspace(vmin,vmax,7); WS=np.linspace(-2.2,2.2,11)
        VV,WW=np.meshgrid(VS,WS); VV=VV.ravel(); WW=WW.ravel()
        N=len(VV)
        px=np.full(N,uav_pos[0]); py=np.full(N,uav_pos[1])
        ang=np.full(N,ha); valid=np.ones(N,bool)
        obs_pos=np.array([[o[0],o[1]] for o in static_obs])
        obs_r  =np.array([o[2]+0.45  for o in static_obs])  # 略微收紧碰撞半径
        traj_pts=[[(uav_pos[0],uav_pos[1])] for _ in range(N)]
        for step in range(steps):
            ang+=WW*dt; px+=VV*np.cos(ang)*dt; py+=VV*np.sin(ang)*dt
            pts=np.stack([px,py],1)
            if len(obs_pos):
                d_obs=np.linalg.norm(pts[:,None]-obs_pos[None],axis=2)
                valid&=~(d_obs<obs_r[None]).any(axis=1)
            if step%3==0:
                for i in range(N):
                    if valid[i]: traj_pts[i].append((px[i],py[i]))
        if not valid.any():
            tg=np.array(goal)-uav_pos; dg=np.linalg.norm(tg)
            return 2.0*tg/dg if dg>0 else np.zeros(2)
        goal_arr=np.array(goal)
        h=np.cos(np.arctan2(goal_arr[1]-py,goal_arr[0]-px)-ang)
        # 目标进度得分：轨迹终点越接近目标越好
        end_pts=np.stack([px,py],1)
        dist_to_goal=np.linalg.norm(end_pts-goal_arr,axis=1)
        progress=1.0-np.clip(dist_to_goal/40.0,0,1)
        rsk=np.array([self.risk_score(traj_pts[i],peds) if valid[i] else 0.0
                      for i in range(N)])
        score=np.where(valid,
            w_head*h + w_safe*rsk + w_vel*VV/3.0 + 0.05*progress,
            -1e9)
        best=np.argmax(score)
        bv=np.array([VV[best]*np.cos(ang[best]),VV[best]*np.sin(ang[best])])
        return (bv-uav_vel)/dt


def run_trial(planner, ghost_obs, seed=0):
    np.random.seed(seed)
    start,goal,bounds,static_obs,static_real,peds=make_scene(seed,ghost_obs)
    pos=np.array(start,dtype=float); vel=np.zeros(2)
    traj=[pos.copy()]; mind=1e9; dt=0.05; success=False
    goal_arr=np.array(goal)
    max_vel=3.0; max_acc=2.0; radius=0.5

    for step in range(int(60/dt)):
        for ped in peds: ped.update(dt,bounds)
        force=planner.compute_force(pos,vel,goal,peds,static_obs)
        acc=np.clip(force,-max_acc,max_acc)
        vel+=acc*dt
        s=np.linalg.norm(vel)
        if s>max_vel: vel=vel/s*max_vel
        pos+=vel*dt; traj.append(pos.copy())

        # 最小安全间距（只对真实障碍计算，不算鬼影）
        for ped in peds: mind=min(mind,np.linalg.norm(pos-ped.pos)-0.3)
        for obs in static_real:
            mind=min(mind,np.linalg.norm(pos-np.array(obs[:2]))-obs[2])

        # 碰撞检测（只对真实障碍检测，鬼影不造成真实碰撞）
        col=False
        for ped in peds:
            if np.linalg.norm(pos-ped.pos)<radius+0.3: col=True; break
        for obs in static_real:
            if np.linalg.norm(pos-np.array(obs[:2]))<radius+obs[2]: col=True; break
        if col: break
        if np.linalg.norm(pos-goal_arr)<1.5: success=True; break

    return {'success':success,'min_dist':round(max(0,mind),3),
            'time':round(step*dt,2),'traj':[p.tolist() for p in traj]}

# ── 主实验 ────────────────────────────────────────────────────────────
N_TRIALS=100
planners=[EGOPlanner(),DWAPlanner(),ImprovedPlanner()]
conditions=[
    ('语义SLAM地图\n（无鬼影）', None),
    ('基线SLAM地图\n（含鬼影）', ghost_clusters),
]
all_results={}

for cond_name, ghost_obs in conditions:
    for planner in planners:
        key=f"{cond_name.split(chr(10))[0]}_{planner.name}"
        print(f"\n>>> {cond_name.split(chr(10))[0]} | {planner.name}")
        results=[]
        for i in range(N_TRIALS):
            r=run_trial(planner,ghost_obs,seed=i)
            results.append(r)
            print(f"  [{i+1:3d}/{N_TRIALS}] {'✓' if r['success'] else '✗'} "
                  f"min_d={r['min_dist']:.2f}m",end='\r')
        sr=sum(r['success'] for r in results)/N_TRIALS*100
        md=np.mean([r['min_dist'] for r in results])
        times=[r['time'] for r in results if r['success']]
        at=np.mean(times) if times else 0
        print(f"\n  成功率:{sr:.1f}%  安全间距:{md:.3f}m  平均时间:{at:.1f}s")
        all_results[key]={
            'condition':cond_name.split('\n')[0],
            'planner':planner.name,
            'success_rate':round(sr,1),
            'avg_min_dist':round(md,3),
            'avg_time':round(at,2),
            'trajs':[r['traj'] for r in results[:6]],
        }

with open('/home/yuan/planning_sim/ghost_injection_results.json','w') as f:
    json.dump(all_results,f,indent=2,ensure_ascii=False)
print("\n实验完成！")

# ── 出图 ─────────────────────────────────────────────────────────────
BG='#0d0d0d'; WHITE='#f1f5f9'; GRAY='#9ca3af'; GOLD='#fbbf24'
COLORS={'EGO-Planner（基线）':'#f87171','DWA':'#fb923c','本文改进方法':'#34d399'}
pnames=['EGO-Planner（基线）','DWA','本文改进方法']
ckeys=['语义SLAM地图','基线SLAM地图']
clabels=['语义SLAM地图\n（无鬼影）','基线SLAM地图\n（含鬼影）']

# 图1：柱状图对比
fig,axes=plt.subplots(1,2,figsize=(16,6),facecolor=BG)
fig.suptitle('SLAM建图精度对路径规划性能的影响\n（商业街场景，鬼影注入实验）',
             color=WHITE,fontsize=13,fontweight='bold',y=1.04)
x=np.arange(2); bw=0.25

for ax,(metric,ylabel,title) in zip(axes,[
    ('success_rate','任务成功率 (%)','任务成功率对比'),
    ('avg_min_dist','平均最小安全间距 (m)','安全间距对比')]):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY,labelsize=9)
    ax.set_ylabel(ylabel,color=GRAY,fontsize=10)
    ax.set_title(title,color=GOLD,fontsize=11,fontweight='bold')
    ax.grid(True,axis='y',color='#1f2937',lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(clabels,color=WHITE,fontsize=9)

    for j,pname in enumerate(pnames):
        vals=[all_results[f"{ck}_{pname}"][metric] for ck in ckeys]
        bars=ax.bar(x+(j-1)*bw,vals,bw,color=COLORS[pname],
                    alpha=0.85,label=pname,edgecolor='#1f2937')
        for bar,val in zip(bars,vals):
            fmt=f'{val:.1f}%' if metric=='success_rate' else f'{val:.3f}m'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(1.5 if metric=='success_rate' else 0.008),
                    fmt,ha='center',va='bottom',color=WHITE,fontsize=8,fontweight='bold')
    ax.legend(facecolor='#111827',edgecolor='#374151',labelcolor=WHITE,fontsize=8)

plt.tight_layout()
plt.savefig('/home/yuan/planning_sim/ghost_results_bar.png',
            dpi=160,bbox_inches='tight',facecolor=BG)
print("柱状图保存完成")

# 图2：轨迹对比图（2行3列：两种地图×三种方法）
fig2,axes2=plt.subplots(2,3,figsize=(18,10),facecolor=BG)
fig2.suptitle('商业街场景路径规划轨迹对比\n（上：语义SLAM地图  下：基线SLAM地图含鬼影）',
              color=WHITE,fontsize=12,fontweight='bold')

for row,(ckey,clabel,ghost_obs) in enumerate(zip(
    ckeys,clabels,[None,ghost_clusters])):
    for col,pname in enumerate(pnames):
        ax=axes2[row][col]; key=f"{ckey}_{pname}"
        ax.set_facecolor('#111827')
        for sp in ax.spines.values(): sp.set_color('#374151')
        ax.tick_params(colors=GRAY,labelsize=8)
        ax.set_xlim(-20,20); ax.set_ylim(-5,5)

        # 真实障碍
        real_obs=[[-5.0,3.5,0.8],[5.0,-3.5,0.8],
                  [0.0,3.8,0.6],[-10.0,-3.5,0.7]]
        for obs in real_obs:
            c=plt.Circle(obs[:2],obs[2],color='#4b5563',zorder=5)
            ax.add_patch(c)

        # 鬼影障碍（半透明橙色，表示虚假障碍）
        if ghost_obs is not None:
            for gp in ghost_obs:
                c=plt.Circle(gp,0.3,color='#f97316',alpha=0.4,zorder=4)
                ax.add_patch(c)

        # 绘制轨迹
        trajs=all_results[key]['trajs']
        for ti,traj in enumerate(trajs[:6]):
            t=np.array(traj)
            if len(t)>1:
                ax.plot(t[:,0],t[:,1],lw=1.0,alpha=0.6,
                        color=COLORS[pname])

        ax.plot(-18,0,'s',color='#60a5fa',ms=8,zorder=10)
        ax.plot( 18,0,'*',color='#fbbf24',ms=10,zorder=10)
        sr=all_results[key]['success_rate']
        md=all_results[key]['avg_min_dist']
        ax.set_title(f'{pname}\n成功率:{sr}%  间距:{md:.3f}m',
                     color=COLORS[pname],fontsize=8.5,fontweight='bold')
        ax.set_xlabel('X (m)',color=GRAY,fontsize=8)
        if col==0: ax.set_ylabel(f'{clabel.split(chr(10))[0]}\nY (m)',
                                  color=GOLD,fontsize=8)
        ax.grid(True,color='#1f2937',lw=0.3,alpha=0.5)
        ax.set_aspect('auto')

plt.tight_layout()
plt.savefig('/home/yuan/planning_sim/ghost_traj_compare.png',
            dpi=160,bbox_inches='tight',facecolor=BG)
print("轨迹图保存完成")
