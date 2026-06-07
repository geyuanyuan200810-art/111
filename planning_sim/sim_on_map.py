import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family']='WenQuanYi Zen Hei'
plt.rcParams['axes.unicode_minus']=False

occ_b=np.load('/home/yuan/planning_sim/map_baseline.npy')
occ_s=np.load('/home/yuan/planning_sim/map_semantic.npy')
with open('/home/yuan/planning_sim/map_params.json') as f: mp=json.load(f)
RES=mp['resolution']; X_MIN=mp['x_min']; Y_MIN=mp['y_min']
W=mp['width']; H=mp['height']

def w2g(x,y): return int((x-X_MIN)/RES), int((y-Y_MIN)/RES)

def is_occ(occ,x,y,r=0.8):
    xi,yi=w2g(x,y); rc=int(r/RES)+1
    x0,x1=max(0,xi-rc),min(W,xi+rc)
    y0,y1=max(0,yi-rc),min(H,yi+rc)
    if x0>=x1 or y0>=y1: return False
    return bool(occ[y0:y1,x0:x1].any())

def obs_force(occ,x,y,r=5.0,ds=1.0):
    """向量化障碍物排斥力"""
    xi,yi=w2g(x,y); rc=int(r/RES)
    x0,x1=max(0,xi-rc),min(W,xi+rc)
    y0,y1=max(0,yi-rc),min(H,yi+rc)
    if x0>=x1 or y0>=y1: return np.zeros(2)
    patch=occ[y0:y1,x0:x1]
    if not patch.any(): return np.zeros(2)
    rows,cols=np.where(patch)
    wx=(cols+x0)*RES+X_MIN+RES/2
    wy=(rows+y0)*RES+Y_MIN+RES/2
    diff=np.array([x,y])-np.stack([wx,wy],1)
    d=np.linalg.norm(diff,axis=1)-0.25
    mask=(d>0.01)&(d<ds*2)
    if not mask.any(): return np.zeros(2)
    dn=diff[mask]/np.linalg.norm(diff[mask],axis=1,keepdims=True).clip(1e-9)
    return (3.0*(ds*2-d[mask])[:,None]*dn).sum(0).clip(-8,8)

class Pedestrian:
    def __init__(self,x,y,vx,vy):
        self.pos=np.array([x,y],dtype=float)
        self.vel=np.array([vx,vy],dtype=float)
    def update(self,dt,occ):
        n=self.pos+self.vel*dt
        if not is_occ(occ,n[0],n[1],0.4): self.pos=n
        else: self.vel*=-1
        self.vel+=np.random.randn(2)*0.05
        s=np.linalg.norm(self.vel)
        if s>1.5: self.vel=self.vel/s*1.5
        if s<0.3: self.vel=self.vel/max(s,1e-9)*0.3

class EGOPlanner:
    name='EGO-Planner（基线）'
    def compute_vel(self,pos,vel,goal,peds,occ):
        f=np.zeros(2)
        tg=goal-pos; dg=np.linalg.norm(tg)
        if dg>0: f+=3.0*tg/dg
        f+=obs_force(occ,pos[0],pos[1],ds=1.2)
        for ped in peds:
            d2=pos-ped.pos; d=np.linalg.norm(d2)-0.3
            if 0<d<1.0: f+=2.5*(1.0-d)*d2/max(np.linalg.norm(d2),1e-9)
        nv=vel+f*0.05; s=np.linalg.norm(nv)
        return nv/s*3.0 if s>3.0 else nv

class DWAPlanner:
    name='DWA'
    def compute_vel(self,pos,vel,goal,peds,occ):
        s=np.linalg.norm(vel)
        ha=np.arctan2(vel[1],vel[0]) if s>0.1 else np.arctan2(goal[1]-pos[1],goal[0]-pos[0])
        # 向量化：预生成所有速度采样
        vs=np.linspace(max(0,s-0.4),min(3.0,s+0.4),6)
        ws=np.linspace(-1.8,1.8,9)
        VV,WW=np.meshgrid(vs,ws); VV=VV.ravel(); WW=WW.ravel()
        best_score=-1e9; best_vel=vel.copy()
        dt=0.05; steps=20
        for vi,wi in zip(VV,WW):
            px,py,ang=pos[0],pos[1],ha
            mind=1e9; ok=True
            for _ in range(steps):
                ang+=wi*dt; px+=vi*np.cos(ang)*dt; py+=vi*np.sin(ang)*dt
                if is_occ(occ,px,py,0.7): ok=False; break
                for ped in peds:
                    d=np.linalg.norm(np.array([px,py])-ped.pos)
                    if d<0.8: ok=False; break
                    mind=min(mind,d)
                if not ok: break
            if not ok: continue
            h=np.cos(np.arctan2(goal[1]-py,goal[0]-px)-ang)
            sc=0.5*h+0.35*min(mind,3.0)/3.0+0.15*vi/3.0
            if sc>best_score:
                best_score=sc
                best_vel=np.array([vi*np.cos(ang),vi*np.sin(ang)])
        return best_vel

class ImprovedPlanner:
    name='本文改进方法'
    def rg(self,pos,pp,pv):
        s=np.linalg.norm(pv); sp=0.7+0.3*s; sn=0.7
        vd=pv/s if s>0.01 else np.array([1.,0.])
        nd=np.array([-vd[1],vd[0]]); df=pos-pp
        dp=np.dot(df,vd); dn=np.dot(df,nd)
        r=np.exp(-0.5*((dp/sp)**2+(dn/sn)**2))
        g=-r*np.array([dp/sp**2*vd[0]+dn/sn**2*nd[0],
                        dp/sp**2*vd[1]+dn/sn**2*nd[1]])
        return r,g
    def compute_vel(self,pos,vel,goal,peds,occ):
        f=np.zeros(2)
        tg=goal-pos; dg=np.linalg.norm(tg)
        if dg>0: f+=3.0*tg/dg
        f+=obs_force(occ,pos[0],pos[1],ds=1.2)
        for ped in peds:
            _,g=self.rg(pos,ped.pos,ped.vel)
            rv=vel-ped.vel; tu=pos-ped.pos
            d=max(np.linalg.norm(tu),0.01)
            w=1.0+0.5*max(0,-np.dot(rv,tu/d))
            f-=1.5*w*g*5.0
            for tau in [0.5,1.0,1.5]:
                _,gp=self.rg(pos,ped.pos+ped.vel*tau,ped.vel)
                f-=1.5*w*gp*3.0*np.exp(-tau/1.5)
        nv=vel+f*0.05; s=np.linalg.norm(nv)
        return nv/s*3.0 if s>3.0 else nv

def run_trial(occ,planner,start,goal,n_peds,seed=0):
    np.random.seed(seed); rng=np.random.default_rng(seed)
    peds=[]
    for _ in range(300):
        if len(peds)>=n_peds: break
        px=rng.uniform(start[0]+20,goal[0]-20)
        py=rng.uniform(-20,20)
        if not is_occ(occ,px,py,1.2):
            s=rng.uniform(0.6,1.4); a=rng.uniform(0,2*np.pi)
            peds.append(Pedestrian(px,py,s*np.cos(a),s*np.sin(a)))
    pos=np.array(start,dtype=float); vel=np.zeros(2)
    traj=[pos.copy()]; mind=1e9; dt=0.05; success=False
    goal_arr=np.array(goal)
    for step in range(int(80/dt)):
        for ped in peds: ped.update(dt,occ)
        vel=planner.compute_vel(pos,vel,goal_arr,peds,occ)
        pos=pos+vel*dt; traj.append(pos.copy())
        for ped in peds: mind=min(mind,np.linalg.norm(pos-ped.pos)-0.3)
        if is_occ(occ,pos[0],pos[1],0.6): break
        for ped in peds:
            if np.linalg.norm(pos-ped.pos)<0.8: break
        if np.linalg.norm(pos-goal_arr)<3.0: success=True; break
    return {'success':success,'min_dist':round(max(0,mind),3),
            'time':round(step*dt,2),'traj':[p.tolist() for p in traj[::5]]}

START=[155.0,0.0]; GOAL=[365.0,0.0]
N_PEDS=6; N_TRIALS=100
planners=[EGOPlanner(),DWAPlanner(),ImprovedPlanner()]
maps=[('基线SLAM地图',occ_b),('语义SLAM地图',occ_s)]
all_results={}

for map_name,occ in maps:
    for planner in planners:
        key=f"{map_name}_{planner.name}"
        print(f"\n>>> {map_name} | {planner.name}")
        results=[]
        for i in range(N_TRIALS):
            r=run_trial(occ,planner,START,GOAL,N_PEDS,seed=i)
            results.append(r)
            print(f"  [{i+1:3d}/{N_TRIALS}] {'✓' if r['success'] else '✗'} "
                  f"min_d={r['min_dist']:.2f}m",end='\r')
        sr=sum(r['success'] for r in results)/N_TRIALS*100
        md=np.mean([r['min_dist'] for r in results])
        times=[r['time'] for r in results if r['success']]
        at=np.mean(times) if times else 0
        print(f"\n  成功率:{sr:.1f}%  安全间距:{md:.3f}m  平均时间:{at:.1f}s")
        all_results[key]={'map':map_name,'planner':planner.name,
            'success_rate':round(sr,1),'avg_min_dist':round(md,3),
            'avg_time':round(at,2),'trajs':[r['traj'] for r in results[:8]]}

with open('/home/yuan/planning_sim/kitti_map_results.json','w') as f:
    json.dump(all_results,f,indent=2,ensure_ascii=False)
print("\n实验完成！")

# ── 出图 ─────────────────────────────────────────────────────────────
BG='#0d0d0d'; WHITE='#f1f5f9'; GRAY='#9ca3af'; GOLD='#fbbf24'
COLORS={'EGO-Planner（基线）':'#f87171','DWA':'#fb923c','本文改进方法':'#34d399'}
pnames=['EGO-Planner（基线）','DWA','本文改进方法']
mkeys=['基线SLAM地图','语义SLAM地图']
mlabels=['基线SLAM地图\n（含动态鬼影）','语义SLAM地图\n（动态点过滤）']

fig,axes=plt.subplots(1,2,figsize=(16,6),facecolor=BG)
fig.suptitle('基于真实KITTI地图的路径规划性能对比\n（城市街道场景，3种规划方法×2种地图）',
             color=WHITE,fontsize=13,fontweight='bold',y=1.04)
x=np.arange(2); bw=0.25

for ax,(metric,ylabel,mtitle) in zip(axes,[
    ('success_rate','任务成功率 (%)','任务成功率对比'),
    ('avg_min_dist','平均最小安全间距 (m)','安全间距对比')]):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY,labelsize=9)
    ax.set_ylabel(ylabel,color=GRAY,fontsize=10)
    ax.set_title(mtitle,color=GOLD,fontsize=11,fontweight='bold')
    ax.grid(True,axis='y',color='#1f2937',lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(mlabels,color=WHITE,fontsize=9)
    for j,pname in enumerate(pnames):
        vals=[all_results[f"{mk}_{pname}"][metric] for mk in mkeys]
        bars=ax.bar(x+(j-1)*bw,vals,bw,color=COLORS[pname],
                    alpha=0.85,label=pname,edgecolor='#1f2937')
        for bar,val in zip(bars,vals):
            fmt=f'{val:.1f}%' if metric=='success_rate' else f'{val:.3f}m'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+(1.5 if metric=='success_rate' else 0.008),
                    fmt,ha='center',va='bottom',color=WHITE,fontsize=8,fontweight='bold')
    ax.legend(facecolor='#111827',edgecolor='#374151',labelcolor=WHITE,fontsize=8)

plt.tight_layout()
plt.savefig('/home/yuan/planning_sim/kitti_results_bar.png',
            dpi=160,bbox_inches='tight',facecolor=BG)

# 图2：轨迹对比
fig2,axes2=plt.subplots(1,2,figsize=(18,7),facecolor=BG)
fig2.suptitle('真实KITTI地图路径规划轨迹对比',color=WHITE,fontsize=12,fontweight='bold')
px0,px1,py0,py1=130,390,-55,55
xi0=int((px0-X_MIN)/RES); xi1=int((px1-X_MIN)/RES)
yi0=int((py0-Y_MIN)/RES); yi1=int((py1-Y_MIN)/RES)

for ax,(map_name,occ),mk in zip(axes2,maps,mkeys):
    ax.set_facecolor('#111827')
    for sp in ax.spines.values(): sp.set_color('#374151')
    ax.tick_params(colors=GRAY,labelsize=8)
    patch=occ[yi0:yi1,xi0:xi1]
    ax.imshow(patch,origin='lower',cmap='Greens',alpha=0.45,
              extent=[px0,px1,py0,py1],aspect='auto')
    for pname,color,ls,lw in [
        ('EGO-Planner（基线）','#f87171','--',1.0),
        ('DWA','#fb923c','-.',1.0),
        ('本文改进方法','#34d399','-',1.5)]:
        key=f"{mk}_{pname}"
        for ti,traj in enumerate(all_results[key]['trajs'][:5]):
            t=np.array(traj)
            if len(t)>1:
                label=pname if ti==0 else None
                ax.plot(t[:,0],t[:,1],lw=lw,alpha=0.6,
                        color=color,linestyle=ls,label=label)
    ax.plot(START[0],START[1],'s',color='#60a5fa',ms=10,zorder=10,label='起点')
    ax.plot(GOAL[0], GOAL[1], '*',color='#fbbf24',ms=12,zorder=10,label='终点')
    sr_imp=all_results[f"{mk}_本文改进方法"]['success_rate']
    sr_ego=all_results[f"{mk}_EGO-Planner（基线）"]['success_rate']
    sr_dwa=all_results[f"{mk}_DWA"]['success_rate']
    ax.set_title(f'{map_name}\n改进:{sr_imp}%  DWA:{sr_dwa}%  EGO:{sr_ego}%',
                 color=WHITE,fontsize=9,fontweight='bold')
    ax.set_xlabel('X (m)',color=GRAY); ax.set_ylabel('Y (m)',color=GRAY)
    ax.set_xlim(px0,px1); ax.set_ylim(py0,py1)
    handles,labels=ax.get_legend_handles_labels()
    by_label=dict(zip(labels,handles))
    ax.legend(by_label.values(),by_label.keys(),
              facecolor='#111827',edgecolor='#374151',labelcolor=WHITE,fontsize=8)

plt.tight_layout()
plt.savefig('/home/yuan/planning_sim/kitti_traj_compare.png',
            dpi=160,bbox_inches='tight',facecolor=BG)
print("图已保存")
