from skys import ideal_world, real_world, particle
import math
import numpy as np
import copy
import pandas as pd

world = ideal_world.World(
    time_span=40,
    time_interval=0.1
    # debug=True
)

# ロボットの移動の際の雑音のパラメータ推定
# 例えば距離rごとに発生する方向θの雑音にXついて
# X~N(0,rσ^2)とする
# (分散の大きさが道のりに比例)
# σ=√(V[r]/E[r])となる。
initial_pose=np.array([0,0,0])
robots=[]
r = real_world.Robot(
    pose=initial_pose,
    sensor=None,
    agent=ideal_world.Agent(0.1,0.0))
for i in range(10):
    copy_r= copy.copy(r)
    copy_r.distance_until_noise=copy_r.noise_pdf.rvs()
    world.append(copy_r)
    robots.append(copy_r)
    
world.draw()

pose_list=[]
for r in robots:
    pose_list.append(
        [math.hypot(*r.pose[:2]),r.pose[2]])
poses=pd.DataFrame(pose_list,columns=["distance","theta"])

print(math.sqrt(poses["theta"].var()/poses["distance"].mean()))


# センサの雑音のパラメータ推定
m=ideal_world.Map()
m.append_landmark(ideal_world.Landmark(1,0))

distance=[]
direction=[]

for i in range(1000):
    c = real_world.Camera(m)
    d=c.data(np.array([0.0,0.0,0.0]))
    if len(d)>0:
        distance.append(d[0][0][0])
        direction.append(d[0][0][1])
        
observed_df=pd.DataFrame({"distance":distance,"direction":direction})

print(observed_df.std())