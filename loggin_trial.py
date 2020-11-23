from skys import ideal_world, real_world, graph_slam
import math
import numpy as np

time_interval = 3
world = ideal_world.World(
    time_span=180,
    time_interval=time_interval,
    debug=False
)

m = ideal_world.Map()
landmark_positions=[(-4, 2), (2, -3), (3, 3), (0, 4), (1, 1), (-3, -1)]
for p in landmark_positions:
    m.append_landmark(ideal_world.Landmark(*p))
    
world.append(m)

### ロボットを作る ###
init_pose=np.array([0,-3,0])
a = graph_slam.LoggerAgent(
    0.2,
    5.0/180*math.pi,
    time_interval,
    init_pose
)
r = real_world.Robot(
    init_pose,
    sensor=graph_slam.PsiCamera(m),
    agent=a,
    color="red"
)

world.append(r)

world.draw()