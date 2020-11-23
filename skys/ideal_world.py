"""
examples:
    from skys import ideal_world
    import numpy as np

    # シミュレートする世界。時間10秒間で、0.5秒ごとに状態遷移
    world=ideal_world.World(
        time_span=10,
        time_interval=0.5
    )

    # ランドマークを配置するための地図、そしてランドマークを追加
    m=ideal_world.Map()
    m.append_landmark(ideal_world.Landmark(2,-2))
    m.append_landmark(ideal_world.Landmark(-1,-3))
    m.append_landmark(ideal_world.Landmark(3,3))
    world.append(m)

    # 制御するエージェント
    straight=ideal_world.Agent(0.1,0.0) #直進
    circling=ideal_world.Agent(0.1,10.0/180*np.pi) #円運動

    # ロボット
    robot1=ideal_world.IdealRobot(
        np.array([2,3,np.pi/6]),
        agent=straight,
        sensor=ideal_world.IdealCamera(m))
    robot2=ideal_world.IdealRobot(
        np.array([-2,-1,np.pi*5/6]),
        agent=circling,
        sensor=ideal_world.IdealCamera(m),
        color="red")
    robot3=ideal_world.IdealRobot(
        np.array([0,0,0]),
        color="blue")

    world.append(robot1) 
    world.append(robot2) 
    world.append(robot3) 
    world.draw()

issue: 
    上田氏は縦ベクトルを表すためnp.array([...]).T
    と.Tをつけているが、1次元のnp.arrayを.Tをつけても
    何も変わらない。
"""
import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np



class World:
    def __init__(
            self,
            time_span,
            time_interval,
            debug=False):
        self.objects = []
        self.time_span = time_span
        self.time_interval = time_interval
        self.debug = debug
        self.first_frame = True

    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        elems = []

        if self.debug:
            for i in range(100):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(
                fig,
                self.one_step,
                fargs=(elems, ax),
                frames=int(self.time_span/self.time_interval),
                interval=int(self.time_interval*1000),
                repeat=False)
            plt.show()

    def one_step(self, i, elems, ax):
        # そのフレームが1番最初のフレームだったら、one_stepを実行しない
        if self.first_frame:
            self.first_frame = False
            return
        while elems:
            elems.pop().remove()
        time_text = ax.text(
            -4.4,
            4.5,
            "t = {:.2f}".format(self.time_interval*i),
            fontsize=10)
        elems.append(time_text)
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)


class IdealRobot:
    def __init__(
            self,
            pose,
            agent=None,
            sensor=None,
            color="black"):

        self.pose = pose
        self.r = 0.2
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    def one_step(self, time_interval):
        """
        ロボットはセンサーで測定してから動き、
        その次のフレームでdrawが走るので、
        センサーの値は1フレーム前のものとなる。
        """
        if not self.agent:
            return
        if self.sensor:
            obs = self.sensor.data(self.pose)
        else:
            obs = None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(
            nu,
            omega,
            time_interval,
            self.pose)

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x+self.r*math.cos(theta)
        yn = y+self.r*math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color=self.color)
        c = patches.Circle(
            xy=(x, y),
            radius=self.r,
            fill=False,
            color=self.color)
        elems.append(ax.add_patch(c))

        self.poses.append(self.pose)
        elems += ax.plot(
            [e[0] for e in self.poses],
            [e[1] for e in self.poses],
            linewidth=0.5,
            color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([
                nu*math.cos(t0),
                nu*math.sin(t0),
                omega])*time

        else:
            return pose + np.array([
                nu/omega*(math.sin(t0+omega*time)-math.sin(t0)),
                nu/omega*(-math.cos(t0+omega*time)+math.cos(t0)),
                omega * time
            ])


class IdealCamera:
    def __init__(
            self,
            env_map,
            distance_range=(0.5, 6.0),
            direction_range=(-math.pi/3, math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] \
            and self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos-cam_pose[0:2]
        phi = math.atan2(diff[1], diff[0]) - cam_pose[2]
        while phi >= math.pi:
            phi -= 2*math.pi
        while phi <= -math.pi:
            phi += 2*math.pi
        return np.array([math.hypot(*diff), phi]).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color="pink")


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega


class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        c = ax.scatter(
            self.pos[0],
            self.pos[1],
            s=100,
            marker="*",
            label="landmarks",
            color="orange"
        )
        text = ax.text(
            self.pos[0],
            self.pos[1],
            "id{:0=2}".format(self.id),
            fontsize=10
        )
        elems.append(c)
        elems.append(text)


class Map:
    def __init__(self):
        self.landmarks = []

    def append_landmark(self, landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)
