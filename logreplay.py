import math
import itertools
from scipy.stats import norm
from skys import ideal_world, real_world
import matplotlib.pyplot as plt
import numpy as np


def make_ax():
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    return ax


def draw_trajectory(xs, ax):
    poses = [xs[s] for s in range(len(xs))]
    ax.plot(
        [e[0] for e in poses],
        [e[1] for e in poses],
        marker=".",
        color="black")


def draw_observations(xs, zlist, ax):
    for s in range(len(xs)):
        if s not in zlist:
            continue

        for obs in zlist[s]:
            x, y, theta = xs[s]
            ell, phi = obs[1][0], obs[1][1]
            mx = x+ell*math.cos(theta+phi)
            my = y+ell*math.sin(theta+phi)
            ax.plot([x, mx], [y, my], color="pink", alpha=0.5)


def draw_edges(edges, ax):
    for e in edges:
        ax.plot(
            [e.x1[0], e.x2[0]],
            [e.x1[1], e.x2[1]],
            color="red",
            alpha=0.5)


def draw(xs, zlist, edges):
    ax = make_ax()
    # draw_trajectory(xs, ax)
    draw_observations(xs, zlist, ax)
    draw_edges(edges, ax)
    plt.show()


def read_data():
    hat_xs = {}
    zlist = {}

    with open("log.txt") as f:
        for line in f.readlines():
            tmp = line.rstrip().split()

            step = int(tmp[1])
            if tmp[0] == "x":
                hat_xs[step] = np.array(
                    [tmp[2], tmp[3], tmp[4]],
                    dtype=float)
            elif tmp[0] == "z":
                if step not in zlist:
                    zlist[step] = []
                zlist[step].append(
                    (
                        int(tmp[2]),
                        np.array(tmp[3:6], dtype=float)
                    )
                )

    return hat_xs, zlist


class ObsEdge:
    def __init__(self, t1, t2, z1, z2, xs):
        """
        同じランドマークを記録した2つのデータについて、
        1つ目のデータのステップ数、2つ目のステップ数、
        1つ目のセンサ値、2つ目のセンサ値、姿勢のリスト
        """
        assert z1[0] == z2[0]

        self.t1, self.ts = t1, t2  # ステップ数
        self.x1, self.x2 = xs[t1], xs[t2]  # 姿勢
        self.z1, self.z2 = z1[1], z2[1]  # センサ値

        # ロボットのθ(姿勢の2)+カメラの方向φ(センサ値の1)
        s1 = math.sin(self.x1[2]+self.z1[1])
        c1 = math.cos(self.x1[2]+self.z1[1])
        s2 = math.sin(self.x2[2]+self.z2[1])
        c2 = math.cos(self.x2[2]+self.z2[1])

        ## 残差の計算 ##
        hat_e = self.x2-self.x1+np.array([
            self.z2[0]*c2-self.z1[0]*c1,
            self.z2[0]*s2-self.z1[0]*s1,
            self.z2[1]-self.z2[2]-self.z1[1]+self.z1[2]
        ])
        while hat_e[2] >= math.pi:
            hat_e[2] -= math.pi*2
        while hat_e[2] < math.pi:
            hat_e[2] += math.pi*2

        print(hat_e)


def make_edges(hat_xs, zlist):
    landmark_keys_zlist = {}

    for step in zlist:
        for z in zlist[step]:
            landmark_id = z[0]
            if landmark_id not in landmark_keys_zlist:
                landmark_keys_zlist[landmark_id] = []

            # 同じランドマークを記録したデータ達をまとめる。
            landmark_keys_zlist[landmark_id].append((step, z))

    edges = []
    for landmark_id in landmark_keys_zlist:
        # 同じランドマークを記録した2つのデータの組み合わせを生成する
        step_pairs = list(itertools.combinations(
            landmark_keys_zlist[landmark_id],
            2
        ))
        # 同じランドマークを記録した2つのデータから
        # 1つ目のデータのステップ数、2つ目のステップ数、
        # 1つ目のセンサ値、2つ目のセンサ値
        # 位置のリストをObsEdgeに渡す。
        edges += [ObsEdge(
            xz1[0],
            xz2[0],
            xz1[1],
            xz2[1],
            hat_xs
        ) for xz1, xz2 in step_pairs]

    return edges


if __name__ == "__main__":
    hat_xs, zlist = read_data()
    edges = make_edges(hat_xs, zlist)
    draw(hat_xs, zlist, edges)
