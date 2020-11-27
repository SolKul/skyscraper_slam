import math
from scipy.stats import norm
from . import ideal_world, real_world


class PsiCamera(real_world.Camera):
    def data(self, cam_pose, orientation_noise=math.pi/90):
        observed = []
        for lm in self.map.landmarks:
            psi = norm.rvs(
                loc=math.atan2(cam_pose[1]-lm.pos[1], cam_pose[0]-lm.pos[0]),
                scale=orientation_noise)
            z = self.observation_function(cam_pose, lm.pos)
            z = self.phantom(cam_pose, z)
            z = self.occlusion(z)
            z = self.oversight(z)
            if self.visible(z):
                z = self.bias(z)
                z = self.noise(z)
                observed.append(
                    ([z[0], z[1], psi], lm.id))

        self.lastdata = observed
        return observed


class LoggerAgent(ideal_world.Agent):
    """
    one_step→drawなので
    最後に描写された状態まで記録される。
    """

    def __init__(
            self,
            nu,
            omega,
            interval_time,
            init_pose):
        super().__init__(nu, omega)
        self.interval_time = interval_time
        self.pose = init_pose
        self.step = 0
        self.log = open("log.txt", "w")

    def decision(self, observation):
        if len(observation) != 0:
            pose_log = "x {} {} {} {}\n".format(
                self.step,
                *self.pose)
            self.log.write(pose_log)
            for obs in observation:
                sensor_log="z {} {} {} {} {}\n".format(
                    self.step,
                    obs[1], 
                    *obs[0])
                self.log.write(sensor_log)

            self.step += 1
            self.log.flush()

        self.pose = ideal_world.IdealRobot.state_transition(
            self.nu,
            self.omega,
            self.interval_time,
            self.pose)

        return self.nu, self.omega
