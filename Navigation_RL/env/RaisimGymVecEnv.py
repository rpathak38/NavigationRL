# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, clip_obs=10., is_random=True):
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.wrapper.init()
        self.num_obs = self.wrapper.getObDim()
        self.num_value_obs = self.wrapper.getValueObDim()
        self.num_acts = self.wrapper.getActionDim()
        self.num_rewards = self.wrapper.getNumOfRewards()
        self.num_nav_obs = self.wrapper.getNavObDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._value_observation = np.zeros([self.num_envs, self.num_value_obs], dtype=np.float32)
        self._nav_observation = np.zeros([self.num_envs, self.num_nav_obs], dtype=np.float32)
        self._action = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.value_obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_value_obs])
        self.nav_obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_nav_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self._rewards = np.zeros([self.num_envs, self.num_rewards], dtype=np.float32)
        self._rewards_stdev = np.zeros(self.num_rewards, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def set_command(self, command):
        self.wrapper.setCommand(command)

    def map_change(self):
        self.wrapper.mapChange()

    def nav_observe(self, update_mean=True):
        self.wrapper.navObserve(self._nav_observation)
        if self.normalize_ob:
            if update_mean:
                self.nav_obs_rms.update(self._nav_observation)
            return self._normalize_nav_observation(self._nav_observation)
        else:
            return self._nav_observation.copy()

    def nav_reset(self):
        self.wrapper.navReset()

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def getrewards(self):
        self.wrapper.getRewards(self._rewards, self._rewards_stdev)
        return self._rewards.copy(), self._rewards_stdev.copy()

    def get_logging_reward(self):
        self.wrapper.getLoggingReward(self._reward)
        return self._reward.copy()

    def load_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.mean = np.tile(np.loadtxt(mean_file_name, dtype=np.float32), (self.obs_rms.mean.shape[0], 1))
        self.obs_rms.var = np.tile(np.loadtxt(var_file_name, dtype=np.float32), (self.obs_rms.var.shape[0], 1))

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean[0])
        np.savetxt(var_file_name, self.obs_rms.var[0])

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)

        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    def value_observe(self, update_mean=True):
        self.wrapper.valueObserve(self._value_observation)

        if self.normalize_ob:
            if update_mean:
                self.value_obs_rms.update(self._value_observation)

            return self._normalize_value_observation(self._value_observation)
        else:
            return self._value_observation.copy()

    def reset(self):
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def _normalize_value_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.value_obs_rms.mean) / np.sqrt(self.value_obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def _normalize_nav_observation(self, obs):
        if self.normalize_ob:
            return np.clip((obs - self.nav_obs_rms.mean) / np.sqrt(self.nav_obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()



class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
