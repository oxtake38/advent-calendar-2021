import random

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
import pandas as pd

random.seed(0)
np.random.seed(0)

N_ITERATION_POISSON = 50


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        # 初期化を実施
        self.cavity_flow = CavityFlow()
        _, s, _ = self.cavity_flow.get_info()

        self.num_action = 1
        self.num_states = len(s)

        self.max_episode_timesteps = self.cavity_flow.NT

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.num_action,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(self.num_states,))

        self.steps = 0

        self.TARGET_V_CENTER = 0.2

    def reset(self) -> np.ndarray:
        """環境をリセットする｡リセット自体はシーケンスに従って実行されるためダミー関数である｡

        Returns:
            np.ndarray: [description]
        """
        # 初期化を実施
        self.steps = 0
        self.cavity_flow = CavityFlow()

        # Return env information
        terminal, states, v_center = self.cavity_flow.get_info()

        # クリッピングと正規化
        states = np.clip(self.cavity_flow.scaler_states.transform(states.reshape(1, -1)).flatten(), 0.0, 1.0)

        self.states = pd.DataFrame(columns=[f"state-{i}" for i in range(self.num_states)])
        self.actions = pd.DataFrame(columns=[f"action-{i}" for i in range(self.num_action)])
        self.rewards = pd.DataFrame(columns=["reward"])
        self.targets = pd.DataFrame(columns=["target"])

        self.reward_sum = 0.0

        return states

    def step(self, actions: np.ndarray):
        """ at -> st+1, rt+1, tt+1
        """
        actions = self.cavity_flow.scaler_actions.inverse_transform(actions.reshape(1, -1)).flatten()

        self.cavity_flow.step(actions)

        # 環境から次ステップの情報を得る
        terminal, states, v_center = self.cavity_flow.get_info()

        # 報酬を計算する
        if self.TARGET_V_CENTER * 0.9 <= v_center <= self.TARGET_V_CENTER * 1.1:
            reward = 1.0
        else:
            reward = 0.0
        # reward = -25 * v_center**2 + 10 * v_center

        self.states.loc[self.steps, :] = states
        self.actions.loc[self.steps, :] = actions
        self.rewards.loc[self.steps, :] = [reward]
        self.targets.loc[self.steps, :] = [v_center]

        # クリッピングと正規化
        states = np.clip(self.cavity_flow.scaler_states.transform(states.reshape(1, -1)).flatten(), 0.0, 1.0)
        if terminal:
            print(self.reward_sum)
        # 次ステップへの準備
        self.steps += 1
        self.reward_sum += reward

        return states, reward, terminal, {"states": self.states,
                                          "actions": self.actions,
                                          "rewards": self.rewards,
                                          "targets": self.targets, }

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class CavityFlow:
    """Cavity Flow with Navier–Stokes
    https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/14_Step_11.ipynb

    """
    MAX_VIN = 5.0
    NX = 41
    NY = 41
    NT = 250
    RHO = 1
    NU = .1
    DT = .005
    SENSOR = np.array([[10, 10], [10, 20], [10, 30], [10, 38], [20, 10], [20, 30], [20, 38], ])
    STATES_MIN_MAX = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    ACTIONS_MIN_MAX = np.array([[0.0, 5.0]])
    TX = 20
    TY = 20

    def __init__(self):
        self.time = 0
        self.u, self.v = np.zeros((self.NY, self.NX)), np.zeros((self.NY, self.NX))
        self.p, self.b = np.zeros((self.NY, self.NX)), np.zeros((self.NY, self.NX))
        self.X, self.Y = np.meshgrid(np.linspace(0, 2, self.NX), np.linspace(0, 2, self.NY))
        self.DX, self.DY = 2 / (self.NX - 1), 2 / (self.NY - 1)
        self.PX, self.PY = self.SENSOR.T[0], self.SENSOR.T[1]

        self.__scaler_states = MinMaxScaler(feature_range=(-1, 1))
        self.__scaler_states.fit(self.STATES_MIN_MAX)
        self.__scaler_actions = MinMaxScaler(feature_range=(-1, 1))
        self.__scaler_actions.fit(self.ACTIONS_MIN_MAX.T)

    def step(self, actions: np.ndarray):
        un = self.u.copy()
        vn = self.v.copy()

        self.b = self.build_up_b(self.b, self.RHO, self.DT, self.u, self.v, self.DX, self.DY)
        self.p = self.pressure_poisson(self.p, self.DX, self.DY, self.b)

        self.u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                              un[1:-1, 1:-1] * self.DT / self.DX *
                              (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                              vn[1:-1, 1:-1] * self.DT / self.DY *
                              (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                              self.DT / (2 * self.RHO * self.DX) * (self.p[1:-1, 2:] - self.p[1:-1, 0:-2]) +
                              self.NU * (self.DT / self.DX**2 *
                                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                                         self.DT / self.DY**2 *
                                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        self.v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                              un[1:-1, 1:-1] * self.DT / self.DX *
                              (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                              vn[1:-1, 1:-1] * self.DT / self.DY *
                              (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                              self.DT / (2 * self.RHO * self.DY) * (self.p[2:, 1:-1] - self.p[0:-2, 1:-1]) +
                              self.NU * (self.DT / self.DX**2 *
                                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                                         self.DT / self.DY**2 *
                                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        self.u[0, :] = 0
        self.u[:, 0] = 0
        self.u[:, -1] = 0
        self.u[-1, :] = actions    # set velocity on cavity lid equal to 1
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.v[:, 0] = 0
        self.v[:, -1] = 0

        self.time += self.DT

    def get_info(self):
        terminal = self.time + self.DT * 0.05 >= self.NT * self.DT
        states = np.zeros(0)
        states = np.concatenate([states,
                                np.linalg.norm(np.vstack((self.u[self.PX, self.PY], self.v[self.PX, self.PY])).T, ord=2, axis=1)])
        states = np.hstack([states, np.array(self.time)])
        velocity = np.linalg.norm([self.u[self.TX, self.TY], self.v[self.TX, self.TY]], ord=2)
        return terminal, states.astype(float).flatten(), float(velocity)

    @staticmethod
    def build_up_b(b, rho, dt, u, v, dx, dy):
        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) /
                                 (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

        return b

    @staticmethod
    def pressure_poisson(p, dx, dy, b):
        pn = np.empty_like(p)
        pn = p.copy()

        for _ in range(N_ITERATION_POISSON):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                              (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                             (2 * (dx**2 + dy**2)) -
                             dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                             b[1:-1, 1:-1])

            p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
            p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
            p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
            p[-1, :] = 0        # p = 0 at y = 2

        return p

    @property
    def scaler_states(self):
        return self.__scaler_states

    @property
    def scaler_actions(self):
        return self.__scaler_actions
