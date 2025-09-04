# Building and Trading Environment for descrete -  problem(SB3 solved) :
# Discrete Env

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnvDiscrete(gym.Env):
    """A simple discrete Buy/Hold/Sell trading environment for a single asset."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, states, prices, initial_cash=100000, reward_type="log_return"):
        super().__init__()
        self.states = states
        self.prices = prices
        self.n_steps = len(prices)

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.current_step = 0
        self.reward_type = reward_type

        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        # Observations are state vectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(states.shape[1],), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cash = self.initial_cash
        self.position = 0
        self.current_step = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        prev_value = self._get_portfolio_value()

        price = self.prices[self.current_step]
        if action == 1 and self.cash >= price:
            self.position += 1
            self.cash -= price
        elif action == 2 and self.position > 0:
            self.position -= 1
            self.cash += price


        self.current_step += 1
        terminated = (self.current_step >= self.n_steps - 1)
        truncated = False

        curr_value = self._get_portfolio_value()

        if self.reward_type == "log_return":
            reward = float(np.log(curr_value / (prev_value + 1e-8)))
        elif self.reward_type == "simple":
            reward = float((curr_value - prev_value) / (prev_value + 1e-8))
        else:
            reward = float(curr_value - prev_value)

        obs = self._get_observation()
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # ensure dtype matches observation_space
        return self.states[self.current_step].astype(np.float32)

    def _get_portfolio_value(self):
        price = self.prices[min(self.current_step, self.n_steps - 1)]
        return self.cash + self.position * price


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, "
              f"Pos: {self.position}, Value: {self._get_portfolio_value():.2f}")
