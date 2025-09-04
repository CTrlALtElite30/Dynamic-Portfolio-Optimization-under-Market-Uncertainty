# Building and Trading Environment for Continuous
# continuous env:
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnvContinuous(gym.Env):
    """Continuous trading environment with portfolio allocation in [-1, 1]."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, states, prices, initial_cash=100000, reward_type="log_return"):
        super().__init__()
        self.states = states
        self.prices = prices
        self.n_steps = len(prices)

        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.allocation = 0.0
        self.current_step = 0
        self.reward_type = reward_type

        # Continuous action: allocation in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # Observations are state vectors
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(states.shape[1],), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.portfolio_value = self.initial_cash
        self.allocation = 0.0
        self.current_step = 0
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        prev_value = self.portfolio_value
        price_t = self.prices[self.current_step]
        price_prev = self.prices[max(self.current_step - 1, 0)]

        # Clip action into [-1,1]
        self.allocation = float(np.clip(action[0], -1.0, 1.0))

        # Update portfolio value based on return and allocation
        price_return = (price_t / price_prev) - 1.0
        self.portfolio_value *= (1 + self.allocation * price_return)

        self.current_step += 1
        terminated = (self.current_step >= self.n_steps - 1)
        truncated = False

        curr_value = self.portfolio_value

        # --- Reward ---
        if self.reward_type == "log_return":
            reward = np.log((curr_value + 1e-8) / (prev_value + 1e-8))
        elif self.reward_type == "simple":
            reward = (curr_value - prev_value) / (prev_value + 1e-8)
        else:  # absolute PnL
            reward = curr_value - prev_value

        # Clip reward for stability
        reward = float(np.clip(reward, -1.0, 1.0))

        obs = self._get_observation()
        info = {"allocation": self.allocation, "portfolio_value": curr_value}

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        return self.states[self.current_step]

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Allocation: {self.allocation:.2f}, "
              f"Portfolio Value: {self.portfolio_value:.2f}")
