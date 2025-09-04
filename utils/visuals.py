import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.eval_path = os.path.join(project_root, "outputs", "evaluations")
        self.visuals_path = os.path.join(project_root, "outputs", "visuals")

        # Ensure outputs/visuals exists
        os.makedirs(self.visuals_path, exist_ok=True)

    # -----------------------
    # Data loading utilities
    # -----------------------
    def load_equity_csv(self, filename: str) -> np.ndarray:
        path = os.path.join(self.eval_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        if "equity" in df.columns:
            eq = df["equity"].astype(float).values
        else:
            eq = df.iloc[:, 0].astype(float).values
        return np.asarray(eq, dtype=float)

    # -----------------------
    # Metrics
    # -----------------------
    @staticmethod
    def compute_drawdown(equity: np.ndarray) -> np.ndarray:
        cummax = np.maximum.accumulate(equity)
        return (equity - cummax) / cummax

    @staticmethod
    def compute_rolling_returns(equity: np.ndarray, window: int = 30) -> pd.Series:
        returns = pd.Series(equity).pct_change().fillna(0)
        return returns.rolling(window).mean()

    @staticmethod
    def compute_cumulative_returns(equity: np.ndarray) -> np.ndarray:
        return (equity / equity[0] - 1) * 100

    # -----------------------
    # Plotting
    # -----------------------
    def save_plot(self, fig, name: str):
        out_path = os.path.join(self.visuals_path, f"{name}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def plot_equity_curves(self, baseline, dqn, ppo):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(baseline, label="Baseline (Buy & Hold)", linewidth=2)
        ax.plot(dqn, label="DQN Agent", linewidth=2)
        ax.plot(ppo, label="PPO Agent", linewidth=2)

        ax.set_title("Equity Curve Comparison", fontsize=14)
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Portfolio Value", fontsize=12)
        ax.legend()
        ax.grid(True)

        self.save_plot(fig, "equity_curves")

    def plot_rolling_and_drawdown(self, baseline, dqn, ppo):
        # Rolling returns
        baseline_roll = self.compute_rolling_returns(baseline)
        dqn_roll = self.compute_rolling_returns(dqn)
        ppo_roll = self.compute_rolling_returns(ppo)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(baseline_roll, label="Baseline Rolling Return", alpha=0.7)
        ax.plot(dqn_roll, label="DQN Rolling Return", alpha=0.7)
        ax.plot(ppo_roll, label="PPO Rolling Return", alpha=0.7)
        ax.set_title("Rolling Returns (30-step window)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True)
        self.save_plot(fig, "rolling_returns")

        # Drawdowns
        baseline_dd = self.compute_drawdown(baseline)
        dqn_dd = self.compute_drawdown(dqn)
        ppo_dd = self.compute_drawdown(ppo)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(baseline_dd, label="Baseline Drawdown", alpha=0.7)
        ax.plot(dqn_dd, label="DQN Drawdown", alpha=0.7)
        ax.plot(ppo_dd, label="PPO Drawdown", alpha=0.7)
        ax.set_title("Drawdowns")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Drawdown")
        ax.legend()
        ax.grid(True)
        self.save_plot(fig, "drawdowns")

    def plot_cumulative_returns(self, baseline, dqn, ppo):
        baseline_cum = self.compute_cumulative_returns(baseline)
        dqn_cum = self.compute_cumulative_returns(dqn)
        ppo_cum = self.compute_cumulative_returns(ppo)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(baseline_cum, label="Baseline (Buy & Hold)")
        ax.plot(dqn_cum, label="DQN Agent")
        ax.plot(ppo_cum, label="PPO Agent")
        ax.set_title("Cumulative Returns (%)")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        ax.grid(True)
        self.save_plot(fig, "cumulative_returns")

        # Save table as CSV
        final_returns = {
            "Baseline": baseline_cum[-1],
            "DQN": dqn_cum[-1],
            "PPO": ppo_cum[-1]
        }
        df = pd.DataFrame(final_returns, index=["Final Cumulative Return (%)"]).T
        df.to_csv(os.path.join(self.visuals_path, "final_cumulative_returns.csv"))
        return df
