import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_agent(env, model=None, name="agent", save_dir="outputs/evaluations"):
    """
    Evaluate an RL agent or baseline strategy.
    Auto-saves results as CSV, JSON, and PNG.

    Args:
        env: Trading environment (must return info['portfolio_value'])
        model: Trained RL model (if None, uses random/baseline policy)
        name: Name of the evaluation (e.g., 'dqn_discrete', 'ppo_continuous')
        save_dir: Root output directory
    """

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join("outputs", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    obs, _ = env.reset()
    done, truncated = False, False
    portfolio_values = []

    while not (done or truncated):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()  # fallback
        obs, reward, done, truncated, info = env.step(action)
        portfolio_values.append(info.get("portfolio_value", getattr(env, "portfolio_value", np.nan)))

    # ---------- Convert to Series ----------
    equity = pd.Series(portfolio_values, index=np.arange(len(portfolio_values)))

    # ---------- Metrics ----------
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_days = len(equity)
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / n_days) - 1) * 100
    daily_returns = equity.pct_change().dropna()
    vol_ann = daily_returns.std() * np.sqrt(252) * 100
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * np.sqrt(252)
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min() * 100
    calmar = cagr / (-max_dd + 1e-8) if max_dd < 0 else None
    downside = daily_returns[daily_returns < 0]
    sortino = daily_returns.mean() / (downside.std() + 1e-8) * np.sqrt(252)

    metrics = {
        "total_return_pct": float(total_return),
        "CAGR_pct": float(cagr),
        "volatility_ann_pct": float(vol_ann),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "calmar": float(calmar) if calmar is not None else None,
        "sortino": float(sortino),
        "n_days": n_days
    }

    # ---------- Save ----------
    csv_path = os.path.join(save_dir, f"{name}_equity.csv")
    json_path = os.path.join(save_dir, f"{name}_metrics.json")
    png_path = os.path.join(plot_dir, f"{name}_equity.png")

    equity.to_csv(csv_path, header=["equity"])
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity.values, label=f"{name} Equity")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value")
    plt.title(f"Equity Curve: {name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(png_path)
    plt.close()

    print(f"✅ {name} Evaluation saved")
    print(f"CSV: {csv_path}")
    print(f"JSON metrics: {json_path}")
    print(f"PNG plot: {png_path}")
    print("Metrics:", metrics)

    return metrics
