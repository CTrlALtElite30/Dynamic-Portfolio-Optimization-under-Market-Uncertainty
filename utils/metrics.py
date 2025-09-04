# for project i run this code (implementation 2nd time)
import numpy as np

def equity_to_metrics(equity_curve):
    n_days = len(equity_curve)
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    CAGR = (equity_curve[-1] / equity_curve[0]) ** (252 / n_days) - 1

    returns = np.diff(equity_curve) / equity_curve[:-1]
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = CAGR / (volatility + 1e-8)
    max_dd = np.min(equity_curve / np.maximum.accumulate(equity_curve) - 1) * 100
    calmar = CAGR / (abs(max_dd / 100) + 1e-8)
    downside = returns[returns < 0]
    sortino = CAGR / (np.std(downside) * np.sqrt(252) + 1e-8)

    return {
        "total_return_pct": total_return,
        "CAGR_pct": CAGR * 100,
        "volatility_ann_pct": volatility * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "calmar": calmar,
        "sortino": sortino,
        "n_days": n_days
    }

