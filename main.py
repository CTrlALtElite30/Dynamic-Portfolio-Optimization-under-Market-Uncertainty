# here only do that code which takes a dataset as input and activates your project all code files and gives output as result. saves plots,files,csv etc all.

# main.py
import argparse
import os
import sys
import subprocess
from pathlib import Path

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports from your utils ---
# Make sure utils/visuals.py defines Visualizer (you already have this)
try:
    from utils.visuals import Visualizer
except Exception as e:
    Visualizer = None
    print("Note: Could not import Visualizer from utils.visuals:", e)

# ---------- Helpers ----------
def run_notebook(nb_path: Path):
    """
    Execute a Jupyter notebook in-place (saves outputs back into the notebook).
    Requires jupyter/nbconvert installed in the active environment.
    """
    nb_path = nb_path.resolve()
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")
    print(f"Executing notebook: {nb_path}")
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--inplace",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        str(nb_path)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)  # ensures imports like utils.* work

    # Show live output
    subprocess.run(cmd, check=True)
    print(f"Finished: {nb_path.name}\n")

def train(alg: str, use_best: bool):
    """
    Train agents by running your existing training notebooks.
    alg: 'dqn' | 'ppo' | 'both'
    use_best: if True, run the *best hyperparameters* notebooks
    """
    agents_dir = PROJECT_ROOT / "agents"

    if alg in ("dqn", "both"):
        nb = ("dqn_best_hyperparameters.ipynb" if use_best
              else "train_dqn_agent.ipynb")
        run_notebook(agents_dir / nb)

    if alg in ("ppo", "both"):
        nb = ("ppo_best_hyperparameters.ipynb" if use_best
              else "train_ppo_agent.ipynb")
        run_notebook(agents_dir / nb)

def evaluate(which: str):
    """
    Evaluate agents by running your existing evaluation notebooks.
    which: 'baseline' | 'dqn' | 'ppo' | 'all'
    """
    agents_dir = PROJECT_ROOT / "agents"

    if which in ("baseline", "all"):
        run_notebook(agents_dir / "evaluate_baseline.ipynb")

    if which in ("dqn", "all"):
        run_notebook(agents_dir / "evaluate_dqn.ipynb")

    if which in ("ppo", "all"):
        run_notebook(agents_dir / "evaluate_ppo.ipynb")

def visualize(mode: str):
    """
    Generate plots & metrics either through your notebook(s) or directly via utils.visuals.Visualizer.
    mode: 'summary' (Visualizer) | 'all' (run visuals + final_comparison notebooks)
    """
    # Fast path: use your utils.visuals.Visualizer (no notebook execution)
    if mode in ("summary", "all"):
        if Visualizer is None:
            print("Visualizer not available; skipping summary visuals.")
        else:
            print("Generating summary visuals with utils.visuals.Visualizer ...")
            viz = Visualizer(str(PROJECT_ROOT))
            # Load saved evaluation CSVs
            baseline_eq = viz.load_equity_csv("baseline_buy_and_hold_equity_test.csv")
            dqn_eq      = viz.load_equity_csv("dqn_discrete_equity_test.csv")
            ppo_eq      = viz.load_equity_csv("ppo_continuous_equity_test.csv")
            # Create/save plots & metrics
            viz.plot_equity_curves(baseline_eq, dqn_eq, ppo_eq)
            viz.plot_rolling_and_drawdown(baseline_eq, dqn_eq, ppo_eq)
            final_df = viz.plot_cumulative_returns(baseline_eq, dqn_eq, ppo_eq)
            print("\nSummary metrics (Cumulative % returns):")
            print(final_df)

    # Full visuals: also execute notebooks
    if mode == "all":
        nb_dir = PROJECT_ROOT / "notebooks"
        for nb in ("visuals.ipynb", "final_comparison.ipynb"):
            run_notebook(nb_dir / nb)

def main():
    parser = argparse.ArgumentParser(
        description="DRL Trading — unified entry point for training, evaluation, and visualization."
    )

    # Mutually independent switches so you can mix & match
    parser.add_argument("--train", choices=["dqn", "ppo", "both"],
                        help="Train selected agent(s).")
    parser.add_argument("--use-best", action="store_true",
                        help="Use best hyperparameters notebooks for training.")
    parser.add_argument("--eval", choices=["baseline", "dqn", "ppo", "all"],
                        help="Evaluate selected agent(s).")
    parser.add_argument("--viz", choices=["summary", "all"],
                        help="Generate visualizations. 'summary' uses utils.visuals, 'all' also runs notebooks.")
    parser.add_argument("--all", action="store_true",
                        help="Run full pipeline: train (both, best) -> evaluate all -> visualize all.")

    args = parser.parse_args()

    if args.all:
        print("\n=== FULL PIPELINE START ===\n")
        train("both", use_best=True)
        evaluate("all")
        visualize("all")
        print("\n=== FULL PIPELINE DONE ===")
        return

    if args.train:
        train(args.train, args.use_best)

    if args.eval:
        evaluate(args.eval)

    if args.viz:
        visualize(args.viz)

    # If no args, print help
    if not any([args.train, args.eval, args.viz, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()
