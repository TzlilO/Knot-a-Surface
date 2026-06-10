"""W&B sweep wrapper: injects fixed args, then execs optimize_nurbs.py.

The sweep config calls this with only the swept hyperparameters; data and
output paths come from the environment (set by run_sweep_agent.sh), so the
yaml stays portable across machines.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = os.environ.get("DTU_DATA_PATH", os.path.expanduser("~/datasets/DTU"))
out = os.environ.get(
    "SWEEP_OUT_PATH",
    os.path.expanduser(f"~/output_dtu/sweep_{os.getpid()}"),
)
iters = os.environ.get("SWEEP_ITERS", "7000")

cmd = [
    sys.executable, os.path.join(REPO, "optimize_nurbs.py"),
    "-s", data,
    "-m", out,
    "-r", "2",
    "--ncc_scale", "0.5",
    "--eval",
    "--use_wandb",
    "--iterations", iters,
] + sys.argv[1:]

print("[sweep_train] exec:", " ".join(cmd), flush=True)
os.chdir(REPO)
os.execv(sys.executable, cmd)
