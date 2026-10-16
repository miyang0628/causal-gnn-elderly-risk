# Runs the five novelty-forward stages in order (base training -> faithfulness ->
# interpretability -> leakage -> personalisation -> figures). Each stage is a
# subprocess so a failure is isolated and logged. Run from anywhere:
#     python run_all.py
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
stages = ["exp_train_base.py", "exp1_faithfulness.py", "exp2_interpret.py",
          "exp3_leakage.py", "exp4_personalized.py", "fig_new.py"]
for s in stages:
    t0 = time.time()
    print(f"\n{'='*60}\n### RUN {s}\n{'='*60}", flush=True)
    r = subprocess.run([sys.executable, "-u", str(HERE / s)], cwd=str(HERE))
    print(f"### DONE {s} rc={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        print(f"!!! STAGE FAILED: {s} — continuing to next stage", flush=True)
print("\n@@@ ALL STAGES COMPLETE", flush=True)
