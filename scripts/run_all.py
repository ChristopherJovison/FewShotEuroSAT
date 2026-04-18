from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import get_dataset_name, slugify


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full EuroSAT benchmark.")
    parser.add_argument("--config", default="configs/benchmark_all.yaml")
    parser.add_argument("--skip_existing", action="store_true", help="Skip method/shot/seed combinations that already have completed metrics.")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as f:
        benchmark = yaml.safe_load(f)["benchmark"]
    for shots in benchmark["shots"]:
        for seed in benchmark["seeds"]:
            for cfg in benchmark["configs"]:
                if args.skip_existing and completed_run_exists(Path(cfg), shots, seed):
                    print(f"Skipping completed run: {cfg} --shots {shots} --seed {seed}")
                    continue
                cmd = [sys.executable, "scripts/run_experiment.py", "--config", cfg, "--shots", str(shots), "--seed", str(seed)]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


def completed_run_exists(config_path: Path, shots: int, seed: int) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp = config["experiment"]
    dataset_name = get_dataset_name(config)
    method = exp["method"]
    output_root = Path(exp.get("output_root", "outputs")) / slugify(dataset_name)
    pattern = f"{method}_shot{shots}_seed{seed}_*"
    scoped_exists = any((run_dir / "metrics.json").exists() for run_dir in output_root.glob(pattern) if run_dir.is_dir())
    if scoped_exists:
        return True
    return False


if __name__ == "__main__":
    main()
