"""Command-line entry point for smoke and frozen formal experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from src.config import ExperimentConfig, RunPaths, SUPPORTED_MODELS
from src.runner import run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-safe HNP/MLP Double-DQN experiment. Smoke mode "
            "uses validation only; formal mode computes cross-configuration "
            "policy outcomes only after checkpointing every selected model."
        )
    )
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model", nargs="+", choices=SUPPORTED_MODELS, dest="models"
    )
    parser.add_argument("--gamma", type=float, help="Recorded gamma override")
    parser.add_argument("--train-seeds", nargs="+", type=int)
    parser.add_argument("--eval-seeds", nargs="+", type=int)
    parser.add_argument(
        "--jammer",
        nargs="+",
        choices=("sweeping", "random"),
        dest="jammer_modes",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        help="Engineering override; the formal preset is 200 episodes",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = ExperimentConfig.preset(args.mode).with_overrides(
        models=None if args.models is None else tuple(args.models),
        gamma=args.gamma,
        train_seeds=(
            None if args.train_seeds is None else tuple(args.train_seeds)
        ),
        eval_seeds=None if args.eval_seeds is None else tuple(args.eval_seeds),
        jammer_modes=(
            None if args.jammer_modes is None else tuple(args.jammer_modes)
        ),
        train_episodes=args.train_episodes,
        device=args.device,
    )
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"{args.mode}_{stamp}"
    summary = run_experiment(
        config,
        RunPaths(data_dir=args.data_dir, output_dir=output_dir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
