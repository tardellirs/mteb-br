"""Quickstart: evaluate any embedding model on a single MTEB-PT task.

Usage::

    pip install mteb-pt
    python examples/quickstart.py --model intfloat/multilingual-e5-base --task HateBR

Runs in a few minutes on CPU for the default ``HateBR`` task. Results are cached
in the mteb results cache (``MTEB_CACHE`` env var, default ``~/.cache/mteb``).
"""

from __future__ import annotations

import argparse

import mteb

import mteb_pt.register  # noqa: F401  -- side-effect import: registers tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single MTEB-PT task on an embedding model.",
    )
    parser.add_argument(
        "--model",
        default="intfloat/multilingual-e5-base",
        help="HuggingFace model identifier (default: intfloat/multilingual-e5-base)",
    )
    parser.add_argument(
        "--task",
        default="HateBR",
        help="MTEB-PT task name (default: HateBR)",
    )
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Task:  {args.task}\n")

    model = mteb.get_model(args.model)
    task = mteb.get_task(args.task)
    results = mteb.evaluate(model, tasks=[task])

    print("\nDone. Results cached in the mteb results cache (MTEB_CACHE / ~/.cache/mteb).")
    print(results)


if __name__ == "__main__":
    main()
