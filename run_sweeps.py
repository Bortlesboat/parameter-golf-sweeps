#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXACT_METRIC_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
CODE_BYTES_RE = re.compile(r"Code size:\s*(?P<bytes>\d+)\s*bytes")
TOTAL_BYTES_RE = re.compile(r"Total submission size int8\+zlib:\s*(?P<bytes>\d+)\s*bytes")


@dataclass
class RunResult:
    name: str
    exit_code: int
    log_path: Path
    val_loss: float | None
    val_bpb: float | None
    bytes_code: int | None
    bytes_total: int | None


def resolve_path(raw: str | None, base: Path) -> Path:
    if raw is None:
        return base
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def parse_log(log_text: str) -> tuple[float | None, float | None, int | None, int | None]:
    metric = EXACT_METRIC_RE.search(log_text)
    code = CODE_BYTES_RE.search(log_text)
    total = TOTAL_BYTES_RE.search(log_text)
    return (
        float(metric.group("val_loss")) if metric else None,
        float(metric.group("val_bpb")) if metric else None,
        int(code.group("bytes")) if code else None,
        int(total.group("bytes")) if total else None,
    )


def load_config(config_path: Path) -> dict[str, object]:
    payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a JSON object.")
    if "command" not in payload:
        raise ValueError("Config must include a command list.")
    if "runs" not in payload:
        raise ValueError("Config must include a runs list.")
    return payload


def run_one(
    *,
    run_index: int,
    workdir: Path,
    log_dir: Path,
    command: list[str],
    base_env: dict[str, str],
    run_cfg: dict[str, object],
) -> RunResult:
    name = str(run_cfg.get("name") or f"run_{run_index:02d}")
    env_overrides = {str(key): str(value) for key, value in dict(run_cfg.get("env") or {}).items()}
    merged_env = os.environ.copy()
    merged_env.update(base_env)
    merged_env.update(env_overrides)
    log_path = log_dir / f"{name}.log"

    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write(f"$ {' '.join(command)}\n")
        process = subprocess.run(  # noqa: S603
            command,
            cwd=str(workdir),
            env=merged_env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    log_text = log_path.read_text(encoding="utf-8")
    val_loss, val_bpb, bytes_code, bytes_total = parse_log(log_text)
    return RunResult(
        name=name,
        exit_code=process.returncode,
        log_path=log_path,
        val_loss=val_loss,
        val_bpb=val_bpb,
        bytes_code=bytes_code,
        bytes_total=bytes_total,
    )


def sort_results(results: list[RunResult]) -> list[RunResult]:
    return sorted(
        results,
        key=lambda item: (
            item.val_bpb is None,
            item.val_bpb if item.val_bpb is not None else math.inf,
            item.exit_code,
            item.name,
        ),
    )


def write_csv(results: list[RunResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name", "exit_code", "log_path", "val_loss", "val_bpb", "bytes_code", "bytes_total"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "name": item.name,
                    "exit_code": item.exit_code,
                    "log_path": item.log_path.name,
                    "val_loss": "" if item.val_loss is None else f"{item.val_loss:.8f}",
                    "val_bpb": "" if item.val_bpb is None else f"{item.val_bpb:.8f}",
                    "bytes_code": "" if item.bytes_code is None else item.bytes_code,
                    "bytes_total": "" if item.bytes_total is None else item.bytes_total,
                }
            )


def write_markdown(results: list[RunResult], output_path: Path) -> None:
    lines = [
        "# Sweep Summary",
        "",
        "| name | exit_code | val_loss | val_bpb | bytes_code | bytes_total | log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            "| {name} | {exit_code} | {val_loss} | {val_bpb} | {bytes_code} | {bytes_total} | {log} |".format(
                name=item.name,
                exit_code=item.exit_code,
                val_loss="" if item.val_loss is None else f"{item.val_loss:.8f}",
                val_bpb="" if item.val_bpb is None else f"{item.val_bpb:.8f}",
                bytes_code="" if item.bytes_code is None else item.bytes_code,
                bytes_total="" if item.bytes_total is None else item.bytes_total,
                log=item.log_path.name,
            )
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small Parameter Golf sweeps from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to the sweep JSON config.")
    parser.add_argument("--output-dir", help="Optional override for the output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    workdir = resolve_path(str(config.get("workdir") or "."), config_dir)
    log_dir = resolve_path(args.output_dir or str(config.get("log_dir") or "sweep_runs"), workdir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(config["command"], list):
        print("error: command must be a JSON array.", file=sys.stderr)
        return 1
    if not isinstance(config["runs"], list):
        print("error: runs must be a JSON array.", file=sys.stderr)
        return 1

    command = [str(part) for part in config["command"]]
    base_env = {str(key): str(value) for key, value in dict(config.get("base_env") or {}).items()}
    runs = config["runs"]

    if not workdir.is_dir():
        print(f"error: workdir does not exist: {workdir}", file=sys.stderr)
        return 1
    if not runs:
        print("error: config contains no runs.", file=sys.stderr)
        return 1

    results: list[RunResult] = []
    for run_index, raw_run in enumerate(runs, start=1):
        if not isinstance(raw_run, dict):
            print("error: each run entry must be a JSON object.", file=sys.stderr)
            return 1
        result = run_one(
            run_index=run_index,
            workdir=workdir,
            log_dir=log_dir,
            command=command,
            base_env=base_env,
            run_cfg=raw_run,
        )
        print(
            "run={name} exit_code={exit_code} val_bpb={val_bpb} bytes_total={bytes_total} log={log}".format(
                name=result.name,
                exit_code=result.exit_code,
                val_bpb="" if result.val_bpb is None else f"{result.val_bpb:.8f}",
                bytes_total="" if result.bytes_total is None else result.bytes_total,
                log=result.log_path,
            )
        )
        results.append(result)

    sorted_results = sort_results(results)
    write_csv(sorted_results, log_dir / "summary.csv")
    write_markdown(sorted_results, log_dir / "summary.md")

    return 0 if all(item.exit_code == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
