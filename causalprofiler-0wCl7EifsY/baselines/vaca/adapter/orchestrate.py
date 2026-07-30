"""
Top-level orchestrator (run under any Python with just stdlib -- no torch or
causal_profiler needed here). Ties together the two-process VACA<->
CausalProfiler bridge:

  1. gen_task.py   (py3.11 CausalProfiler venv)  -> generates one SCM sample
     + queries for a given (space, seed, run) and serializes to JSON.
  2. run_vaca.py   (py3.9 `vaca` conda env)       -> trains VACA on that task
     and answers its queries `num_tries` times, writing a run-level result
     JSON matching examples/evaluation/evaluate.py's per-run result schema.

This mirrors the harness's own evaluate() loop structure (seed -> run ->
try), except training happens once per run (as required for VACA, which
needs to fit a fresh model to each run's fresh SCM) rather than reusing a
single long-lived method instance across runs, which the stock evaluate.py
harness does for stateless methods but which cannot work for VACA.

Usage:
    python orchestrate.py --space linear_medium --seeds 1,2,3 --num_runs 2 \
        --num_tries 3 --out_dir ../results
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CP_PYTHON = "/home/rec1/Desktop/AI_Safety/ICML_reproduce/.venv/bin/python"
VACA_CONDA_PYTHON = "/home/rec1/anaconda3/envs/vaca/bin/python"
GEN_TASK = os.path.join(HERE, "gen_task.py")
RUN_VACA = os.path.join(HERE, "run_vaca.py")


def run_one(space, seed, run, out_dir, n_samples, node_min, node_max,
            number_of_queries, num_tries, max_epochs, min_epochs, batch_size):
    tasks_dir = os.path.join(out_dir, "tasks")
    os.makedirs(tasks_dir, exist_ok=True)
    tag = f"{space}_seed{seed}_run{run}"
    task_path = os.path.join(tasks_dir, f"task_{tag}.json")
    result_path = os.path.join(tasks_dir, f"result_{tag}.json")

    gen_cmd = [
        CP_PYTHON, GEN_TASK,
        "--space", space, "--seed", str(seed), "--run", str(run),
        "--out", task_path, "--n_samples", str(n_samples),
        "--number_of_nodes_min", str(node_min),
        "--number_of_nodes_max", str(node_max),
        "--number_of_queries", str(number_of_queries),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(gen_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[orchestrate] gen_task FAILED for {tag}:\n{r.stderr}", file=sys.stderr)
        return None
    gen_time = time.perf_counter() - t0

    vaca_cmd = [
        VACA_CONDA_PYTHON, RUN_VACA,
        "--task", task_path, "--out", result_path,
        "--num_tries", str(num_tries),
        "--max_epochs", str(max_epochs), "--min_epochs", str(min_epochs),
        "--batch_size", str(batch_size), "--seed", str(seed),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(vaca_cmd, capture_output=True, text=True)
    vaca_time = time.perf_counter() - t0
    if r.returncode != 0:
        print(f"[orchestrate] run_vaca FAILED for {tag}:\n{r.stderr[-4000:]}", file=sys.stderr)
        return None

    with open(result_path) as f:
        result = json.load(f)
    result["gen_task_time_s"] = gen_time
    result["run_vaca_wall_time_s"] = vaca_time
    print(
        f"[{tag}] error_mean={result['run_error_mean']:.4f} "
        f"failures_mean={result['run_failures_mean']:.2f} "
        f"train_failed={result.get('train_failed')} "
        f"(gen {gen_time:.1f}s, vaca {vaca_time:.1f}s)"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, choices=["linear_medium", "nn_medium"])
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--num_runs", type=int, default=2)
    ap.add_argument("--num_tries", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=800)
    ap.add_argument("--number_of_nodes_min", type=int, default=5)
    ap.add_argument("--number_of_nodes_max", type=int, default=6)
    ap.add_argument("--number_of_queries", type=int, default=3)
    ap.add_argument("--max_epochs", type=int, default=15)
    ap.add_argument("--min_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "..", "results"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    all_results = []
    t_start = time.perf_counter()
    for seed in seeds:
        for run in range(args.num_runs):
            res = run_one(
                args.space, seed, run, out_dir,
                n_samples=args.n_samples,
                node_min=args.number_of_nodes_min, node_max=args.number_of_nodes_max,
                number_of_queries=args.number_of_queries,
                num_tries=args.num_tries,
                max_epochs=args.max_epochs, min_epochs=args.min_epochs,
                batch_size=args.batch_size,
            )
            if res is not None:
                all_results.append(res)

    total_time = time.perf_counter() - t_start
    summary_path = os.path.join(out_dir, f"summary_{args.space}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    n_ok = len(all_results)
    n_total = len(seeds) * args.num_runs
    print(f"\n=== {args.space}: {n_ok}/{n_total} runs completed in {total_time:.1f}s ===")
    if n_ok > 0:
        import statistics
        means = [r["run_error_mean"] for r in all_results]
        fails = [r["run_failures_mean"] for r in all_results]
        print(f"Mean error across runs: {statistics.mean(means):.4f} (std {statistics.pstdev(means):.4f})")
        print(f"Mean failures/run across runs: {statistics.mean(fails):.4f}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
