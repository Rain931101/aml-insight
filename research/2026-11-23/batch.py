"""Bounded local orchestration; no silent retries or overwriting successful/failed runs."""

import concurrent.futures
import subprocess
import sys
import time

from core import HERE, PLAN, REPORT, ROOT, RUNS, write


def job(item):
    model, variant, seed = item
    name = f"{model}-{variant}-seed{seed}"
    if (RUNS / name).exists():
        return {"job": name, "status": "existing_not_rerun"}
    log = REPORT / f"{name}.log"
    start = time.perf_counter()
    with log.open("x") as stream:
        r = subprocess.run(
            [sys.executable, str(HERE / "run_amlsim.py"), model, variant, str(seed)],
            check=False,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    result = {
        "job": name,
        "returncode": r.returncode,
        "seconds": time.perf_counter() - start,
        "log": str(log.relative_to(ROOT)),
    }
    print(result, flush=True)
    return result


if __name__ == "__main__":
    REPORT.mkdir(exist_ok=True)
    jobs = [("gnn", v, s) for v in PLAN["variants"] for s in PLAN["seeds"]]
    jobs += [("tabular", v, s) for v in PLAN["xgb_variants"] for s in PLAN["seeds"]]
    # Two workers bound memory/CPU on the shared laptop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(job, jobs))
    write(REPORT / "batch_receipt.json", result)
    if any(r.get("returncode", 0) for r in result):
        raise SystemExit("Some jobs failed; inspect logs, no automatic retries")
