"""One external worker; avoid competing with the two AMLSim training workers."""

import subprocess
import sys

from core import HERE, INPUT, REPORT, ROOT, write

if __name__ == "__main__":
    records = []
    for seed in range(5):
        path = INPUT / "elliptic/runs" / f"seed{seed}"
        if path.exists():
            records.append({"seed": seed, "status": "existing_not_retried"})
            continue
        with (REPORT / f"elliptic-seed{seed}.log").open("x") as log:
            r = subprocess.run(
                [sys.executable, str(HERE / "run_external.py"), str(seed)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        records.append({"seed": seed, "returncode": r.returncode})
        print(records[-1], flush=True)
        if r.returncode:
            break
    write(REPORT / "external_batch_receipt.json", records)
