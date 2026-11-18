"""Identify the actual Python process before framework imports, including Windows venv redirectors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

import psutil


def bootstrap_argv(argv: list[str], script: str, receipt: Path) -> list[str]:
    if receipt.exists():
        raise ValueError("Python bootstrap receipt must be fresh.")
    if len(argv) < 2 or Path(argv[1]).suffix.lower() != ".py" or not Path(argv[1]).is_file():
        raise ValueError("Python bootstrap requires a direct interpreter + script command.")
    return [argv[0], str(Path(script).resolve()), "--receipt", str(receipt.resolve()),
            "--target", str(Path(argv[1]).resolve()), "--", *argv[2:]]


def stop_launch(process) -> None:
    """Terminate only this launch and its descendants when the handshake fails."""
    try:
        for child in psutil.Process(process.pid).children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def resolve_python_process(process, receipt: Path, timeout: float = 30,
                           expected_target: Path | None = None, expected_bootstrap: Path | None = None) -> dict:
    """Validate actual PID/birth/lineage before any sampler attaches."""
    try:
        launcher = psutil.Process(process.pid)
        launcher_birth = launcher.create_time()
        deadline = time.monotonic() + timeout
        while not receipt.is_file():
            if process.poll() is not None:
                raise ValueError("Python launcher exited before its real-process bootstrap receipt.")
            if time.monotonic() >= deadline:
                raise ValueError("Actual Python PID bootstrap timed out.")
            time.sleep(.02)
        raw = receipt.read_bytes()
        record = json.loads(raw)
        pid, birth = record.get("pid"), record.get("process_create_time")
        if (record.get("schema") != "benchmark-python-process-v1" or not isinstance(pid, int)
                or isinstance(pid, bool) or pid <= 0 or not isinstance(birth, (float, int))
                or isinstance(birth, bool) or not math.isfinite(birth) or birth <= 0):
            raise ValueError("Python bootstrap PID/birth schema is invalid.")
        for label, path in (("target", expected_target), ("bootstrap", expected_bootstrap)):
            if path is not None:
                if hashlib.sha256(path.read_bytes()).hexdigest() != record.get(label + "_sha256"):
                    raise ValueError("Python " + label + " source differs from the bootstrap receipt.")
        if expected_target is not None and Path(record.get("target", "")).resolve() != expected_target.resolve():
            raise ValueError("Python bootstrap executed a different target script.")
        actual = psutil.Process(pid)
        if abs(actual.create_time() - birth) > .001:
            raise ValueError("Actual Python process birth differs from the bootstrap (PID reuse or stale evidence).")
        if pid != process.pid and process.pid not in {parent.pid for parent in actual.parents()}:
            raise ValueError("Claimed Python process is not this launch or its descendant.")
        if birth + .001 < launcher_birth:
            raise ValueError("Actual Python process predates its launcher.")
        return {"pid": pid, "process_create_time": birth, "launcher_pid": process.pid,
                "launcher_create_time": launcher_birth, "actual_process_image": actual.exe(),
                "python_process_bootstrap": {"path": str(receipt.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                                             "record": record}}
    except BaseException:
        stop_launch(process)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = args.target.resolve()
    receipt = args.receipt.resolve()
    if receipt.exists():
        raise ValueError("Refusing to replace an existing Python process receipt.")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    record = {"schema": "benchmark-python-process-v1", "pid": os.getpid(),
              "process_create_time": psutil.Process().create_time(), "written_unix_ns": time.time_ns(),
              "python_executable": sys.executable, "target": str(target),
              "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "bootstrap_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    temporary = receipt.with_name(receipt.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
        stream.write("\n")
    temporary.replace(receipt)
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
