"""Run end-to-end AiDotNet/Python PyTorch training and inference pairs from this checkout."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pytorch"))
from compare import _index

CUDA_ENGINE = "Direct GPU Engine (CUDA "
MODELS = "mlp lstm transformer cnn resnet18 resnet50 vgg11 vit gru mobilenetv2 densenet121".split()
REFERENCE = ("mlp", "cnn", "lstm", "transformer")
BATCHES = [1, 8, 16]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def revision(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    return dict(commit=git("rev-parse", "HEAD"), changes=git("status", "--short"))


def run(cmd, stem, env):
    print(stem.name, flush=True)
    started = now()
    with stem.with_suffix(".log").open("w") as log:
        code = subprocess.run(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    save(stem.with_suffix(".receipt.json"), dict(command=cmd, started=started, finished=now(), exitCode=code))
    if code:
        raise RuntimeError(f"{stem.name} exited {code}. See {stem.with_suffix('.log')}")


class build_lock:
    """Hold an exclusive cross-process lock, released by the OS if this process dies."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = self.path.open("a+b")
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    self.file.seek(0)
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    pass  # LK_LOCK gives up after 10 s; keep waiting for the other build
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_EX)

    def __exit__(self, *exc):
        self.file.close()


def measurements(path, side, model, device, epochs, batches):
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    if len(rows) != 1 or obj["Framework" if ai else "framework"] != ("AiDotNet" if ai else "PyTorch"):
        raise ValueError("Unexpected framework or number of models")
    if ai:
        env = obj["Environment"]
        if (env["CpuThreads"], env["EngineDevice"]) != (8, {"cpu":"CpuEngine","cuda":"DirectGpuTensorEngine"}[device]):
            raise ValueError("AiDotNet engine/thread count mismatch")
        if device == "cuda" and not env["EngineName"].startswith(CUDA_ENGINE):
            raise ValueError(f"AiDotNet measured {env['EngineName']!r}, not the CUDA backend")
    elif obj["device"] != device or obj["torch_num_threads"] != 8 or (device == "cuda" and not obj["cuda_available"]):
        raise ValueError("PyTorch device/thread count mismatch")
    row = _index(obj)[model]
    if row.device != device or row.training is None or set(row.inference) != set(BATCHES):
        raise ValueError("Incomplete or mismatched benchmark row")
    raw = rows[0]
    training = raw["Training" if ai else "training"]
    losses = [training["FirstLoss" if ai else "first_loss"], training["LastLoss" if ai else "last_loss"]]
    if (not all(math.isfinite(v) for v in losses)
        or len(training["EpochSeconds" if ai else "epoch_seconds"]) != epochs
        or training["StepsPerEpoch" if ai else "steps_per_epoch"] != batches):
        raise ValueError("Training schedule or loss mismatch")
    times = [row.training, *(row.inference[b] for b in BATCHES)]
    if any(not math.isfinite(t.median) or t.median <= 0 for t in times):
        raise ValueError("Invalid timing")
    return raw["Parameters" if ai else "parameters"], times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", choices=["cuda","cpu"], default=["cuda","cpu"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--launches", type=int, help="Independent process pairs (default 2, smoke 1)")
    parser.add_argument("--smoke", action="store_true", help="Two epochs of two updates, three measured inference calls.")
    parser.add_argument("--tensors", type=Path, help="Build against this Tensors checkout instead of the repository's NuGet dependency.")
    parser.add_argument("--output", type=Path, default=HERE / "results", help="Parent for unique run directories")
    args = parser.parse_args()
    launches = args.launches if args.launches is not None else (1 if args.smoke else 2)
    if launches < 1 or len(set(args.devices)) != len(args.devices) or len(set(args.models)) != len(args.models):
        parser.error("Select distinct devices/models and a positive launch count.")
    import torch
    import torchvision
    import torch._dynamo.config as dynamo
    if dynamo.disable or dynamo.suppress_errors or os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        parser.error("Enable torch.compile and disable silent compilation-error fallback.")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select --devices cpu or install CUDA-enabled PyTorch.")
    if os.name == "nt" and not shutil.which("cl"):
        parser.error("Use Developer PowerShell for Visual Studio with the x64 C++ build tools.")
    args.output.mkdir(exist_ok=True, parents=True)
    dest = Path(tempfile.mkdtemp(dir=args.output.resolve(), prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")))
    print(f"Results: {dest}", flush=True)
    env = os.environ.copy()
    # A CUDA row must not fall back to OpenCL/HIP when CUDA fails to initialise. CPU launches never read this.
    env["AIDOTNET_DIRECTGPU_BACKENDS"] = "cuda"
    for key in ("AIDOTNET_CPU_THREADS", "AIDOTNET_BLAS_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = "8"
    epochs, batches, iters, warm = (2, 2, 3, 1) if args.smoke else (3, 20, 100, 10)
    common = ["--epochs", str(epochs), "--train-batches", str(batches), "--batch-size", "64", "--inference-batches", "1,8,16",
              "--inference-iterations", str(iters), "--warmup-iterations", str(warm), "--warmup-min-ms", "1000", "--seed", "1234"]
    refs = [m for m in REFERENCE if m in args.models]
    info = dict(started=now(), models=args.models, devices=args.devices, launches=launches, smoke=args.smoke, common=common,
                threads=8, referenceModels=refs, ai=revision(HERE.parents[1]), torch=torch.__version__, torchvision=torchvision.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                os=platform.platform(), cpu=platform.processor(), python=sys.version, sdk=subprocess.check_output(["dotnet", "--version"],text=True).strip())
    if shutil.which("nvidia-smi"):
        info["driver"] = subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv"],text=True,capture_output=True).stdout.strip()
    build = ["dotnet","build",str(HERE / "AiDotNet.PyTorchParity.csproj"),"-c","Release","-p:GeneratePackageOnBuild=false","-p:UseSharedCompilation=false","-m:1"]
    if args.tensors:
        tensors = args.tensors.resolve()
        info["tensors"] = revision(tensors)
        build += ["-p:UseLocalTensors=true",f"-p:LocalTensorsRoot={tensors}"]
    else:
        build += ["-p:UseLocalTensors=false"]
    save(dest / "protocol.json", info)
    def digest(p):
        with p.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    # The build output directory is shared by every invocation, so build, check and copy it under a lock,
    # then hash and launch only this invocation's private copy.
    dll = dest / "bin/AiDotNet.PyTorchParity.dll"
    with build_lock(HERE / ".run_matrix.lock"):
        run(build, dest / "build", env)
        shutil.copytree(HERE / "bin/Release/net10.0", dll.parent)
        if args.tensors and digest(dll.with_name("AiDotNet.Tensors.dll")) != digest(tensors / "src/AiDotNet.Tensors/bin/Release/net10.0/AiDotNet.Tensors.dll"):
            raise RuntimeError("Loaded Tensors assembly differs from the requested checkout")
    hashes = {p: digest(dll.with_name(p)) for p in (dll.name, "AiDotNet.dll", "AiDotNet.Tensors.dll")}
    save(dest / "assemblies.json", hashes)
    if refs:
        ref = dest / "reference"
        run([sys.executable,str(HERE / "pytorch/export_reference.py"),"--models",",".join(refs),"--out",str(ref)],dest/"reference-export",env)
        for device in args.devices:
            stem = dest / f"reference-{device}"
            run(["dotnet",str(dll),"--verify-step1",str(ref),"--models",",".join(refs),"--device",device],stem,env)
            lines = stem.with_suffix(".log").read_text(encoding="utf-8", errors="replace").splitlines()
            if not all(any(line.startswith(f"[step1] {m}: PASS") for line in lines) for m in refs):
                raise RuntimeError(f"Missing first-update verdict. See {stem.with_suffix('.log')}")
    problems = []
    with (dest / "measurements.csv").open("w", newline="", encoding="utf-8") as out:
        table = csv.writer(out)
        table.writerow(["model","device","launch","operation","batch","ai_parameters","torch_parameters","ai_median_ms","torch_median_ms","torch_over_ai"])
        for launch in range(1, launches + 1):
            for device in args.devices:
                for model in args.models:
                    results = {}
                    for side in (("aidotnet","torch") if launch % 2 else ("torch","aidotnet")):
                        cmd = (["dotnet",str(dll)] if side == "aidotnet" else
                               [sys.executable,str(HERE/"pytorch/benchmark.py"),"--compile","--threads","8"])
                        tag = dest / f"{side}-{launch}-{device}-{model}"
                        report = tag.with_suffix(".json")
                        try:
                            run([*cmd,"--models",model,"--device",device,*common,"--output",str(report)],tag,env)
                            results[side] = measurements(report,side,model,device,epochs,batches)
                        except (KeyError, ValueError, OSError, RuntimeError, TypeError) as error:
                            problems.append(f"{tag.name}: {error}")
                    if len(results) == 2:
                        a, t = results["aidotnet"], results["torch"]
                        for i, batch in enumerate([64,*BATCHES]):
                            ours, theirs = a[1][i].median, t[1][i].median
                            table.writerow([model,device,launch,"train" if i==0 else "inference",batch,a[0],t[0],ours,theirs,theirs/ours])
                        out.flush()
    save(dest / "completion.json", dict(finished=now(),failures=problems))
    if problems:
        sys.exit("\n".join(problems))
    print(dest / "measurements.csv")


if __name__ == "__main__":
    main()
