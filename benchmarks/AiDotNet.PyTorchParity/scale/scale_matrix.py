"""The large-data plan, report validation and summaries for run_matrix.py --scale.

Each classifier trains for whole epochs over a real dataset and a 1,000,000-sample synthetic one (scale/prepare_data.py),
in its own fresh process per framework, launch and dataset, and scores the dataset's test split after every epoch.
Throughput is samples trained per second of training time over all epochs; both frameworks first run the same 20
untimed warm-up updates, so compilation and JIT stay out of it.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

VISION = ("resnet18", "resnet50", "vgg11", "mobilenetv2", "densenet121", "vit")
DATASETS = {
    **{m: ("emnist-digits", "synth-mnist") for m in ("mlp", "cnn")},
    **{m: ("emnist-seq32", "synth-seq32") for m in ("lstm", "gru", "transformer")},
    **{m: ("cinic10", "synth-rgb32") for m in VISION},
}
REAL_EPOCHS, SYNTHETIC_EPOCHS = 3, 1
# On the CPU the vision models train one epoch of the real dataset only: a CPU DenseNet-121 epoch over CINIC-10's
# 180,000 images takes about 3.4 hours for the two frameworks together, and the synthetic set is 5.6 times larger.
CPU_VISION_REAL_EPOCHS = 1


def default_plan(devices: list[str], models: list[str]) -> dict[str, dict[str, list[list]]]:
    plan: dict[str, dict[str, list[list]]] = {}
    for device in devices:
        plan[device] = {}
        for model in models:
            real, synthetic = DATASETS[model]
            if device == "cpu" and model in VISION:
                plan[device][model] = [[real, CPU_VISION_REAL_EPOCHS]]
            else:
                plan[device][model] = [[real, REAL_EPOCHS], [synthetic, SYNTHETIC_EPOCHS]]
    return plan


def fastest_configurations(summary_csv: Path) -> dict[str, dict[str, str]]:
    """The fastest PyTorch configuration for each model's training row in a run_matrix summary.csv."""
    chosen: dict[str, dict[str, str]] = {}
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["operation"] == "train":
                chosen.setdefault(row["device"], {})[row["model"]] = row["torch_configuration"]
    return chosen


def tag_name(side: str, launch: int, device: str, model: str, dataset: str) -> str:
    return f"{side}-{launch}-{device}-{model}-{dataset}"


def validate(path: Path, side: str, model: str, device: str, dataset_meta: dict, epochs: int) -> dict:
    """The scale block of one report, checked against the dataset and schedule. The caller checks the protocol."""
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    ai = side == "aidotnet"
    rows = obj["Results" if ai else "results"]
    if len(rows) != 1:
        raise ValueError("Unexpected number of models")
    scale = rows[0]["Scale" if ai else "scale"]
    if not scale:
        raise ValueError("The report has no large-data block")

    def get(key_ai: str, key_py: str, source: dict | None = None):
        source = scale if source is None else source
        return source[key_ai if ai else key_py]

    if (get("Dataset", "dataset") != dataset_meta["name"]
            or get("TrainSha256", "train_sha256") != dataset_meta["files"]["train-x"]["sha256"]
            or get("TrainSamples", "train_samples") != dataset_meta["train_samples"]
            or get("BatchSize", "batch_size") != 64):
        raise ValueError("Dataset or batch size mismatch")
    epoch_rows = get("Epochs", "epochs")
    if len(epoch_rows) != epochs:
        raise ValueError(f"{len(epoch_rows)} epochs, expected {epochs}")
    out = []
    for entry in epoch_rows:
        record = dict(epoch=get("Epoch", "epoch", entry), train_seconds=get("TrainSeconds", "train_seconds", entry),
                      samples_per_second=get("SamplesPerSecond", "samples_per_second", entry),
                      train_loss=get("TrainLoss", "train_loss", entry),
                      test_accuracy=get("TestAccuracy", "test_accuracy", entry),
                      eval_seconds=get("EvalSeconds", "eval_seconds", entry),
                      data_wait_seconds=get("DataWaitSeconds", "data_wait_seconds", entry),
                      cumulative_train_seconds=get("CumulativeTrainSeconds", "cumulative_train_seconds", entry))
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in record.values()):
            raise ValueError(f"Non-finite value in epoch {record['epoch']}")
        if not 0 <= record["test_accuracy"] <= 1 or record["train_seconds"] <= 0:
            raise ValueError(f"Invalid accuracy or time in epoch {record['epoch']}")
        out.append(record)
    resources = get("Resources", "resources")
    return dict(parameters=rows[0]["Parameters" if ai else "parameters"], steps=get("StepsPerEpoch", "steps_per_epoch"),
                epochs=out, rss_mb_peak=resources["ProcessRssMbPeak" if ai else "rss_mb_peak"],
                gpu_mem_mb_peak=resources["GpuMemMbPeak" if ai else "gpu_mem_mb_peak"])


COLUMNS = ("model", "device", "dataset", "launch", "framework", "configuration", "epoch", "train_seconds",
           "samples_per_second", "train_loss", "test_accuracy", "cumulative_train_seconds", "eval_seconds",
           "data_wait_seconds", "parameters", "rss_mb_peak", "gpu_mem_mb_peak", "flags")
# A side whose slowest launch's throughput is below its fastest's by more than this factor is flagged unstable: the
# same rule as run_matrix.py's UNSTABLE_SPREAD for the main matrix.
UNSTABLE_SPREAD = 1.5
# Models whose AiDotNet training is not bit-reproducible on either device: identical runs report different losses from
# the first epoch, on the CPU and on the GPU at a locked, verified clock (README). For them, launches that disagree say
# nothing about the hardware, so ai-launches-disagree is not raised.
NONDETERMINISTIC = frozenset({"gru", "vit"})


def summarize(rows: list[dict]) -> list[dict]:
    """Per model, device and dataset: each side's throughput over all epochs (median and range across launches), the
    final test accuracy, peak memory, and PyTorch's training time divided by AiDotNet's."""
    runs: dict[tuple, dict] = {}
    for r in rows:
        key = (r["model"], r["device"], r["dataset"], r["configuration"], r["launch"])
        run = runs.setdefault(key, dict(samples=0.0, seconds=0.0, accuracy=None, last_epoch=-1, rss=r["rss_mb_peak"],
                                        gpu=r["gpu_mem_mb_peak"], flags=set(), trajectory=set()))
        run["trajectory"].add((r["epoch"], r["train_loss"], r["test_accuracy"]))
        run["samples"] += r["samples_per_second"] * r["train_seconds"]
        run["seconds"] += r["train_seconds"]
        if r["epoch"] > run["last_epoch"]:
            run["last_epoch"], run["accuracy"] = r["epoch"], r["test_accuracy"]
        run["flags"].update(f for f in r["flags"].split(";") if f)
    groups: dict[tuple, dict[str, list]] = {}
    for (model, device, dataset, configuration, launch), run in sorted(runs.items(), key=lambda kv: kv[0]):
        groups.setdefault((model, device, dataset), {}).setdefault(configuration, []).append((launch, run))
    summary = []
    for (model, device, dataset), sides in groups.items():
        ai = dict(sides.get("aidotnet", []))
        torch_sides = {name: dict(v) for name, v in sides.items() if name != "aidotnet"}
        if not ai or not torch_sides:
            continue
        # The fastest PyTorch configuration by median throughput, when more than one ran.
        best = max(torch_sides, key=lambda name: statistics.median(r["samples"] / r["seconds"] for r in torch_sides[name].values()))
        pt = torch_sides[best]
        launches = sorted(set(ai) & set(pt))
        ratios = [ai[l]["samples"] / ai[l]["seconds"] / (pt[l]["samples"] / pt[l]["seconds"]) for l in launches]
        ai_rate = [ai[l]["samples"] / ai[l]["seconds"] for l in launches]
        pt_rate = [pt[l]["samples"] / pt[l]["seconds"] for l in launches]
        flags = set().union(*(ai[l]["flags"] | pt[l]["flags"] for l in launches))
        # AiDotNet's training outside NONDETERMINISTIC is deterministic on both devices: every launch trains on the same
        # data in the same order and must report the same loss and accuracy for every epoch. Launches that disagree
        # computed different values for identical work, which on a healthy machine does not happen (a GPU fault, see
        # the README).
        if model not in NONDETERMINISTIC and len({frozenset(ai[l]["trajectory"]) for l in launches}) > 1:
            flags.add("ai-launches-disagree")
        if max(ai_rate) / min(ai_rate) > UNSTABLE_SPREAD or max(pt_rate) / min(pt_rate) > UNSTABLE_SPREAD:
            flags.add("unstable")
        summary.append(dict(
            model=model, device=device, dataset=dataset, launches=len(launches),
            samples_per_launch=round(ai[launches[0]]["samples"]),
            ai_samples_per_second=round(statistics.median(ai_rate), 3),
            torch_configuration=best, torch_samples_per_second=round(statistics.median(pt_rate), 3),
            torch_time_over_ai=round(statistics.median(ratios), 4), ratio_min=round(min(ratios), 4),
            ratio_max=round(max(ratios), 4),
            ai_hours=round(statistics.median(ai[l]["seconds"] for l in launches) / 3600, 4),
            torch_hours=round(statistics.median(pt[l]["seconds"] for l in launches) / 3600, 4),
            ai_test_accuracy=round(statistics.median(ai[l]["accuracy"] for l in launches), 4),
            torch_test_accuracy=round(statistics.median(pt[l]["accuracy"] for l in launches), 4),
            ai_rss_mb_peak=round(statistics.median(ai[l]["rss"] for l in launches), 1),
            torch_rss_mb_peak=round(statistics.median(pt[l]["rss"] for l in launches), 1),
            flags=";".join(sorted(flags))))
    return summary


def markdown(summary: list[dict]) -> str:
    lines = ["| Model | Device | Dataset | Samples per launch | AiDotNet samples/s | PyTorch samples/s (configuration) | "
             "PyTorch time ÷ AiDotNet time, median (range) | Final test accuracy, AiDotNet / PyTorch | Flags |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for s in summary:
        lines.append(f"| {s['model']} | {s['device']} | {s['dataset']} | {s['samples_per_launch']:,} | "
                     f"{s['ai_samples_per_second']:,.0f} | {s['torch_samples_per_second']:,.0f} (`{s['torch_configuration']}`) | "
                     f"{s['torch_time_over_ai']:.2f} ({s['ratio_min']:.2f}–{s['ratio_max']:.2f}) | "
                     f"{s['ai_test_accuracy']:.2%} / {s['torch_test_accuracy']:.2%} | "
                     f"{', '.join(s['flags'].split(';')) if s['flags'] else ''} |")
    return "\n".join(lines) + "\n"


def write_outputs(dest: Path, rows: list[dict]) -> None:
    with (dest / "scale-measurements.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with (dest / "scale-summary.csv").open("w", newline="", encoding="utf-8") as out:
        if summary:
            writer = csv.DictWriter(out, fieldnames=list(summary[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary)
    (dest / "scale-summary.md").write_text(markdown(summary), encoding="utf-8", newline="\n")
