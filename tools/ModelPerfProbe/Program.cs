using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace AiDotNet.Tools.ModelPerfProbe;

/// <summary>
/// Generic per-model performance probe. Covers every <c>IFullModel&lt;float, Tensor,
/// Tensor&gt;</c> in the registry; emits per-step ms / MB / GC counts so we can
/// see exactly which models exceed the model-family scaffold's 120 s Training_*
/// budget and WHY (CPU-bound vs allocation-bound).
///
/// Usage:
///   ModelPerfProbe --list
///   ModelPerfProbe --profile LayoutXLM [--steps 3 --seq 16]
///   ModelPerfProbe --profile-all [--steps 3 --seq 16
///                                  --slow-step-ms 1000 --slow-alloc-mb 50
///                                  --per-model-budget-s 60
///                                  --output artifacts/perf-baseline.json
///                                  --diff-baseline artifacts/perf-baseline.json
///                                  --regression-step-ratio 1.5]
///   ModelPerfProbe --deep-profile LayoutXLM [--steps 3 --seq 16
///                                             --trace-dir artifacts/perf-traces]
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        var opts = ParseArgs(args);
        if (opts is null) return PrintUsage();

        if (opts.List)
        {
            var models = ModelRegistry.Discover();
            Console.WriteLine($"# {models.Count} probeable models:");
            foreach (var t in models)
                Console.WriteLine($"  {t.Name}");
            return 0;
        }

        if (!string.IsNullOrEmpty(opts.DeepProfileModel))
            return DeepProfile(opts);

        if (!string.IsNullOrEmpty(opts.Model))
        {
            var t = ModelRegistry.ResolveByName(opts.Model!);
            if (t is null)
            {
                Console.Error.WriteLine($"ERROR: no probeable model named {opts.Model!} (try --list).");
                return 2;
            }
            var result = ProbeRunner.Run(t, opts.Steps, opts.Seq, opts.SlowStepMs, opts.SlowAllocMb,
                TimeSpan.FromSeconds(opts.PerModelBudgetS));
            PrintHuman(result);
            if (opts.OutputJson is { } outPath)
                File.WriteAllText(outPath, SerializeOne(result));
            return result.Status == "ok" ? 0 : 1;
        }

        if (opts.ProfileAll)
            return ProfileAll(opts);

        return PrintUsage();
    }

    private static int ProfileAll(Options opts)
    {
        var models = ModelRegistry.Discover();
        Console.WriteLine($"# Probing {models.Count} models (steps={opts.Steps}, seq={opts.Seq}, per-model budget {opts.PerModelBudgetS:F0}s)");
        var results = new List<ProbeResult>(models.Count);
        int idx = 0;
        foreach (var t in models)
        {
            idx++;
            Console.WriteLine($"[{idx}/{models.Count}] {t.Name}");
            var r = ProbeRunner.Run(t, opts.Steps, opts.Seq, opts.SlowStepMs, opts.SlowAllocMb,
                TimeSpan.FromSeconds(opts.PerModelBudgetS));
            results.Add(r);
            PrintHumanCompact(r);
        }

        var flagged = results.Where(r => r.Flagged).OrderByDescending(r => r.AvgStepMs).ToList();
        Console.WriteLine();
        Console.WriteLine($"# Slow models (above --slow-step-ms {opts.SlowStepMs:F0} or --slow-alloc-mb {opts.SlowAllocMb:F0}): {flagged.Count}");
        foreach (var r in flagged)
            Console.WriteLine($"  {r.Model,-40}  step={r.AvgStepMs,8:F1} ms  alloc={r.AllocMbPerStep,7:F1} MB  {r.FlagReason}");

        if (opts.OutputJson is { } path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            File.WriteAllText(path, SerializeAll(results));
            Console.WriteLine($"# Wrote manifest: {path}");
        }

        if (opts.DiffBaseline is { } baselinePath)
            return ApplyBaselineDiff(results, baselinePath, opts.RegressionStepRatio);

        return 0;
    }

    /// <summary>
    /// Per-model deep profile: wraps a single probe in a child-process dotnet-trace
    /// + dotnet-gcdump collection so a slow finding becomes a hot-method report
    /// without manual tooling. The child runs <c>--profile</c> on the same model so
    /// the probe metrics and the trace cover the same workload.
    /// </summary>
    private static int DeepProfile(Options opts)
    {
        var t = ModelRegistry.ResolveByName(opts.DeepProfileModel!);
        if (t is null)
        {
            Console.Error.WriteLine($"ERROR: no probeable model named {opts.DeepProfileModel!} (try --list).");
            return 2;
        }

        var traceDir = Path.GetFullPath(opts.TraceDir ?? "artifacts/perf-traces");
        Directory.CreateDirectory(traceDir);
        var stem = $"{t.Name.Replace('`', '_')}-{DateTime.UtcNow:yyyyMMddTHHmmss}";
        var nettrace = Path.Combine(traceDir, stem + ".nettrace");
        var gcdump = Path.Combine(traceDir, stem + ".gcdump");

        var ownDll = Process.GetCurrentProcess().MainModule!.FileName!.Equals("dotnet.exe", StringComparison.OrdinalIgnoreCase)
            ? Environment.GetCommandLineArgs()[0]
            : typeof(Program).Assembly.Location;
        var childArgs = $"\"{ownDll}\" --profile {opts.DeepProfileModel} --steps {opts.Steps} --seq {opts.Seq} --per-model-budget-s {opts.PerModelBudgetS:F0}";

        Console.WriteLine($"# Deep-profile {t.Name}");
        Console.WriteLine($"#   trace : {nettrace}");
        Console.WriteLine($"#   gcdump: {gcdump}");

        var traceProc = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "dotnet-trace",
                Arguments = $"collect " +
                    $"--providers Microsoft-Windows-DotNETRuntime:0x1:5,Microsoft-DotNETCore-SampleProfiler:0xF00000000000:5 " +
                    $"--output \"{nettrace}\" " +
                    $"--duration 00:00:{Math.Max(10, opts.PerModelBudgetS * 2):F0} " +
                    $"-- dotnet {childArgs}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            },
        };
        traceProc.OutputDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine($"[trace] {e.Data}"); };
        traceProc.ErrorDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine($"[trace.err] {e.Data}"); };
        traceProc.Start();
        traceProc.BeginOutputReadLine();
        traceProc.BeginErrorReadLine();
        traceProc.WaitForExit();
        if (traceProc.ExitCode != 0)
            Console.WriteLine($"# dotnet-trace exit {traceProc.ExitCode} (continuing)");

        // gcdump uses pid attach, so we'd need to re-launch and attach mid-flight.
        // Simpler: skip gcdump in this iteration; nettrace already includes GC
        // events sufficient to identify the dominant allocators.
        Console.WriteLine($"# Done. Convert to speedscope with:");
        Console.WriteLine($"#   dotnet-trace convert \"{nettrace}\" --format speedscope");
        return 0;
    }

    private static int ApplyBaselineDiff(List<ProbeResult> current, string baselinePath, double regressionStepRatio)
    {
        if (!File.Exists(baselinePath))
        {
            Console.Error.WriteLine($"WARN: baseline {baselinePath} not found — skipping regression diff.");
            return 0;
        }

        var baseline = JsonSerializer.Deserialize<List<ProbeResult>>(File.ReadAllText(baselinePath))
            ?? new List<ProbeResult>();
        var baselineByName = baseline.ToDictionary(r => r.Model, r => r);

        int regressions = 0;
        foreach (var c in current)
        {
            if (c.Status != "ok") continue;
            if (!baselineByName.TryGetValue(c.Model, out var b) || b.Status != "ok") continue;
            if (b.AvgStepMs <= 0) continue;
            double ratio = c.AvgStepMs / b.AvgStepMs;
            if (ratio > regressionStepRatio)
            {
                Console.WriteLine(
                    $"REGRESSION  {c.Model,-40}  step {b.AvgStepMs,8:F1} → {c.AvgStepMs,8:F1} ms  ({ratio:F2}x, > {regressionStepRatio:F2}x threshold)");
                regressions++;
            }
        }

        if (regressions > 0)
        {
            Console.WriteLine($"# {regressions} step-time regressions ≥ {regressionStepRatio:F2}x baseline. Failing.");
            return 1;
        }
        Console.WriteLine($"# No regressions ≥ {regressionStepRatio:F2}x baseline.");
        return 0;
    }

    private sealed class Options
    {
        public bool List;
        public string? Model;
        public string? DeepProfileModel;
        public bool ProfileAll;
        public int Steps = 3;
        public int Seq = 16;
        public double SlowStepMs = 1000;
        public double SlowAllocMb = 50;
        public double PerModelBudgetS = 60;
        public string? OutputJson;
        public string? DiffBaseline;
        public double RegressionStepRatio = 1.5;
        public string? TraceDir;
    }

    private static Options? ParseArgs(string[] args)
    {
        var o = new Options();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--list": o.List = true; break;
                case "--profile": o.Model = args[++i]; break;
                case "--deep-profile": o.DeepProfileModel = args[++i]; break;
                case "--profile-all": o.ProfileAll = true; break;
                case "--steps": o.Steps = int.Parse(args[++i]); break;
                case "--seq": o.Seq = int.Parse(args[++i]); break;
                case "--slow-step-ms": o.SlowStepMs = double.Parse(args[++i]); break;
                case "--slow-alloc-mb": o.SlowAllocMb = double.Parse(args[++i]); break;
                case "--per-model-budget-s": o.PerModelBudgetS = double.Parse(args[++i]); break;
                case "--output": o.OutputJson = args[++i]; break;
                case "--diff-baseline": o.DiffBaseline = args[++i]; break;
                case "--regression-step-ratio": o.RegressionStepRatio = double.Parse(args[++i]); break;
                case "--trace-dir": o.TraceDir = args[++i]; break;
                default:
                    Console.Error.WriteLine($"Unknown arg: {args[i]}");
                    return null;
            }
        }
        return o;
    }

    private static int PrintUsage()
    {
        Console.WriteLine("ModelPerfProbe — measure per-model construct + train-step latency / allocation");
        Console.WriteLine();
        Console.WriteLine("Usage:");
        Console.WriteLine("  ModelPerfProbe --list");
        Console.WriteLine("  ModelPerfProbe --profile <ModelName> [--steps N --seq L --per-model-budget-s S --output path.json]");
        Console.WriteLine("  ModelPerfProbe --profile-all [--steps N --seq L --per-model-budget-s S");
        Console.WriteLine("                                --slow-step-ms 1000 --slow-alloc-mb 50");
        Console.WriteLine("                                --output artifacts/perf-baseline.json");
        Console.WriteLine("                                --diff-baseline artifacts/perf-baseline.json");
        Console.WriteLine("                                --regression-step-ratio 1.5]");
        Console.WriteLine("  ModelPerfProbe --deep-profile <ModelName> [--steps N --seq L --trace-dir dir]");
        return 1;
    }

    private static void PrintHuman(ProbeResult r)
    {
        Console.WriteLine($"# {r.Model}");
        if (r.Status != "ok")
        {
            Console.WriteLine($"  status: {r.Status}");
            if (r.Error != null) Console.WriteLine($"  error:  {r.Error}");
            // budget-truncated retains valid partial measurements; construct-failed
            // does not (StepCount only ever advances inside the measured loop).
            if (r.StepCount > 0 && r.AvgStepMs > 0)
            {
                Console.WriteLine($"  (partial-measurement) avg step : {r.AvgStepMs,8:F1} ms over {r.StepCount} steps");
                Console.WriteLine($"  (partial-measurement) alloc    : {r.AllocMbPerStep,8:F1} MB/step");
            }
            return;
        }
        Console.WriteLine($"  construct      : {r.ConstructMs,8:F1} ms");
        Console.WriteLine($"  warm-up fwd    : {r.WarmupForwardMs,8:F1} ms");
        Console.WriteLine($"  warm-up train  : {r.WarmupTrainMs,8:F1} ms");
        Console.WriteLine($"  avg step       : {r.AvgStepMs,8:F1} ms  ({r.StepCount} steps over {r.TotalMs,8:F1} ms)");
        Console.WriteLine($"  alloc / step   : {r.AllocMbPerStep,8:F1} MB  ({r.AllocBytes / (1024.0 * 1024.0):F1} MB total)");
        Console.WriteLine($"  GC counts      : gen0={r.Gen0} gen1={r.Gen1} gen2={r.Gen2}");
        Console.WriteLine($"  projected 30   : {r.Projected30IterS,8:F1} s  (xUnit timeout 120 s)");
        if (r.Flagged) Console.WriteLine($"  FLAGGED        : {r.FlagReason}");
    }

    private static void PrintHumanCompact(ProbeResult r)
    {
        if (r.Status == "ok")
            Console.WriteLine($"    step={r.AvgStepMs,8:F1} ms  alloc={r.AllocMbPerStep,7:F1} MB  gen0/1/2={r.Gen0}/{r.Gen1}/{r.Gen2}  proj30={r.Projected30IterS,5:F1} s{(r.Flagged ? "  FLAGGED" : "")}");
        else if (r.StepCount > 0)
            Console.WriteLine($"    {r.Status}: step={r.AvgStepMs,8:F1} ms/alloc={r.AllocMbPerStep,7:F1} MB over {r.StepCount} steps");
        else
            Console.WriteLine($"    SKIPPED: {r.Status} ({r.Error})");
    }

    private static string SerializeOne(ProbeResult r) => JsonSerializer.Serialize(r, JsonOpts);
    private static string SerializeAll(IReadOnlyList<ProbeResult> rs) => JsonSerializer.Serialize(rs, JsonOpts);

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
    };
}
