using System;
using System.Collections.Generic;
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
///                                  --output artifacts/perf-baseline.json]
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

        if (!string.IsNullOrEmpty(opts.Model))
        {
            var t = ModelRegistry.ResolveByName(opts.Model!);
            if (t is null)
            {
                Console.Error.WriteLine($"ERROR: no probeable model named {opts.Model!} (try --list).");
                return 2;
            }
            var result = ProbeRunner.Run(t, opts.Steps, opts.Seq, opts.SlowStepMs, opts.SlowAllocMb);
            PrintHuman(result);
            if (opts.OutputJson is { } outPath)
                File.WriteAllText(outPath, SerializeOne(result));
            return result.Status == "ok" ? 0 : 1;
        }

        if (opts.ProfileAll)
        {
            var models = ModelRegistry.Discover();
            Console.WriteLine($"# Probing {models.Count} models (steps={opts.Steps}, seq={opts.Seq})");
            var results = new List<ProbeResult>(models.Count);
            int idx = 0;
            foreach (var t in models)
            {
                idx++;
                Console.WriteLine($"[{idx}/{models.Count}] {t.Name}");
                var r = ProbeRunner.Run(t, opts.Steps, opts.Seq, opts.SlowStepMs, opts.SlowAllocMb);
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
            return 0;
        }

        return PrintUsage();
    }

    private sealed class Options
    {
        public bool List;
        public string? Model;
        public bool ProfileAll;
        public int Steps = 3;
        public int Seq = 16;
        public double SlowStepMs = 1000;
        public double SlowAllocMb = 50;
        public string? OutputJson;
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
                case "--profile-all": o.ProfileAll = true; break;
                case "--steps": o.Steps = int.Parse(args[++i]); break;
                case "--seq": o.Seq = int.Parse(args[++i]); break;
                case "--slow-step-ms": o.SlowStepMs = double.Parse(args[++i]); break;
                case "--slow-alloc-mb": o.SlowAllocMb = double.Parse(args[++i]); break;
                case "--output": o.OutputJson = args[++i]; break;
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
        Console.WriteLine("  ModelPerfProbe --profile <ModelName> [--steps N --seq L --output path.json]");
        Console.WriteLine("  ModelPerfProbe --profile-all [--steps N --seq L");
        Console.WriteLine("                                --slow-step-ms 1000 --slow-alloc-mb 50");
        Console.WriteLine("                                --output artifacts/perf-baseline.json]");
        return 1;
    }

    private static void PrintHuman(ProbeResult r)
    {
        Console.WriteLine($"# {r.Model}");
        if (r.Status != "ok")
        {
            Console.WriteLine($"  status: {r.Status}");
            if (r.Error != null) Console.WriteLine($"  error:  {r.Error}");
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
