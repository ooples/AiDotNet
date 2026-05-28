using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Diffusion.FastGeneration;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.ConsistencyModelPerfDiag;

/// <summary>
/// #1305 ConsistencyModel.ScaledInput_ShouldChangeOutput bottleneck instrumentation.
///
/// Mirrors what the failing test does:
///   var input = CreateRandomTensor([1, 4, 64, 64], rng);
///   var scaledInput = input * 10.0;
///   var output1 = model.Predict(input);
///   var output2 = model.Predict(scaledInput);
///
/// Reports:
///   - Model construction wall time
///   - First Predict wall time (cold — includes any lazy init / compile)
///   - Second Predict wall time (warm — steady-state)
///   - Direct UNet PredictNoise wall time (isolates UNet cost from diffusion-loop overhead)
///   - DefaultInferenceSteps value actually used
///
/// Output goes to both console and perf-1305-consistencymodel.log.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        // Fall back to console-only when the working directory isn't writable
        // (CI sandboxes, read-only mounts) instead of crashing before any
        // diagnostics run — the whole point of this tool is to capture data.
        System.IO.StreamWriter? log = null;
        try
        {
            log = new System.IO.StreamWriter("perf-1305-consistencymodel.log") { AutoFlush = true };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[warn] perf log file disabled — could not open 'perf-1305-consistencymodel.log': {ex.Message}");
        }
        using var _logScope = log; // dispose if non-null
        void Log(string s)
        {
            Console.WriteLine(s);
            if (log is not null) { log.WriteLine(s); log.Flush(); }
        }

        Log($"=== ConsistencyModel #1305 perf bottleneck — {DateTime.UtcNow:O} ===");
        Log($".NET {Environment.Version}, Processors: {Environment.ProcessorCount}");

        // Step 0: construct the model exactly as the test does
        var swCtor = Stopwatch.StartNew();
        var model = new ConsistencyModel<double>(seed: 42);
        swCtor.Stop();
        Log($"Construction:           {swCtor.Elapsed.TotalSeconds,8:F3} s   (paramCount={model.ParameterCount:N0})");

        // Step 1: build the test inputs
        var rng = new Random(42);
        int[] shape = new[] { 1, 4, 64, 64 };
        int length = 1 * 4 * 64 * 64;
        var input = new Tensor<double>(shape);
        var scaled = new Tensor<double>(shape);
        for (int i = 0; i < length; i++)
        {
            input[i] = rng.NextDouble();
            scaled[i] = input[i] * 10.0;
        }
        Log($"Input shape:            [{string.Join(",", shape)}]   length={length}");

        // Step 2: first Predict (cold) — mirrors test's `model.Predict(input)`
        var swPredict1 = Stopwatch.StartNew();
        var output1 = model.Predict(input);
        swPredict1.Stop();
        Log($"First Predict (cold):   {swPredict1.Elapsed.TotalSeconds,8:F3} s   (output len={output1.Length}, shape=[{string.Join(",", output1.Shape)}])");

        // Step 3: second Predict (warm) — mirrors test's `model.Predict(scaledInput)`
        var swPredict2 = Stopwatch.StartNew();
        var output2 = model.Predict(scaled);
        swPredict2.Stop();
        Log($"Second Predict (warm):  {swPredict2.Elapsed.TotalSeconds,8:F3} s   (output len={output2.Length})");

        // Total test cost = both Predicts. Test budget is 120s.
        var totalTest = swPredict1.Elapsed.TotalSeconds + swPredict2.Elapsed.TotalSeconds;
        Log($"--");
        Log($"Total Predict cost:     {totalTest,8:F3} s   (test budget = 120 s)");
        Log($"--");

        // Step 4: isolate UNet cost — call PredictNoise directly on the noise predictor
        // at the same shape, bypassing the diffusion sampling loop. Run 3 warm-up calls
        // first so any compile/init is done, then measure 3 steady-state calls.
        var unet = model.NoisePredictor;
        Log($"UNet:                   {unet.GetType().Name}");
        Log($"UNet input channels:    {unet.InputChannels}");

        // Warm-up
        var warm1 = unet.PredictNoise(input, timestep: 0, conditioning: null);
        var warm2 = unet.PredictNoise(input, timestep: 1, conditioning: null);
        var warm3 = unet.PredictNoise(input, timestep: 2, conditioning: null);
        Log($"UNet warmup output len: {warm1.Length}   (expected {length})");

        const int n = 5;
        var samples = new double[n];
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            var _ = unet.PredictNoise(input, timestep: 10 + i, conditioning: null);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalSeconds;
        }
        double minUnet = double.MaxValue, maxUnet = double.MinValue, sumUnet = 0;
        for (int i = 0; i < n; i++)
        {
            if (samples[i] < minUnet) minUnet = samples[i];
            if (samples[i] > maxUnet) maxUnet = samples[i];
            sumUnet += samples[i];
        }
        double meanUnet = sumUnet / n;

        Log($"UNet PredictNoise (n={n}, warm):");
        Log($"  min:                  {minUnet,8:F3} s");
        Log($"  mean:                 {meanUnet,8:F3} s");
        Log($"  max:                  {maxUnet,8:F3} s");
        Log($"  samples:              [{string.Join(", ", Array.ConvertAll(samples, x => x.ToString("F3")))}]");

        // === Phase 6: sub-phase breakdown via ForwardProfilingSink ===
        Log("--");
        Log("Sub-phase breakdown of one warm PredictNoise:");
        // Type-guard: the sink is a static field on UNetNoisePredictor<double>; if a
        // different INoisePredictor implementation is wired in (e.g. a future swap),
        // the sink would never fill and the breakdown would silently report empty —
        // fail fast instead so the diagnostic never lies about what it measured.
        if (unet is not UNetNoisePredictor<double>)
        {
            Log($"  (skipped — predictor is {unet.GetType().FullName}, not UNetNoisePredictor<double>; sub-phase sink unavailable for this implementation)");
            return 1;
        }
        var sink = new ConcurrentQueue<(string section, double ms)>();
        UNetNoisePredictor<double>.ForwardProfilingSink = sink;
        try
        {
            var _ = unet.PredictNoise(input, timestep: 20, conditioning: null);
        }
        finally
        {
            UNetNoisePredictor<double>.ForwardProfilingSink = null;
        }

        var entries = sink.ToArray();
        double totalSubMs = entries.Sum(e => e.ms);
        // Aggregate by section kind: input_conv, output_conv, enc.*, mid.*, dec.*
        var grouped = entries
            .GroupBy(e =>
            {
                var s = e.section;
                if (s.StartsWith("enc[")) return "encoder " + s.Substring(s.IndexOf(']') + 2);
                if (s.StartsWith("mid[")) return "middle  " + s.Substring(s.IndexOf(']') + 2);
                if (s.StartsWith("dec[")) return "decoder " + s.Substring(s.IndexOf(']') + 2);
                return s;
            })
            .Select(g => new { Key = g.Key, TotalMs = g.Sum(e => e.ms), Count = g.Count() })
            .OrderByDescending(g => g.TotalMs)
            .ToArray();

        Log($"  total sub-phase time:  {totalSubMs / 1000.0,8:F3} s  ({entries.Length} entries)");
        Log($"  by kind (desc, top 20):");
        foreach (var g in grouped.Take(20))
        {
            double share = totalSubMs > 0 ? 100.0 * g.TotalMs / totalSubMs : 0;
            Log($"    {g.Key,-28} {g.TotalMs,10:F1} ms  ({share,5:F1}%)  ×{g.Count}");
        }

        // Per-individual-section dump for the dominant ones
        Log($"  per-section detail (top 15 individual sections):");
        var perSection = entries
            .GroupBy(e => e.section)
            .Select(g => new { Section = g.Key, TotalMs = g.Sum(e => e.ms) })
            .OrderByDescending(g => g.TotalMs)
            .Take(15)
            .ToArray();
        foreach (var p in perSection)
        {
            double share = totalSubMs > 0 ? 100.0 * p.TotalMs / totalSubMs : 0;
            Log($"    {p.Section,-28} {p.TotalMs,10:F1} ms  ({share,5:F1}%)");
        }

        // Step 5: report decomposition. Test does 2 Predicts. Each Predict runs
        // numSteps PredictNoise calls (numSteps = DefaultInferenceSteps from the
        // diffusion options). Loop overhead = total - numSteps × UNet cost.
        var opts = model.GetOptions();
        Log($"--");
        Log($"Diffusion options:      {opts.GetType().Name}");
        // GetOptions() returns ModelOptions base; we want DefaultInferenceSteps.
        // Use reflection in a way that won't compile-break if the property is renamed.
        var typed = opts.GetType().GetProperty("DefaultInferenceSteps");
        int? steps = typed?.GetValue(opts) as int?;
        Log($"DefaultInferenceSteps:  {steps?.ToString() ?? "(unknown)"}");

        if (steps is int s2)
        {
            double impliedUnetPerPredict = s2 * meanUnet;
            double impliedTotalUnet = 2 * impliedUnetPerPredict;
            double impliedOverhead = totalTest - impliedTotalUnet;
            Log($"--");
            Log($"Decomposition (assuming {s2} steps/Predict × 2 Predicts = {2 * s2} UNet forwards):");
            Log($"  UNet share (mean):    {impliedTotalUnet,8:F3} s   ({100.0 * impliedTotalUnet / totalTest:F1}%)");
            Log($"  Loop overhead:        {impliedOverhead,8:F3} s   ({100.0 * impliedOverhead / totalTest:F1}%)");
            Log($"  Per-step UNet:        {meanUnet,8:F3} s");
            Log($"  Need per-step:        {(120.0 - 1.0) / (2 * s2),8:F3} s   (to fit 120s budget with 1s headroom)");
            double speedupNeeded = meanUnet / ((120.0 - 1.0) / (2 * s2));
            Log($"  UNet speedup needed:  {speedupNeeded,8:F2}×");
        }

        return 0;
    }
}
