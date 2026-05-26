using System;
using System.Diagnostics;
using AiDotNet.Diffusion.FastGeneration;
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
        using var log = new System.IO.StreamWriter("perf-1305-consistencymodel.log") { AutoFlush = true };
        void Log(string s) { Console.WriteLine(s); log.WriteLine(s); log.Flush(); }

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
