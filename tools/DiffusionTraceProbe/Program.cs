using System;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.DiffusionTraceProbe;

/// <summary>
/// Replays the EXACT workload of the timeout-prone diffusion model-family tests
/// (<c>Training_ShouldReducePredictionError</c> / <c>ForwardPass_ShouldBeFinite_AfterTraining</c>)
/// for a single named model, so a sampling profiler (dotnet-trace) attached to
/// this process produces a clean, single-model hot-path report — without the
/// xunit/testhost noise and without the 16 GB-runner cross-model contention.
///
/// Construction and the [1,4,64,64] latent input mirror DiffusionModelTestBase&lt;float&gt;.
///
/// Usage:
///   DiffusionTraceProbe &lt;ModelName&gt; [--train-iters N] [--predict 0|1]
///   e.g.  DiffusionTraceProbe KandinskyModel --train-iters 5 --predict 1
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: DiffusionTraceProbe <ModelName> [--train-iters N] [--predict 0|1]");
            return 2;
        }

        string modelName = args[0];
        int trainIters = ArgInt(args, "--train-iters", 5);
        bool doPredict = ArgInt(args, "--predict", 1) != 0;

        // Pin the CPU engine + disable GPU so the profile reflects the CPU path the
        // CI shard runs (AIDOTNET_DISABLE_GPU=1 is also set by the test harness).
        AiDotNetEngine.Current = new CpuEngine();

        var model = Construct(modelName);
        if (model is null) return 1;

        try
        {
            var diff = (IDiffusionModel<float>)model;
            var full = (IFullModel<float, Tensor<float>, Tensor<float>>)model;

            var rng = new Random(42);
            var input = RandomTensor(new[] { 1, 4, 64, 64 }, rng);

            Console.WriteLine($"=== DiffusionTraceProbe: {modelName} ===");
            Console.WriteLine($".NET {Environment.Version}  ParameterCount={SafeParamCount(full)}");

            // Phase 1: warm-up noise-prediction probe (the test's errBefore measurement).
            int probeT = Math.Max(1, diff.Scheduler.TrainTimesteps / 2);
            var noisy = RandomTensor(new[] { 1, 4, 64, 64 }, rng);
            var sw = Stopwatch.StartNew();
            _ = diff.PredictNoise(noisy, probeT);
            sw.Stop();
            Console.WriteLine($"[PredictNoise warm-up]      {sw.Elapsed.TotalSeconds,8:F2} s");

            // Phase 2: training loop (forward + backward) — the timeout test runs 5-10 iters.
            sw.Restart();
            for (int i = 0; i < trainIters; i++)
            {
                full.Train(input, input);
                Console.WriteLine($"  train iter {i + 1}/{trainIters}   {sw.Elapsed.TotalSeconds,8:F2} s cumulative");
            }
            sw.Stop();
            Console.WriteLine($"[Train x{trainIters}]               {sw.Elapsed.TotalSeconds,8:F2} s total");

            // Phase 3: full sampling Predict (DDIM loop x SD-UNet forward) — the heaviest call.
            if (doPredict)
            {
                sw.Restart();
                var output = full.Predict(input);
                sw.Stop();
                Console.WriteLine($"[Predict (sampling loop)]   {sw.Elapsed.TotalSeconds,8:F2} s  (out len {output.Length})");
            }

            Console.WriteLine("=== done ===");
            return 0;
        }
        finally
        {
            if (model is IDisposable d) d.Dispose();
        }
    }

    private static object? Construct(string modelName)
    {
        var asm = typeof(AiDotNet.NeuralNetworks.NeuralNetworkBase<>).Assembly;
        var open = asm.GetTypes().FirstOrDefault(t =>
            t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
            && t.GetGenericArguments().Length == 1
            && string.Equals(t.Name, modelName + "`1", StringComparison.Ordinal));

        if (open is null)
        {
            // Allow passing the raw CLR name "Foo`1" too.
            open = asm.GetTypes().FirstOrDefault(t =>
                t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                && string.Equals(t.Name, modelName, StringComparison.Ordinal));
        }
        if (open is null)
        {
            Console.Error.WriteLine($"ERROR: no generic model type named '{modelName}' in {asm.GetName().Name}.");
            return null;
        }

        var closed = open.MakeGenericType(typeof(float));

        // Validate that the closed type implements IDiffusionModel<float> before attempting to cast it.
        var diffusionModelInterface = typeof(IDiffusionModel<float>);
        if (!diffusionModelInterface.IsAssignableFrom(closed))
        {
            Console.Error.WriteLine($"ERROR: {closed.Name} does not implement IDiffusionModel<float>.");
            return null;
        }

        // Pick the ctor with the most parameters that are ALL optional (matches the
        // (arch?, options?, scheduler?, predictor?, vae?, conditioner?, seed?) shape);
        // fill each with its compile-time default, but pin seed=42 like the tests.
        var ctor = closed.GetConstructors()
            .Where(c => c.GetParameters().All(p => p.HasDefaultValue))
            .OrderByDescending(c => c.GetParameters().Length)
            .FirstOrDefault();

        if (ctor is null)
        {
            Console.Error.WriteLine($"ERROR: {closed.Name} has no all-optional ctor to probe.");
            return null;
        }

        var pars = ctor.GetParameters();
        var ctorArgs = new object?[pars.Length];
        for (int i = 0; i < pars.Length; i++)
            ctorArgs[i] = string.Equals(pars[i].Name, "seed", StringComparison.OrdinalIgnoreCase)
                ? 42
                : pars[i].DefaultValue;

        try
        {
            var swCtor = Stopwatch.StartNew();
            var model = ctor.Invoke(ctorArgs);
            swCtor.Stop();
            Console.WriteLine($"[construct]                 {swCtor.Elapsed.TotalSeconds,8:F2} s");
            return model;
        }
        catch (TargetInvocationException tie)
        {
            var inner = tie.InnerException ?? tie;
            Console.Error.WriteLine($"ERROR constructing {closed.Name}: {inner.GetType().Name}: {inner.Message}");
            return null;
        }
    }

    private static long SafeParamCount(IFullModel<float, Tensor<float>, Tensor<float>> model)
    {
        try { return ((IDiffusionModel<float>)model).ParameterCount; }
        catch { return -1; }
    }

    private static Tensor<float> RandomTensor(int[] shape, Random rng)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)rng.NextDouble();
        return t;
    }

    private static int ArgInt(string[] args, string flag, int fallback)
    {
        int idx = Array.IndexOf(args, flag);
        if (idx >= 0 && idx + 1 < args.Length && int.TryParse(args[idx + 1], out int v)) return v;
        return fallback;
    }
}
