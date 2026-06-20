using System;
using System.Diagnostics;
using System.Reflection;
using AiDotNet.Diffusion.Control;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Tensors;

namespace AiDotNetTestConsole;

// Standalone profiling harness for ControlNetFluxModel<float> at the EXACT
// configuration the failing CI shard uses (ModelFamilyTests.Diffusion.
// ControlNetFluxModelTests: new ControlNetFluxModel<float>(seed: 42),
// input [1,16,32,32]). Reports per-phase wall time, managed-heap delta,
// total-allocated-bytes delta (cumulative churn), and per-gen GC counts so
// the OOM/perf hot spots are attributable to a phase without an external
// profiler. Also runnable under dotnet-trace / dotnet-gcdump for stack-level
// detail:
//   dotnet-trace collect --format Speedscope -- dotnet run --project testconsole -c Release -- controlnetflux-profile
//   dotnet-gcdump collect -p <pid>     (while paused at a phase barrier)
internal static class ControlNetFluxProfile
{
    private sealed class Phase : IDisposable
    {
        private readonly string _name;
        private readonly Stopwatch _sw;
        private readonly long _allocStart;
        private readonly long _heapStart;
        private readonly int _g0, _g1, _g2;

        public Phase(string name)
        {
            _name = name;
            // Settle the heap so the delta reflects THIS phase, not carry-over.
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            _allocStart = GC.GetTotalAllocatedBytes(precise: true);
            _heapStart = GC.GetTotalMemory(forceFullCollection: false);
            _g0 = GC.CollectionCount(0); _g1 = GC.CollectionCount(1); _g2 = GC.CollectionCount(2);
            _sw = Stopwatch.StartNew();
        }

        public void Dispose()
        {
            _sw.Stop();
            long allocEnd = GC.GetTotalAllocatedBytes(precise: true);
            long heapEnd = GC.GetTotalMemory(forceFullCollection: false);
            double gib(long b) => b / (1024.0 * 1024.0 * 1024.0);
            Console.WriteLine(
                $"  [{_name,-26}] {_sw.Elapsed.TotalSeconds,8:F3}s  " +
                $"alloc-churn={gib(allocEnd - _allocStart),7:F3} GiB  " +
                $"heap-delta={gib(heapEnd - _heapStart),7:F3} GiB  " +
                $"GC(g0/g1/g2)={GC.CollectionCount(0) - _g0}/{GC.CollectionCount(1) - _g1}/{GC.CollectionCount(2) - _g2}");
        }
    }

    // Disables weight-streaming auto-engagement by raising the internal
    // NoisePredictorBase<float>.StreamingThresholdOverride above any real model
    // size. Reflection because the knob is internal to AiDotNet; this is a
    // diagnostic harness, not production code.
    private static void DisableWeightStreaming()
    {
        var prop = typeof(NoisePredictorBase<float>).GetProperty(
            "StreamingThresholdOverride", BindingFlags.NonPublic | BindingFlags.Static);
        if (prop is null) { Console.WriteLine("  (could not find StreamingThresholdOverride)"); return; }
        prop.SetValue(null, long.MaxValue);
        Console.WriteLine("  weight streaming DISABLED (threshold = long.MaxValue)");
    }

    public static void RunNoStream()
    {
        Console.WriteLine("=== ControlNetFlux profile — STREAMING DISABLED (inference isolation) ===");
        DisableWeightStreaming();
        Run();
    }

    public static void Run()
    {
        Console.WriteLine("=== ControlNetFlux profile (ModelFamilyTests config: <float>, seed 42, input [1,16,32,32]) ===");
        Console.WriteLine($"ProcessorCount={Environment.ProcessorCount}  64bit={Environment.Is64BitProcess}");

        ControlNetFluxModel<float> model;
        using (new Phase("construct"))
        {
            model = new ControlNetFluxModel<float>(seed: 42);
        }

        Console.WriteLine($"  ParameterCount(total) = {model.ParameterCount:N0}  " +
                          $"(~{model.ParameterCount * 4.0 / (1024 * 1024 * 1024):F2} GiB as fp32 weights)");

        var input = new Tensor<float>(new[] { 1, 16, 32, 32 });
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (float)rng.NextDouble();

        // First Predict triggers lazy weight materialization + the 10-step
        // denoising loop. Second isolates steady-state inference cost.
        using (new Phase("predict #1 (lazy+10step)")) { _ = model.Predict(input); }
        using (new Phase("predict #2 (steady 10step)")) { _ = model.Predict(input); }

        // One training step: builds the GradientTape (retains all 57-block
        // activations) + optimizer state. Gated behind an env flag because it's
        // a separate concern from inference and can OOM on its own.
        if (Environment.GetEnvironmentVariable("AIDOTNET_PROFILE_TRAIN") == "1")
        {
            using (new Phase("train step #1 (tape+opt)")) { model.Train(input, input); }
            using (new Phase("train step #2 (steady)")) { model.Train(input, input); }
        }

        Console.WriteLine("=== done ===");
        GC.KeepAlive(model);
    }
}
