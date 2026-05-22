using System;
using System.Diagnostics;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tools.DiffusionPerfDiag;

/// <summary>
/// #1305 — pin down WHICH phase of compile-mode hangs on SD UNet ops.
/// Try compiling a single ResBlock-like graph (GroupNorm -> Swish -> Conv2D -> Add)
/// and time trace / compile / first execute / second execute.
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        AiDotNetEngine.Current = new CpuEngine();
        var log = new System.IO.StreamWriter("perf-1305-compile-phases.log") { AutoFlush = true };
        void Log(string s) { Console.WriteLine(s); log.WriteLine(s); log.Flush(); }
        Log($"=== Diffusion #1305 — compile-mode phase timing ===");
        Log($".NET {Environment.Version}");

        var engine = (CpuEngine)AiDotNetEngine.Current;
        var rng = new Random(0);

        // Build SD-style ResBlock inputs
        int batch = 1, channels = 320, h = 64, w = 64;
        int numGroups = 32;
        var input = new Tensor<double>(new[] { batch, channels, h, w });
        var gamma = new Tensor<double>(new[] { channels });
        var beta = new Tensor<double>(new[] { channels });
        var kernel = new Tensor<double>(new[] { channels, channels, 3, 3 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 0.1;
        for (int i = 0; i < channels; i++) { gamma[i] = 1.0; beta[i] = 0.0; }
        for (int i = 0; i < kernel.Length; i++) kernel[i] = rng.NextDouble() * 0.01;

        Log($"Inputs ready: [{batch},{channels},{h},{w}] doubles");

        // Phase A: eager baseline
        var swA = Stopwatch.StartNew();
        for (int it = 0; it < 3; it++)
        {
            var n = engine.GroupNorm<double>(input, numGroups, gamma, beta, 1e-5, out _, out _);
            engine.SwishInPlace<double>(n);
            var c = engine.Conv2D<double>(n, kernel, stride: new[] { 1, 1 }, padding: new[] { 1, 1 }, dilation: new[] { 1, 1 });
            var _o = engine.TensorAdd<double>(c, input);
        }
        swA.Stop();
        Log($"Phase A — eager ResBlock-like (3 iters): {swA.Elapsed.TotalMilliseconds / 3.0:F1} ms/iter");

        // Phase B: trace under GraphMode (no compile yet)
        Log("Phase B — trace under GraphMode...");
        Tensor<double> tracedOutput;
        var swB = Stopwatch.StartNew();
        GraphScope<double>? scope = null;
        try
        {
            scope = GraphMode.Enable<double>();
            var n = engine.GroupNorm<double>(input, numGroups, gamma, beta, 1e-5, out _, out _);
            engine.SwishInPlace<double>(n);
            var c = engine.Conv2D<double>(n, kernel, stride: new[] { 1, 1 }, padding: new[] { 1, 1 }, dilation: new[] { 1, 1 });
            tracedOutput = engine.TensorAdd<double>(c, input);
        }
        catch (Exception ex)
        {
            swB.Stop();
            Log($"Phase B FAILED at {swB.Elapsed.TotalMilliseconds:F1} ms: {ex.GetType().Name}: {ex.Message}");
            scope?.Dispose();
            return 1;
        }
        swB.Stop();
        Log($"Phase B — trace completed: {swB.Elapsed.TotalMilliseconds:F1} ms");

        // Phase C: compile the traced graph
        Log("Phase C — compile the traced graph...");
        ICompiledPlan<double> plan;
        var swC = Stopwatch.StartNew();
        try
        {
            plan = scope!.CompileInference<double>(tracedOutput, input._shape);
        }
        catch (Exception ex)
        {
            swC.Stop();
            Log($"Phase C FAILED at {swC.Elapsed.TotalMilliseconds:F1} ms: {ex.GetType().Name}: {ex.Message}");
            scope.Dispose();
            return 1;
        }
        swC.Stop();
        Log($"Phase C — compile completed: {swC.Elapsed.TotalMilliseconds:F1} ms");
        scope.Dispose();

        // Phase D: first plan.Execute()
        Log("Phase D — first plan.Execute() (warm dispatch path)...");
        var swD = Stopwatch.StartNew();
        try
        {
            var _o = plan.Execute();
        }
        catch (Exception ex)
        {
            swD.Stop();
            Log($"Phase D FAILED at {swD.Elapsed.TotalMilliseconds:F1} ms: {ex.GetType().Name}: {ex.Message}");
            return 1;
        }
        swD.Stop();
        Log($"Phase D — first Execute: {swD.Elapsed.TotalMilliseconds:F1} ms");

        // Phase E: 5 more plan.Execute() to see steady-state
        var swE = Stopwatch.StartNew();
        for (int it = 0; it < 5; it++) { var _o = plan.Execute(); }
        swE.Stop();
        Log($"Phase E — steady-state Execute: {swE.Elapsed.TotalMilliseconds / 5.0:F1} ms/call");

        plan.Dispose();
        Log("Done.");
        return 0;
    }
}
