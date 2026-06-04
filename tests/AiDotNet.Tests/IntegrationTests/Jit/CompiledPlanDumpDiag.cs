// TEMPORARY diagnostic — dumps the compiled plan's recorded steps for the CNN and
// MLP so the missing/wrong op in the CNN trace is visible. Deleted once the
// parity defect is root-caused.

using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Jit;

public class CompiledPlanDumpDiag
{
    private readonly ITestOutputHelper _out;
    public CompiledPlanDumpDiag(ITestOutputHelper o) => _out = o;

    [Fact]
    public void DumpPlans()
    {
        AiDotNetEngine.ResetToCpu();
        var orig = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });

            var cnn = CompiledInferenceParityTestsAccessor.BuildCnn();
            var cin = CompiledInferenceParityTestsAccessor.MakeInput(new[] { 2, 1, 28, 28 }, 7);
            cnn.CompileForward(cin);
            cnn.PredictCompiled(cin); // populate hot plan
            Dump("CNN", cnn);

            var mlp = CompiledInferenceParityTestsAccessor.BuildMlp();
            var min = CompiledInferenceParityTestsAccessor.MakeInput(new[] { 2, 784 }, 7);
            mlp.CompileForward(min);
            mlp.PredictCompiled(min);
            Dump("MLP", mlp);

            // Per-step replay timing at the pathological shape [128,784].
            var mlp2 = CompiledInferenceParityTestsAccessor.BuildMlp();
            var big = CompiledInferenceParityTestsAccessor.MakeInput(new[] { 128, 784 }, 9);
            mlp2.CompileForward(big);
            mlp2.PredictCompiled(big);
            TimeSteps("MLP[128,784]", mlp2, big);
        }
        finally { TensorCodecOptions.SetCurrent(orig); }
    }

    private void TimeSteps(string label, object network, Tensor<float> input)
    {
        var host = FindField(network.GetType(), "_compileHost")?.GetValue(network);
        var plan = FindField(host!.GetType(), "_hotPlan")?.GetValue(host);
        if (plan is null) { _out.WriteLine($"{label}: no hot plan"); return; }

        // Time SetInputs + full Execute.
        var setInputs = plan.GetType().GetMethod("SetInputs");
        var execute = plan.GetType().GetMethod("Execute", Type.EmptyTypes);
        double tSet = MinUs(50, () => setInputs!.Invoke(plan, new object[] { new[] { input } }));
        double tExec = MinUs(50, () => execute!.Invoke(plan, null));
        _out.WriteLine($"{label}: SetInputs={tSet:F0}us  Execute={tExec:F0}us");

        // Per-step closures.
        var steps = plan.GetType().GetProperty("Steps", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(plan) as Array;
        if (steps is null) return;
        var engine = AiDotNetEngine.Current;
        int i = 0;
        foreach (var s in steps)
        {
            var opName = s!.GetType().GetField("OpName", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s);
            var outBuf = s.GetType().GetField("OutputBuffer", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s);
            var exec = s.GetType().GetField("Execute", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s) as Delegate;
            if (exec is null) { _out.WriteLine($"  [{i++}] {opName}: no Execute"); continue; }
            double t = MinUs(50, () => exec.DynamicInvoke(engine, outBuf));
            var m = exec.Method;
            _out.WriteLine($"  [{i++}] {opName}: {t:F0}us  exec={m.DeclaringType?.FullName}.{m.Name}");

            // For FusedLinear steps, inspect the captured inputs' fast-path-relevant
            // properties and re-time the same call with fresh contiguous copies.
            if ((string)opName! == "FusedLinear")
            {
                var ins = s.GetType().GetField("Inputs", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s) as Tensor<float>[];
                if (ins is not null)
                {
                    for (int k = 0; k < ins.Length; k++)
                    {
                        var tns = ins[k];
                        _out.WriteLine($"      in[{k}] shape=[{string.Join(",", tns.Shape)}] contig={tns.IsContiguous} sparse={tns.IsSparse} type={tns.GetType().Name}");
                    }
                    if (ins.Length >= 2)
                    {
                        var a = new Tensor<float>(ins[0].AsSpan().ToArray(), ins[0].Shape.ToArray());
                        var w = new Tensor<float>(ins[1].AsSpan().ToArray(), ins[1].Shape.ToArray());
                        var b = ins.Length > 2 ? new Tensor<float>(ins[2].AsSpan().ToArray(), ins[2].Shape.ToArray()) : null;
                        var capA = ins[0]; var capW = ins[1]; var capB = ins.Length > 2 ? ins[2] : null;
                        double tf = MinUs(50, () => engine.FusedLinear(a, w, b, AiDotNet.Tensors.Engines.FusedActivationType.ReLU));
                        double tCapA = MinUs(50, () => engine.FusedLinear(capA, w, b, AiDotNet.Tensors.Engines.FusedActivationType.ReLU));
                        double tCapW = MinUs(50, () => engine.FusedLinear(a, capW, capB, AiDotNet.Tensors.Engines.FusedActivationType.ReLU));
                        double tCapBoth = MinUs(50, () => engine.FusedLinear(capA, capW, capB, AiDotNet.Tensors.Engines.FusedActivationType.ReLU));
                        _out.WriteLine($"      fresh-all={tf:F0}us  capA+freshW={tCapA:F0}us  freshA+capW={tCapW:F0}us  capBoth={tCapBoth:F0}us");
                        _out.WriteLine($"      backing lens: capA={capA.GetDataArray().Length}/{capA.Length}  capW={capW.GetDataArray().Length}/{capW.Length}");
                    }
                }
            }
        }
    }

    private static double MinUs(int iters, Action act)
    {
        for (int i = 0; i < 3; i++) act();
        double min = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            act();
            double us = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1e6 / System.Diagnostics.Stopwatch.Frequency;
            if (us < min) min = us;
        }
        return min;
    }

    private void Dump(string label, object network)
    {
        var hostField = network.GetType().BaseType is { } ? FindField(network.GetType(), "_compileHost") : null;
        hostField ??= FindField(network.GetType(), "_compileHost");
        var host = hostField?.GetValue(network);
        if (host is null) { _out.WriteLine($"{label}: no _compileHost"); return; }
        var planField = FindField(host.GetType(), "_hotPlan");
        var plan = planField?.GetValue(host);
        if (plan is null) { _out.WriteLine($"{label}: no hot plan (replay not captured)"); return; }
        _out.WriteLine($"{label}: plan type = {plan.GetType().Name}");
        var stepsProp = plan.GetType().GetProperty("Steps", BindingFlags.NonPublic | BindingFlags.Instance);
        var steps = stepsProp?.GetValue(plan) as Array;
        if (steps is null) { _out.WriteLine($"{label}: no Steps"); return; }
        _out.WriteLine($"{label}: {steps.Length} steps:");
        int i = 0;
        foreach (var s in steps)
        {
            var opName = s!.GetType().GetField("OpName", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s);
            var outBuf = s.GetType().GetField("OutputBuffer", BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance)?.GetValue(s) as Tensor<float>;
            _out.WriteLine($"  [{i++}] {opName}  out=[{(outBuf is null ? "?" : string.Join(",", outBuf.Shape))}]");
        }
    }

    private static FieldInfo? FindField(Type t, string name)
    {
        for (var cur = t; cur is not null; cur = cur.BaseType)
        {
            var f = cur.GetField(name, BindingFlags.NonPublic | BindingFlags.Instance);
            if (f is not null) return f;
        }
        return null;
    }
}

// Re-exposes the parity test's builders for the diagnostic without duplicating them.
internal static class CompiledInferenceParityTestsAccessor
{
    public static AiDotNet.NeuralNetworks.NeuralNetworkBase<float> BuildCnn()
        => (AiDotNet.NeuralNetworks.NeuralNetworkBase<float>)typeof(CompiledInferenceParityTests)
            .GetMethod("BuildAisEvalCnn", BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, null)!;
    public static AiDotNet.NeuralNetworks.NeuralNetworkBase<float> BuildMlp()
        => (AiDotNet.NeuralNetworks.NeuralNetworkBase<float>)typeof(CompiledInferenceParityTests)
            .GetMethod("BuildAisEvalMlp", BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, null)!;
    public static Tensor<float> MakeInput(int[] shape, int seed)
        => (Tensor<float>)typeof(CompiledInferenceParityTests)
            .GetMethod("MakeInput", BindingFlags.NonPublic | BindingFlags.Static)!.Invoke(null, new object[] { shape, seed })!;
}
