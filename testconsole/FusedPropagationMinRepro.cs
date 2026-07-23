using System.Diagnostics;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole;

/// <summary>
/// Minimal reproducer that isolates which layer type breaks fused-path
/// parameter propagation. Each case asks: "after N fused Adam steps via
/// TryStepWithFusedOptimizer, did the Tensor&lt;T&gt; references that
/// layer.GetTrainableParameters() returned change in-place?"
/// </summary>
public static class FusedPropagationMinRepro
{
    public static void Run()
    {
        var listener = new TextWriterTraceListener(Console.Out);
        Trace.Listeners.Add(listener);
        Trace.AutoFlush = true;

        Console.WriteLine($"EnableCompilation: {TensorCodecOptions.Current.EnableCompilation}");

        RunCase("DenseLayer (known good baseline)", () =>
        {
            IActivationFunction<float> idAct = new IdentityActivation<float>();
            var dense = new DenseLayer<float>(4, idAct);
            return ((IReadOnlyList<ITrainableLayer<float>>)new List<ITrainableLayer<float>> { dense },
                (Func<Tensor<float>, Tensor<float>>)(x => dense.Forward(x)),
                new int[] { 4, 8 }, new int[] { 4, 4 });
        });

        RunCase("ConvolutionalLayer (1x1, 1→4, stride=1)", () =>
        {
            IActivationFunction<float> idAct = new IdentityActivation<float>();
            var conv = new ConvolutionalLayer<float>(
                outputDepth: 4, kernelSize: 1, stride: 1, padding: 0,
                activationFunction: idAct);
            return ((IReadOnlyList<ITrainableLayer<float>>)new List<ITrainableLayer<float>> { conv },
                (Func<Tensor<float>, Tensor<float>>)(x => conv.Forward(x)),
                new int[] { 1, 1, 4, 4 }, new int[] { 1, 4, 4, 4 });
        });

        RunCase("BatchNormalizationLayer (rank-2, training mode)", () =>
        {
            var bn = new BatchNormalizationLayer<float>();
            bn.SetTrainingMode(true);
            return ((IReadOnlyList<ITrainableLayer<float>>)new List<ITrainableLayer<float>> { bn },
                (Func<Tensor<float>, Tensor<float>>)(x => bn.Forward(x)),
                new int[] { 4, 4 }, new int[] { 4, 4 });
        });

        RunCase("Conv1x1 → BN → LeakyReLU (mini-GraFPrint stem)", () =>
        {
            IActivationFunction<float> idAct = new IdentityActivation<float>();
            var conv = new ConvolutionalLayer<float>(
                outputDepth: 4, kernelSize: 1, stride: 1, padding: 0,
                activationFunction: idAct);
            var bn = new BatchNormalizationLayer<float>();
            bn.SetTrainingMode(true);
            IActivationFunction<float> leakyRelu = new LeakyReLUActivation<float>(0.2);
            var act = new ActivationLayer<float>(leakyRelu);
            return ((IReadOnlyList<ITrainableLayer<float>>)new List<ITrainableLayer<float>> { conv, bn, act },
                (Func<Tensor<float>, Tensor<float>>)(x => act.Forward(bn.Forward(conv.Forward(x)))),
                new int[] { 1, 1, 4, 4 }, new int[] { 1, 4, 4, 4 });
        });
    }

    private static void RunCase(
        string label,
        Func<(IReadOnlyList<ITrainableLayer<float>> layers,
              Func<Tensor<float>, Tensor<float>> forward,
              int[] inputShape, int[] outputShape)> build)
    {
        Console.WriteLine();
        Console.WriteLine($"--- {label} ---");

        using var arena = TensorArena.Create();
        var (layers, forward, inputShape, outputShape) = build();

        var input = new Tensor<float>(inputShape);
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        var target = new Tensor<float>(outputShape);
        for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);

        // Warmup forward — materializes lazy params before we collect refs.
        forward(input);

        var paramRefs = new List<Tensor<float>>();
        foreach (var layer in layers)
            foreach (var p in layer.GetTrainableParameters()) paramRefs.Add(p);

        Console.WriteLine($"  param refs: {paramRefs.Count} tensor(s), total {paramRefs.Sum(p => p.Length)} scalars");

        var preSnapshot = paramRefs.Select(p =>
        {
            var arr = new float[p.Length];
            for (int i = 0; i < p.Length; i++) arr[i] = p[i];
            return arr;
        }).ToList();

        Func<Tensor<float>, Tensor<float>, Tensor<float>> mseLoss = (pred, tgt) =>
        {
            var engine = AiDotNetEngine.Current;
            var diff = engine.TensorSubtract(pred, tgt);
            var sq = engine.TensorMultiply(diff, diff);
            return engine.ReduceSum(sq, null);
        };

        var methodInfo = typeof(AiDotNet.Training.CompiledTapeTrainingStep<float>)
            .GetMethod("TryStepWithFusedOptimizer",
                System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic);
        if (methodInfo is null)
        {
            Console.WriteLine("  TryStepWithFusedOptimizer not reachable; aborting case.");
            return;
        }

        AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();

        const int numSteps = 20;
        int successCount = 0;
        float? firstLoss = null, lastLoss = null;
        for (int s = 0; s < numSteps; s++)
        {
            var args = new object?[]
            {
                layers, input, target, forward, mseLoss,
                OptimizerType.Adam, 0.01f, 0.9f, 0.999f, 1e-8f, 0f,
                0f, // out lossValue
            };
            bool ok = false;
            try { ok = (bool)methodInfo.Invoke(null, args)!; }
            catch (Exception ex) { Console.WriteLine($"  step {s} threw: {ex.InnerException?.Message ?? ex.Message}"); break; }
            if (!ok)
            {
                Console.WriteLine($"  step {s}: fused returned false (count={successCount})");
                break;
            }
            successCount++;
            float l = (float)args[11]!;
            if (firstLoss is null) firstLoss = l;
            lastLoss = l;
        }

        Console.WriteLine($"  fused steps: {successCount}/{numSteps}, loss {firstLoss?.ToString("F6") ?? "n/a"} → {lastLoss?.ToString("F6") ?? "n/a"}");

        for (int i = 0; i < paramRefs.Count; i++)
        {
            float maxAbs = 0;
            var p = paramRefs[i];
            var prev = preSnapshot[i];
            for (int k = 0; k < Math.Min(p.Length, prev.Length); k++)
            {
                float d = Math.Abs(p[k] - prev[k]);
                if (d > maxAbs) maxAbs = d;
            }
            Console.WriteLine($"  param[{i}] len={p.Length,-6} max |Δ|={maxAbs:E6}");
        }

        // Also report fresh GetTrainableParameters after training — if the
        // post-step list is identity-different from paramRefs, the layer
        // rebound its tensor references between collect-time and now.
        var paramRefsAfter = new List<Tensor<float>>();
        foreach (var layer in layers)
            foreach (var p in layer.GetTrainableParameters()) paramRefsAfter.Add(p);
        int rebinds = 0;
        for (int i = 0; i < paramRefs.Count && i < paramRefsAfter.Count; i++)
            if (!ReferenceEquals(paramRefs[i], paramRefsAfter[i])) rebinds++;
        Console.WriteLine($"  tensor identity rebinds (collect-time vs post-training GetTrainableParameters): {rebinds} / {paramRefs.Count}");
    }
}
