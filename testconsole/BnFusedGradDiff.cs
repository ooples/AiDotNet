using System.Diagnostics;
using System.Reflection;
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
/// Pinpoints WHERE the BatchNormalizationLayer fused-mode divergence comes
/// from by running ONE Adam step on identical input/target/init under both
/// eager and fused paths, then diffing the resulting gamma/beta tensors
/// element-by-element. If a single step produces different parameter values,
/// the underlying gradient computation differs — proving the bug is in
/// Tensors' BN backward, not in Adam itself.
/// </summary>
public static class BnFusedGradDiff
{
    public static void Run()
    {
        var listener = new TextWriterTraceListener(Console.Out);
        Trace.Listeners.Add(listener);
        Trace.AutoFlush = true;

        // Phase 0: forward-only sanity. Run the SAME BN.Forward twice on
        // the same input — once with EnableCompilation=true, once with
        // false. If outputs differ at all, the bug is in BN's compile-mode
        // forward, not in the backward / Adam step.
        ForwardOnlyDiff();

        // Same input/target seed for both runs so any divergence is purely
        // in the backward computation.
        const int seed = 42;
        var inputShape = new[] { 4, 4 };
        var outputShape = new[] { 4, 4 };

        Console.WriteLine();
        Console.WriteLine("=== EAGER trajectory (TensorCodecOptions.EnableCompilation=false) ===");
        var (eagerGamma, eagerBeta, eagerLoss) = RunOneStep(inputShape, outputShape, seed, fused: false);

        Console.WriteLine();
        Console.WriteLine("=== FUSED trajectory (compile-mode TryStepWithFusedOptimizer) ===");
        var (fusedGamma, fusedBeta, fusedLoss) = RunOneStep(inputShape, outputShape, seed, fused: true);

        Console.WriteLine();
        Console.WriteLine("=== Per-parameter divergence after 1 Adam step ===");
        Console.WriteLine($"loss eager = {eagerLoss:F8}");
        Console.WriteLine($"loss fused = {fusedLoss:F8}");
        Console.WriteLine();

        Console.WriteLine("gamma:");
        for (int i = 0; i < eagerGamma.Length; i++)
        {
            float diff = fusedGamma[i] - eagerGamma[i];
            Console.WriteLine($"  [{i}]  eager={eagerGamma[i],12:F8}  fused={fusedGamma[i],12:F8}  Δ={diff,+12:F8}  signSwap={Math.Sign(eagerGamma[i] - 1) != Math.Sign(fusedGamma[i] - 1)}");
        }

        Console.WriteLine();
        Console.WriteLine("beta:");
        for (int i = 0; i < eagerBeta.Length; i++)
        {
            float diff = fusedBeta[i] - eagerBeta[i];
            Console.WriteLine($"  [{i}]  eager={eagerBeta[i],12:F8}  fused={fusedBeta[i],12:F8}  Δ={diff,+12:F8}  signSwap={Math.Sign(eagerBeta[i]) != Math.Sign(fusedBeta[i])}");
        }
    }

    /// <summary>
    /// Pure forward-pass diff: same fresh BN, same input. Toggle
    /// EnableCompilation between runs and compare outputs element-by-element.
    /// If they differ, the divergence in TryStepWithFusedOptimizer's loss
    /// is upstream of the backward pass.
    /// </summary>
    private static void ForwardOnlyDiff()
    {
        Console.WriteLine("=== Phase 0: BN forward-only diff (no Adam, no backward) ===");

        var input = new Tensor<float>(new[] { 4, 4 });
        var rng = new Random(42);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        var bn1 = new BatchNormalizationLayer<float>();
        bn1.SetTrainingMode(true);

        var bn2 = new BatchNormalizationLayer<float>();
        bn2.SetTrainingMode(true);

        // Run 1: EnableCompilation=false. Same fresh layer, same input.
        var savedOpts = TensorCodecOptions.Current;
        var eagerOpts = new TensorCodecOptions
        {
            EnableCompilation = false,
            EnableDataflowFusion = savedOpts.EnableDataflowFusion,
            EnableAlgebraicBackward = savedOpts.EnableAlgebraicBackward,
            EnableSpectralDecomposition = savedOpts.EnableSpectralDecomposition,
            SpectralErrorTolerance = savedOpts.SpectralErrorTolerance,
            DataflowFusionMaxHidden = savedOpts.DataflowFusionMaxHidden,
            EnableConvBnFusion = savedOpts.EnableConvBnFusion,
            EnableAttentionFusion = savedOpts.EnableAttentionFusion,
            EnablePointwiseFusion = savedOpts.EnablePointwiseFusion,
            EnableForwardCSE = savedOpts.EnableForwardCSE,
            EnableBlasBatch = savedOpts.EnableBlasBatch,
            EnableMixedPrecision = savedOpts.EnableMixedPrecision,
        };
        TensorCodecOptions.SetCurrent(eagerOpts);
        var outEager = bn1.Forward(input);
        TensorCodecOptions.SetCurrent(savedOpts);

        // Run 2: EnableCompilation=true. Different fresh layer, same input.
        var outFused = bn2.Forward(input);

        Console.WriteLine($"  input:  {string.Join(",", Enumerable.Range(0, input.Length).Select(i => input[i].ToString("F6")))}");
        Console.WriteLine($"  eager:  {string.Join(",", Enumerable.Range(0, outEager.Length).Select(i => outEager[i].ToString("F6")))}");
        Console.WriteLine($"  fused:  {string.Join(",", Enumerable.Range(0, outFused.Length).Select(i => outFused[i].ToString("F6")))}");

        float maxDiff = 0;
        int firstDiff = -1;
        int n = Math.Min(outEager.Length, outFused.Length);
        for (int i = 0; i < n; i++)
        {
            float d = Math.Abs(outEager[i] - outFused[i]);
            if (d > maxDiff) { maxDiff = d; firstDiff = i; }
        }
        Console.WriteLine($"  max |Δ| = {maxDiff:E6} at index {firstDiff}");
        Console.WriteLine($"  >>> {(maxDiff < 1e-6 ? "Forward outputs MATCH — bug is in backward/Adam" : "Forward outputs DIFFER — bug is in compile-mode BN forward!")}");
    }

    private static (float[] gamma, float[] beta, float lossValue) RunOneStep(
        int[] inputShape, int[] outputShape, int seed, bool fused)
    {
        // Toggle compilation per run. Eager run gets EnableCompilation=false
        // so the tape walks backward via the eager autograd path; fused run
        // calls TryStepWithFusedOptimizer directly.
        var savedOpts = TensorCodecOptions.Current;
        if (!fused)
        {
            var eagerOpts = new TensorCodecOptions
            {
                EnableCompilation = false,
                EnableDataflowFusion = savedOpts.EnableDataflowFusion,
                EnableAlgebraicBackward = savedOpts.EnableAlgebraicBackward,
                EnableSpectralDecomposition = savedOpts.EnableSpectralDecomposition,
                SpectralErrorTolerance = savedOpts.SpectralErrorTolerance,
                DataflowFusionMaxHidden = savedOpts.DataflowFusionMaxHidden,
                EnableConvBnFusion = savedOpts.EnableConvBnFusion,
                EnableAttentionFusion = savedOpts.EnableAttentionFusion,
                EnablePointwiseFusion = savedOpts.EnablePointwiseFusion,
                EnableForwardCSE = savedOpts.EnableForwardCSE,
                EnableBlasBatch = savedOpts.EnableBlasBatch,
                EnableMixedPrecision = savedOpts.EnableMixedPrecision,
            };
            TensorCodecOptions.SetCurrent(eagerOpts);
        }

        try
        {
            using var arena = TensorArena.Create();

            // Fresh BN layer (gamma=1, beta=0 init by default). Forward once
            // on a deterministic input to materialize lazy params, then
            // capture starting values so we can verify both runs start equal.
            var bn = new BatchNormalizationLayer<float>();
            bn.SetTrainingMode(true);

            var input = new Tensor<float>(inputShape);
            var rng = new Random(seed);
            for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);

            var target = new Tensor<float>(outputShape);
            for (int i = 0; i < target.Length; i++) target[i] = (float)(rng.NextDouble() - 0.5);

            // Warmup forward to materialize lazy gamma/beta tensors.
            bn.Forward(input);

            var paramRefs = new List<Tensor<float>>();
            foreach (var p in bn.GetTrainableParameters()) paramRefs.Add(p);
            // BN's GetTrainableParameters yields gamma then beta, both length=4
            // for a 4-feature input.

            Func<Tensor<float>, Tensor<float>> forward = x => bn.Forward(x);
            Func<Tensor<float>, Tensor<float>, Tensor<float>> mseLoss = (pred, tgt) =>
            {
                var engine = AiDotNetEngine.Current;
                var diff = engine.TensorSubtract(pred, tgt);
                var sq = engine.TensorMultiply(diff, diff);
                return engine.ReduceSum(sq, null);
            };

            float lossValue = 0f;

            if (fused)
            {
                AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();
                AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();

                var methodInfo = typeof(AiDotNet.Training.CompiledTapeTrainingStep<float>)
                    .GetMethod("TryStepWithFusedOptimizer",
                        BindingFlags.Static | BindingFlags.NonPublic);
                if (methodInfo is null)
                {
                    Console.WriteLine("  TryStepWithFusedOptimizer not reachable!");
                    return (Array.Empty<float>(), Array.Empty<float>(), 0f);
                }

                var args = new object?[]
                {
                    (IReadOnlyList<ITrainableLayer<float>>)new List<ITrainableLayer<float>> { bn },
                    input, target, forward, mseLoss,
                    OptimizerType.Adam, 0.01f, 0.9f, 0.999f, 1e-8f, 0f,
                    0f,
                };
                bool ok = (bool)methodInfo.Invoke(null, args)!;
                if (!ok) Console.WriteLine("  fused step returned false");
                lossValue = (float)args[11]!;
            }
            else
            {
                // Eager path: open a tape, forward, compute loss, walk backward,
                // apply Adam manually so the step matches what fused would do.
                using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                var pred = forward(input);
                var loss = mseLoss(pred, target);
                lossValue = loss[0];

                var grads = tape.ComputeGradients(loss, sources: null);

                // Print per-grad diagnostic: gamma_grad and beta_grad
                Console.WriteLine($"  loss = {lossValue:F8}");
                for (int i = 0; i < paramRefs.Count; i++)
                {
                    var p = paramRefs[i];
                    if (grads.TryGetValue(p, out var g))
                    {
                        Console.Write($"  param[{i}] grad = ");
                        for (int k = 0; k < g.Length; k++)
                            Console.Write($"{g[k],10:F6} ");
                        Console.WriteLine();
                    }
                    else
                    {
                        Console.WriteLine($"  param[{i}] NO GRAD found in tape");
                    }
                }

                // Apply one Adam step manually with the same hyperparams
                // (lr=0.01, β1=0.9, β2=0.999, ε=1e-8). One-step bias correction
                // makes mHat=grad, vHat=grad², so the update simplifies to
                // p ← p − lr · grad / (|grad| + ε).
                var engine = AiDotNetEngine.Current;
                const float lr = 0.01f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
                float oneMinusB1 = 1 - b1, oneMinusB2 = 1 - b2;
                float bc1 = 1 - b1, bc2 = 1 - b2;

                for (int i = 0; i < paramRefs.Count; i++)
                {
                    var p = paramRefs[i];
                    if (!grads.TryGetValue(p, out var grad)) continue;

                    var m = engine.TensorMultiplyScalar(grad, oneMinusB1);
                    var v = engine.TensorMultiplyScalar(engine.TensorMultiply(grad, grad), oneMinusB2);
                    var mHat = engine.TensorDivideScalar(m, bc1);
                    var vHat = engine.TensorDivideScalar(v, bc2);
                    var denom = engine.TensorAddScalar(engine.TensorSqrt(vHat), eps);
                    var update = engine.TensorMultiplyScalar(engine.TensorDivide(mHat, denom), lr);
                    engine.TensorSubtractInPlace(p, update);
                }
            }

            // Snapshot final parameter values
            var gamma = new float[paramRefs[0].Length];
            for (int i = 0; i < gamma.Length; i++) gamma[i] = paramRefs[0][i];
            var beta = new float[paramRefs[1].Length];
            for (int i = 0; i < beta.Length; i++) beta[i] = paramRefs[1][i];

            return (gamma, beta, lossValue);
        }
        finally
        {
            if (!fused) TensorCodecOptions.SetCurrent(savedOpts);
        }
    }
}
