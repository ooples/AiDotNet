using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Fused per-example DP-SGD step (Abadi et al. 2016 §3, Algorithm 1). Runs a batch
/// of examples through per-example forward+backward, clips each per-example gradient
/// against the GLOBAL parameter-vector L2 norm, aggregates the clipped gradients,
/// adds a single Gaussian noise draw <c>N(0, σ² C² I)</c>, and averages by batch size.
///
/// <para><b>Note</b>: this is an AiDotNet-local mirror of the same helper in Tensors PR #763
/// (<c>AiDotNet.Tensors.Engines.Training.DpSgdStep&lt;T&gt;</c>). Once #763 merges and the
/// AiDotNet.Tensors NuGet publishes, callers should migrate to the Tensors version; this
/// local mirror is retained until then to unblock the AiDotNet-side wire-ups. Both are
/// drop-in equivalent — same public API, same clip-then-aggregate contract.</para>
///
/// <para>The clip-BEFORE-aggregate order is the L2-sensitivity bound the DP proof
/// requires — reversing it (aggregate-then-clip) breaks the privacy guarantee. This
/// helper enforces the correct order so callers cannot regress it.</para>
/// </summary>
/// <typeparam name="T">Numeric type of the parameter tensors.</typeparam>
public static class DpSgdStep<T>
{
    /// <summary>
    /// Runs a DP-SGD training step over <paramref name="batchSize"/> examples.
    /// See the Tensors version for full documentation.
    /// </summary>
    public static Dictionary<Tensor<T>, Tensor<T>> ComputeClippedAggregatedGradients(
        int batchSize,
        Func<int, Tensor<T>> perExampleLoss,
        IReadOnlyList<Tensor<T>> parameters,
        double clipNorm,
        double noiseMultiplier,
        Random rng)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize,
                "DP-SGD batch size must be positive.");
        if (clipNorm <= 0)
            throw new ArgumentOutOfRangeException(nameof(clipNorm), clipNorm,
                "DP-SGD clip norm must be positive (defines the L2-sensitivity bound).");
        if (perExampleLoss is null) throw new ArgumentNullException(nameof(perExampleLoss));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        var ops = MathHelper.GetNumericOperations<T>();

        var sums = new Dictionary<Tensor<T>, Tensor<T>>(AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var p in parameters)
        {
            if (p is null)
                throw new ArgumentException("parameters must not contain null tensors.", nameof(parameters));
            sums[p] = new Tensor<T>(p._shape);
        }

        for (int example = 0; example < batchSize; example++)
        {
            using var tape = new GradientTape<T>();
            var loss = perExampleLoss(example);
            if (loss is null)
                throw new InvalidOperationException(
                    $"perExampleLoss({example}) returned null; must return the scalar loss tensor.");
            var grads = tape.ComputeGradients(loss, parameters);

            // Global L2 norm across all parameters — Abadi 2016 Algorithm 1 line 4.
            double normSquared = 0.0;
            foreach (var g in grads.Values)
            {
                if (g is null) continue;
                var span = g.AsSpan();
                for (int i = 0; i < span.Length; i++)
                {
                    double v = ops.ToDouble(span[i]);
                    normSquared += v * v;
                }
            }
            double clipFactor = Math.Min(1.0, clipNorm / Math.Sqrt(normSquared + 1e-12));

            // Accumulate clipped per-example gradient — Abadi 2016 Algorithm 1 line 5.
            foreach (var p in parameters)
            {
                if (!grads.TryGetValue(p, out var g) || g is null) continue;
                var sumSpan = sums[p].AsWritableSpan();
                var gSpan = g.AsSpan();
                for (int i = 0; i < gSpan.Length; i++)
                {
                    double v = ops.ToDouble(sumSpan[i]) + ops.ToDouble(gSpan[i]) * clipFactor;
                    sumSpan[i] = ops.FromDouble(v);
                }
            }
        }

        // Noise + average — Abadi 2016 Algorithm 1 line 6.
        double invBatch = 1.0 / batchSize;
        double noiseStd = clipNorm * noiseMultiplier * invBatch;
        var result = new Dictionary<Tensor<T>, Tensor<T>>(AiDotNet.Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var p in parameters)
        {
            var sum = sums[p];
            var averaged = new Tensor<T>(p._shape);
            var sumSpan = sum.AsSpan();
            var avgSpan = averaged.AsWritableSpan();
            for (int i = 0; i < sumSpan.Length; i++)
            {
                double noise = noiseStd > 0 ? SampleGaussian(rng) * noiseStd : 0.0;
                avgSpan[i] = ops.FromDouble(ops.ToDouble(sumSpan[i]) * invBatch + noise);
            }
            result[p] = averaged;
        }
        return result;
    }

    private static double SampleGaussian(Random rng)
    {
        double u1;
        do { u1 = rng.NextDouble(); } while (u1 < 1e-300);
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
