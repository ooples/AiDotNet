// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Generation;

/// <summary>
/// Picks the next token from a logits vector for autoregressive decoding (#1632 / #95): greedy
/// (argmax), temperature, top-k, and top-p (nucleus) sampling. Centralises the logic that was
/// previously reimplemented inside GPT4Vision / Blip / Flamingo. Pure function of the logits +
/// <see cref="SamplingOptions"/> (no model coupling), so it's fully unit-testable.
/// </summary>
/// <typeparam name="T">Numeric type of the logits.</typeparam>
public static class TokenSampler<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Index of the maximum logit — deterministic greedy decode.</summary>
    public static int ArgMax(Vector<T> logits)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (logits.Length == 0) throw new ArgumentException("logits must be non-empty.", nameof(logits));
        int best = 0;
        double bestVal = NumOps.ToDouble(logits[0]);
        for (int i = 1; i < logits.Length; i++)
        {
            double v = NumOps.ToDouble(logits[i]);
            if (v > bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    /// <summary>
    /// Samples the next-token index from <paramref name="logits"/> under <paramref name="options"/>.
    /// Greedy options return <see cref="ArgMax"/>. Otherwise applies temperature, then top-k, then
    /// top-p masking, softmaxes the survivors, and draws from that distribution.
    /// </summary>
    /// <param name="rng">RNG to draw from. When null, a seeded RNG (if <see cref="SamplingOptions.Seed"/>
    /// is set) or the shared thread-safe RNG is used.</param>
    public static int Sample(Vector<T> logits, SamplingOptions options, Random? rng = null)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (logits.Length == 0) throw new ArgumentException("logits must be non-empty.", nameof(logits));
        if (options.IsGreedy) return ArgMax(logits);

        int n = logits.Length;
        // Temperature-scaled logits in double space (sampling returns an index, so no T output).
        var scaled = new double[n];
        for (int i = 0; i < n; i++) scaled[i] = NumOps.ToDouble(logits[i]) / options.Temperature;

        // Mask = "is this token a candidate". Top-k and top-p both narrow it.
        var keep = new bool[n];
        for (int i = 0; i < n; i++) keep[i] = true;

        int topK = options.TopK;
        if (topK > 0 && topK < n)
        {
            // Keep the topK highest-scaled logits.
            var order = Enumerable.Range(0, n).OrderByDescending(i => scaled[i]).ToArray();
            for (int r = topK; r < n; r++) keep[order[r]] = false;
        }

        // Softmax over the currently-kept set (numerically stable).
        var probs = Softmax(scaled, keep);

        double topP = options.TopP;
        if (topP > 0.0 && topP < 1.0)
        {
            // Nucleus: keep the smallest high-prob set whose cumulative prob ≥ topP.
            var order = Enumerable.Range(0, n).Where(i => keep[i]).OrderByDescending(i => probs[i]).ToArray();
            double cum = 0.0;
            int cutoff = order.Length; // default: keep all
            for (int r = 0; r < order.Length; r++)
            {
                cum += probs[order[r]];
                if (cum >= topP) { cutoff = r + 1; break; }
            }
            for (int r = cutoff; r < order.Length; r++) keep[order[r]] = false;
            probs = Softmax(scaled, keep); // renormalise over the nucleus
        }

        // Draw u in [0,1) and walk the cumulative distribution over the kept tokens.
        double u = (rng ?? ResolveRng(options)).NextDouble();
        double acc = 0.0;
        int last = -1;
        for (int i = 0; i < n; i++)
        {
            if (!keep[i]) continue;
            last = i;
            acc += probs[i];
            if (u < acc) return i;
        }
        // Floating-point slack: return the last kept token (probs sum to ~1).
        return last >= 0 ? last : ArgMax(logits);
    }

    private static double[] Softmax(double[] scaled, bool[] keep)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < scaled.Length; i++)
            if (keep[i] && scaled[i] > max) max = scaled[i];
        if (double.IsNegativeInfinity(max)) max = 0.0;

        var probs = new double[scaled.Length];
        double sum = 0.0;
        for (int i = 0; i < scaled.Length; i++)
        {
            if (!keep[i]) { probs[i] = 0.0; continue; }
            double e = Math.Exp(scaled[i] - max);
            probs[i] = e;
            sum += e;
        }
        if (sum > 0.0)
            for (int i = 0; i < scaled.Length; i++) probs[i] /= sum;
        return probs;
    }

    private static Random ResolveRng(SamplingOptions options)
        => options.Seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(options.Seed.Value)
            : Tensors.Helpers.RandomHelper.ThreadSafeRandom;
}
