using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Metrics;

/// <summary>
/// Static helpers for evaluating language-model / classification predictions over a vocabulary:
/// <b>perplexity</b> (how surprised the model is by the true tokens) and <b>top-k accuracy</b>
/// (how often the true token is among the model's k most likely guesses).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A language model outputs, for each position, a score for every word
/// in its vocabulary. Two standard ways to grade those outputs:</para>
/// <list type="bullet">
///   <item><b>Perplexity</b> — think of it as "on average, how many words was the model torn
///   between?". A perfect model that always puts all its confidence on the right word has
///   perplexity 1. A model that is uniformly unsure across a 50,000-word vocabulary has
///   perplexity 50,000. Lower is better. It is exactly <c>exp(mean cross-entropy)</c>.</item>
///   <item><b>Top-k accuracy</b> — the fraction of positions where the true word is in the model's
///   top k guesses. Top-1 is ordinary accuracy; top-5 is more forgiving. Higher is better.</item>
/// </list>
/// <para>These live next to the other <c>AiDotNet.Metrics</c> helpers and are the standard signals
/// for gating a language-model training run (e.g. "abort if perplexity stops falling").</para>
/// </remarks>
public static class LanguageModelMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes perplexity = <c>exp(mean negative-log-likelihood)</c> of the true tokens under the
    /// model's predicted distributions.
    /// </summary>
    /// <param name="predictions">A [N, V] tensor: for each of N positions, a score over V vocabulary
    /// entries. Interpreted as raw logits when <paramref name="fromLogits"/> is true (a numerically
    /// stable softmax is applied), or as already-normalized probabilities otherwise.</param>
    /// <param name="targets">The N true vocabulary indices (one per row).</param>
    /// <param name="fromLogits">True (default) if <paramref name="predictions"/> are raw logits;
    /// false if they are probabilities that already sum to 1 per row.</param>
    /// <returns>The perplexity as a <typeparamref name="T"/> (≥ 1; lower is better).</returns>
    /// <exception cref="ArgumentNullException">A required argument is null.</exception>
    /// <exception cref="ArgumentException">Shapes are inconsistent or empty.</exception>
    public static T Perplexity(Tensor<T> predictions, int[] targets, bool fromLogits = true)
        => NumOps.FromDouble(Math.Exp(MeanCrossEntropyDouble(predictions, targets, fromLogits)));

    /// <summary>
    /// Computes the mean cross-entropy (average negative log-likelihood, in nats) of the true tokens.
    /// Perplexity is <c>exp</c> of this value; it is exposed separately because a training monitor
    /// typically streams the loss (cross-entropy) and reports perplexity alongside it.
    /// </summary>
    /// <inheritdoc cref="Perplexity(Tensor{T}, int[], bool)"/>
    /// <returns>
    /// The mean cross-entropy (negative log-likelihood) in nats, &gt;= 0; lower is better. (Perplexity is
    /// <c>exp</c> of this, so it is &gt;= 1 — this method overrides the inherited perplexity return text.)
    /// </returns>
    public static T CrossEntropy(Tensor<T> predictions, int[] targets, bool fromLogits = true)
        => NumOps.FromDouble(MeanCrossEntropyDouble(predictions, targets, fromLogits));

    /// <summary>
    /// Computes top-k accuracy: the fraction of positions whose true token is among the model's k
    /// highest-scoring vocabulary entries. <paramref name="k"/> = 1 is ordinary top-1 accuracy.
    /// </summary>
    /// <param name="predictions">A [N, V] tensor of per-position scores over the vocabulary. Only the
    /// ordering matters, so logits and probabilities give the same result.</param>
    /// <param name="targets">The N true vocabulary indices (one per row).</param>
    /// <param name="k">How many top guesses to consider (1 ≤ k ≤ V).</param>
    /// <returns>The fraction in [0, 1] as a <typeparamref name="T"/> (higher is better).</returns>
    public static T TopKAccuracy(Tensor<T> predictions, int[] targets, int k)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        var (n, v) = ValidateShape(predictions, targets);
        if (k < 1 || k > v)
            throw new ArgumentException($"k must be in [1, {v}] (the vocabulary size), was {k}.", nameof(k));

        int hits = 0;
        for (int i = 0; i < n; i++)
        {
            int target = targets[i];
            if (target < 0 || target >= v)
                throw new ArgumentException($"target[{i}]={target} is outside the vocabulary [0, {v}).", nameof(targets));

            // The true token is in the top k iff strictly fewer than k other tokens outscore it.
            double targetScore = NumOps.ToDouble(predictions[i, target]);
            int strictlyGreater = 0;
            for (int j = 0; j < v; j++)
            {
                if (j == target) continue;
                if (NumOps.ToDouble(predictions[i, j]) > targetScore && ++strictlyGreater >= k)
                    break;
            }
            if (strictlyGreater < k) hits++;
        }
        return NumOps.FromDouble((double)hits / n);
    }

    // ---- shared numerics ----

    private static double MeanCrossEntropyDouble(Tensor<T> predictions, int[] targets, bool fromLogits)
    {
        if (predictions is null) throw new ArgumentNullException(nameof(predictions));
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        var (n, v) = ValidateShape(predictions, targets);

        double sumNll = 0.0;
        for (int i = 0; i < n; i++)
        {
            int target = targets[i];
            if (target < 0 || target >= v)
                throw new ArgumentException($"target[{i}]={target} is outside the vocabulary [0, {v}).", nameof(targets));

            if (fromLogits)
            {
                // Numerically stable log-softmax: nll = logSumExp(row) - row[target].
                double max = double.NegativeInfinity;
                for (int j = 0; j < v; j++)
                {
                    double z = NumOps.ToDouble(predictions[i, j]);
                    if (z > max) max = z;
                }
                double sumExp = 0.0;
                for (int j = 0; j < v; j++)
                    sumExp += Math.Exp(NumOps.ToDouble(predictions[i, j]) - max);
                double logSumExp = max + Math.Log(sumExp);
                sumNll += logSumExp - NumOps.ToDouble(predictions[i, target]);
            }
            else
            {
                double p = NumOps.ToDouble(predictions[i, target]);
                // Guard against log(0) for a zero-probability true token.
                sumNll += -Math.Log(Math.Max(p, 1e-12));
            }
        }
        return sumNll / n;
    }

    private static (int n, int v) ValidateShape(Tensor<T> predictions, int[] targets)
    {
        if (predictions.Rank != 2)
            throw new ArgumentException($"predictions must be a 2-D [N, V] tensor, had rank {predictions.Rank}.", nameof(predictions));
        int n = predictions.Shape[0];
        int v = predictions.Shape[1];
        if (n == 0 || v == 0)
            throw new ArgumentException("predictions must be non-empty in both dimensions.", nameof(predictions));
        if (targets.Length != n)
            throw new ArgumentException($"targets length {targets.Length} must equal the number of rows {n}.", nameof(targets));
        return (n, v);
    }
}
