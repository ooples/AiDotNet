using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine;

/// <summary>
/// Turns a row of next-token logits into a concrete token id under a set of <see cref="SamplingParameters"/>.
/// This is the serving engine's decoding kernel: it applies repetition / presence / frequency penalties, then
/// temperature, then the top-k / top-p (nucleus) / min-p filters, and finally samples (or takes the argmax for
/// greedy decoding) — the same pipeline and ordering used by vLLM and Hugging Face so results line up with the
/// wider ecosystem.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> a model doesn't output a word — it outputs a score (logit) for every possible
/// next token. This class is the referee that turns those thousands of scores into one chosen token. It first
/// nudges scores to discourage repeating what was already said (penalties), then decides how adventurous to be
/// (temperature), then narrows the field to the most plausible tokens (top-k / top-p / min-p), and finally
/// either picks the single best one (greedy) or rolls a weighted die among the survivors.</para>
/// <para>The method is pure and deterministic given the same logits, parameters, token history, and RNG state,
/// which is what makes greedy decoding reproducible and seeded sampling repeatable.</para>
/// </remarks>
public static class LogitsSampler
{
    /// <summary>
    /// Samples a token id from <paramref name="logits"/> under <paramref name="parameters"/>.
    /// </summary>
    /// <typeparam name="T">The model's numeric type.</typeparam>
    /// <param name="logits">Next-token logits over the full vocabulary (length == vocab size).</param>
    /// <param name="parameters">Decoding parameters (already validated).</param>
    /// <param name="contextTokenIds">Tokens seen so far (prompt + generated) used for the repetition /
    /// presence / frequency penalties. Pass an empty list to disable penalties.</param>
    /// <param name="rng">RNG used for stochastic sampling; ignored on the greedy path. Seed it per request for
    /// reproducibility.</param>
    /// <returns>The chosen token id (an index into the vocabulary).</returns>
    public static int Sample<T>(
        Vector<T> logits,
        SamplingParameters parameters,
        IReadOnlyList<int> contextTokenIds,
        Random rng)
    {
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (rng is null) throw new ArgumentNullException(nameof(rng));

        int vocab = logits.Length;
        if (vocab == 0) throw new ArgumentException("Logits must be non-empty.", nameof(logits));

        var scores = new double[vocab];
        for (int i = 0; i < vocab; i++) scores[i] = Convert.ToDouble(logits[i]);

        ApplyPenalties(scores, parameters, contextTokenIds);

        // Greedy decoding: the argmax of the (penalized) logits. Temperature scaling by a positive constant
        // does not change the argmax, so we can return here directly.
        if (parameters.IsGreedy) return ArgMax(scores);

        double temperature = parameters.Temperature;
        for (int i = 0; i < vocab; i++) scores[i] /= temperature;

        var probabilities = Softmax(scores);

        // Candidate token ids sorted by descending probability so the filters slice from the front.
        var candidates = new List<int>(vocab);
        for (int i = 0; i < vocab; i++) candidates.Add(i);
        candidates.Sort((a, b) => probabilities[b].CompareTo(probabilities[a]));

        ApplyTopK(candidates, parameters.TopK);
        ApplyMinP(candidates, probabilities, parameters.MinP);
        ApplyTopP(candidates, probabilities, parameters.TopP);

        return SampleFromCandidates(candidates, probabilities, rng);
    }

    private static void ApplyPenalties(double[] scores, SamplingParameters p, IReadOnlyList<int> context)
    {
        bool anyPenalty = Math.Abs(p.RepetitionPenalty - 1.0) > 1e-12
            || p.PresencePenalty != 0.0
            || p.FrequencyPenalty != 0.0;
        if (!anyPenalty || context is null || context.Count == 0) return;

        var counts = new Dictionary<int, int>();
        foreach (int id in context)
        {
            if (id < 0 || id >= scores.Length) continue;
            counts[id] = counts.TryGetValue(id, out int c) ? c + 1 : 1;
        }

        foreach (var kvp in counts)
        {
            int id = kvp.Key;
            int count = kvp.Value;

            // Repetition penalty (Keskar et al. 2019): divide positive logits / multiply negative logits by
            // the penalty so already-seen tokens become less likely regardless of logit sign.
            if (Math.Abs(p.RepetitionPenalty - 1.0) > 1e-12)
                scores[id] = scores[id] > 0 ? scores[id] / p.RepetitionPenalty : scores[id] * p.RepetitionPenalty;

            // Presence (once seen) and frequency (scaled by count) penalties, OpenAI-style additive form.
            scores[id] -= p.PresencePenalty;
            scores[id] -= p.FrequencyPenalty * count;
        }
    }

    private static double[] Softmax(double[] scores)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];

        var probs = new double[scores.Length];
        double sum = 0.0;
        for (int i = 0; i < scores.Length; i++)
        {
            double e = Math.Exp(scores[i] - max);
            probs[i] = e;
            sum += e;
        }
        if (sum <= 0.0 || double.IsNaN(sum)) // degenerate (e.g. all -inf); fall back to uniform.
        {
            double u = 1.0 / scores.Length;
            for (int i = 0; i < scores.Length; i++) probs[i] = u;
            return probs;
        }
        for (int i = 0; i < scores.Length; i++) probs[i] /= sum;
        return probs;
    }

    private static void ApplyTopK(List<int> candidates, int topK)
    {
        if (topK > 0 && topK < candidates.Count)
            candidates.RemoveRange(topK, candidates.Count - topK);
    }

    private static void ApplyMinP(List<int> candidates, double[] probs, double minP)
    {
        if (minP <= 0.0 || candidates.Count == 0) return;
        // candidates[0] holds the max probability (list is sorted descending).
        double threshold = minP * probs[candidates[0]];
        int keep = candidates.Count;
        for (int i = 0; i < candidates.Count; i++)
        {
            if (probs[candidates[i]] < threshold) { keep = i; break; }
        }
        if (keep < candidates.Count && keep >= 1)
            candidates.RemoveRange(keep, candidates.Count - keep);
    }

    private static void ApplyTopP(List<int> candidates, double[] probs, double topP)
    {
        if (topP >= 1.0 || candidates.Count == 0) return;
        double cumulative = 0.0;
        int keep = candidates.Count;
        for (int i = 0; i < candidates.Count; i++)
        {
            cumulative += probs[candidates[i]];
            if (cumulative >= topP) { keep = i + 1; break; } // keep through the token that crosses the mass
        }
        if (keep < candidates.Count)
            candidates.RemoveRange(keep, candidates.Count - keep);
    }

    private static int SampleFromCandidates(List<int> candidates, double[] probs, Random rng)
    {
        double total = 0.0;
        foreach (int id in candidates) total += probs[id];
        if (total <= 0.0) return candidates[0]; // degenerate; return the most probable survivor.

        double threshold = rng.NextDouble() * total;
        double accumulated = 0.0;
        foreach (int id in candidates)
        {
            accumulated += probs[id];
            if (threshold <= accumulated) return id;
        }
        return candidates[candidates.Count - 1]; // floating-point guard
    }

    private static int ArgMax(double[] scores)
    {
        int best = 0;
        double bestScore = scores[0];
        for (int i = 1; i < scores.Length; i++)
        {
            if (scores[i] > bestScore) { bestScore = scores[i]; best = i; }
        }
        return best;
    }
}
