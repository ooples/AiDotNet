using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine.Speculative;

// RandomHelper (AiDotNet.Tensors.Helpers) provides seeded/secure RNG; MathHelper provides numeric ops.

/// <summary>Counters describing how effective speculation was over a generation run.</summary>
public sealed class SpeculationStatistics
{
    /// <summary>Total draft tokens proposed by the drafter.</summary>
    public long DraftedTokens { get; internal set; }

    /// <summary>Draft tokens accepted by the target (matched its greedy choice).</summary>
    public long AcceptedTokens { get; internal set; }

    /// <summary>Total tokens emitted (accepted drafts + correction/bonus tokens).</summary>
    public long GeneratedTokens { get; internal set; }

    /// <summary>Number of target-model forward passes performed (one per speculation round).</summary>
    public long TargetForwardPasses { get; internal set; }

    /// <summary>Fraction of drafted tokens that were accepted (0..1); 0 when nothing was drafted.</summary>
    public double AcceptanceRate => DraftedTokens == 0 ? 0.0 : (double)AcceptedTokens / DraftedTokens;

    /// <summary>
    /// Average tokens emitted per target forward pass. Greater than 1 means speculation paid off (each
    /// expensive target pass produced more than one token).
    /// </summary>
    public double TokensPerForwardPass => TargetForwardPasses == 0 ? 0.0 : (double)GeneratedTokens / TargetForwardPasses;
}

/// <summary>
/// Speculative decoding over any <see cref="ICausalLmModel{T}"/> (Leviathan 2023 / Chen 2023): a cheap
/// <see cref="ISpeculativeDrafter"/> proposes several next tokens, the target model verifies them all in one
/// forward pass (its logits already cover every drafted position), and each is accepted or corrected — plus one
/// guaranteed bonus token per fully-accepted round. Supports both greedy decoding (output <b>bit-identical</b>
/// to plain greedy) and stochastic sampling (the emitted-token distribution is provably <b>exactly</b> the
/// target's sampling distribution). Either way only the number of expensive target passes drops.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> normally a model makes one token per (slow) step. Here a fast guesser proposes,
/// say, 4 tokens; the slow model checks all 4 at once. Every guess the model agrees with is kept for free, and
/// the model always contributes at least one token itself — so you can get several tokens out of a single slow
/// step, with exactly the same result you would have gotten one-at-a-time.</para>
/// </remarks>
/// <typeparam name="T">The model's numeric type.</typeparam>
public sealed class SpeculativeGenerator<T>
{
    private readonly ICausalLmModel<T> _target;
    private readonly ISpeculativeDrafter _drafter;
    private readonly int _maxDraftTokens;

    /// <summary>Creates a greedy speculative generator.</summary>
    /// <param name="target">The target (full-accuracy) model.</param>
    /// <param name="drafter">The draft source. Defaults to a model-free <see cref="PromptLookupDrafter"/>.</param>
    /// <param name="maxDraftTokens">Tokens proposed per round (the speculation length). Default 4.</param>
    public SpeculativeGenerator(ICausalLmModel<T> target, ISpeculativeDrafter? drafter = null, int maxDraftTokens = 4)
    {
        _target = target ?? throw new ArgumentNullException(nameof(target));
        if (maxDraftTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxDraftTokens));
        _drafter = drafter ?? new PromptLookupDrafter();
        _maxDraftTokens = maxDraftTokens;
    }

    /// <summary>
    /// Generates a continuation for a prompt using greedy speculative decoding, returning the generated token
    /// ids and (optionally) speculation statistics.
    /// </summary>
    /// <param name="promptTokenIds">The prompt token ids.</param>
    /// <param name="sampling">Sampling parameters. Must be greedy (temperature 0); stop conditions honored.</param>
    /// <param name="statistics">Receives speculation counters for this run.</param>
    public IReadOnlyList<int> Generate(
        IReadOnlyList<int> promptTokenIds,
        SamplingParameters sampling,
        out SpeculationStatistics statistics)
    {
        if (promptTokenIds is null) throw new ArgumentNullException(nameof(promptTokenIds));
        if (promptTokenIds.Count == 0) throw new ArgumentException("Prompt must be non-empty.", nameof(promptTokenIds));
        if (sampling is null) throw new ArgumentNullException(nameof(sampling));
        sampling.Validate();

        statistics = new SpeculationStatistics();
        var context = new List<int>(promptTokenIds);
        var generated = new List<int>();
        int vocab = _target.VocabularySize;
        var rng = sampling.Seed is { } seed
            ? RandomHelper.CreateSeededRandom(seed)
            : RandomHelper.CreateSecureRandom();

        while (generated.Count < sampling.MaxTokens)
        {
            var draft = _drafter.Draft(context, Math.Min(_maxDraftTokens, sampling.MaxTokens - generated.Count));
            int d = draft.Count;
            statistics.DraftedTokens += d;

            // One target pass verifies all d drafted positions plus the next-token position after them.
            var verifyInput = new List<int>(context.Count + d);
            verifyInput.AddRange(context);
            verifyInput.AddRange(draft);
            var logits = Forward(verifyInput);
            statistics.TargetForwardPasses++;

            int basePos = context.Count - 1; // position whose logits predict the first continuation token
            bool stopped = sampling.IsGreedy
                ? GreedyRound(logits, draft, basePos, vocab, context, generated, statistics, sampling)
                : StochasticRound(logits, draft, basePos, vocab, context, generated, statistics, sampling, rng);

            if (stopped) break;
        }

        return generated;
    }

    // Greedy: accept a draft token iff it equals the target's argmax; emit the target's argmax on rejection,
    // plus a bonus argmax when the whole draft is accepted. Output equals plain greedy decoding.
    private bool GreedyRound(
        Tensor<T> logits, IReadOnlyList<int> draft, int basePos, int vocab,
        List<int> context, List<int> generated, SpeculationStatistics stats, SamplingParameters sampling)
    {
        int d = draft.Count;
        int i = 0;
        for (; i < d; i++)
        {
            int targetToken = ArgMaxAt(logits, basePos + i, vocab);
            if (targetToken != draft[i])
            {
                Emit(targetToken, context, generated, stats, sampling);
                return generated.Count >= sampling.MaxTokens || IsStop(targetToken, generated, sampling);
            }
            stats.AcceptedTokens++;
            if (Emit(draft[i], context, generated, stats, sampling)) return true;
        }
        int bonus = ArgMaxAt(logits, basePos + d, vocab);
        Emit(bonus, context, generated, stats, sampling);
        return generated.Count >= sampling.MaxTokens || IsStop(bonus, generated, sampling);
    }

    // Stochastic speculative sampling (Leviathan 2023 / Chen 2023): with a point-mass drafter, accept draft
    // token x with probability p(x) under the target's sampling distribution; on rejection sample from the
    // residual (p with x removed, renormalized); emit a bonus draw from p when the whole draft is accepted. The
    // emitted-token distribution is exactly the target's sampling distribution.
    private bool StochasticRound(
        Tensor<T> logits, IReadOnlyList<int> draft, int basePos, int vocab,
        List<int> context, List<int> generated, SpeculationStatistics stats, SamplingParameters sampling, Random rng)
    {
        int d = draft.Count;
        int i = 0;
        for (; i < d; i++)
        {
            var p = LogitsSampler.ComputeProbabilities(RowVector(logits, basePos + i, vocab), sampling, context);
            double acceptProb = p[draft[i]];
            if (rng.NextDouble() < acceptProb)
            {
                stats.AcceptedTokens++;
                if (Emit(draft[i], context, generated, stats, sampling)) return true;
            }
            else
            {
                var residual = (double[])p.Clone();
                residual[draft[i]] = 0.0;
                int replacement = LogitsSampler.SampleFromDistribution(residual, rng);
                Emit(replacement, context, generated, stats, sampling);
                return generated.Count >= sampling.MaxTokens || IsStop(replacement, generated, sampling);
            }
        }
        var bonusP = LogitsSampler.ComputeProbabilities(RowVector(logits, basePos + d, vocab), sampling, context);
        int bonus = LogitsSampler.SampleFromDistribution(bonusP, rng);
        Emit(bonus, context, generated, stats, sampling);
        return generated.Count >= sampling.MaxTokens || IsStop(bonus, generated, sampling);
    }

    // Extracts the vocabulary logits row at a sequence position as a Vector, tolerating [seq,vocab] / [1,seq,vocab].
    private static Vector<T> RowVector(Tensor<T> logits, int position, int vocab)
    {
        var shape = logits.Shape;
        var row = new T[vocab];
        if (shape.Length == 2)
            for (int v = 0; v < vocab; v++) row[v] = logits[position, v];
        else if (shape.Length == 3)
            for (int v = 0; v < vocab; v++) row[v] = logits[0, position, v];
        else
            throw new InvalidOperationException(
                $"Speculative verification needs multi-position logits ([seq, vocab] or [1, seq, vocab]); got rank {shape.Length}.");
        return new Vector<T>(row);
    }

    /// <summary>Convenience overload without statistics.</summary>
    public IReadOnlyList<int> Generate(IReadOnlyList<int> promptTokenIds, SamplingParameters sampling)
        => Generate(promptTokenIds, sampling, out _);

    // Appends a token to context+output; returns true if generation should stop after it.
    private bool Emit(int token, List<int> context, List<int> generated, SpeculationStatistics stats, SamplingParameters sampling)
    {
        generated.Add(token);
        context.Add(token);
        stats.GeneratedTokens++;
        return IsStop(token, generated, sampling) || generated.Count >= sampling.MaxTokens;
    }

    private bool IsStop(int token, List<int> generated, SamplingParameters sampling)
    {
        if (generated.Count < sampling.MinTokens) return false;
        if (!sampling.IgnoreEos && _target.EosTokenId is { } eos && token == eos) return true;
        if (sampling.StopTokenIds is { } stops && stops.Contains(token)) return true;
        return false;
    }

    private Tensor<T> Forward(IReadOnlyList<int> tokens)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var input = new Tensor<T>(new[] { 1, tokens.Count });
        for (int i = 0; i < tokens.Count; i++) input[0, i] = numOps.FromDouble(tokens[i]);
        return _target.ForwardLogits(input);
    }

    // Argmax over the vocab at a specific sequence position, tolerating [seq,vocab] / [1,seq,vocab] logits.
    private static int ArgMaxAt(Tensor<T> logits, int position, int vocab)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var shape = logits.Shape;
        Func<int, double> at = shape.Length switch
        {
            2 => v => numOps.ToDouble(logits[position, v]),
            3 => v => numOps.ToDouble(logits[0, position, v]),
            _ => throw new InvalidOperationException(
                $"Speculative verification needs multi-position logits ([seq, vocab] or [1, seq, vocab]); got rank {shape.Length}."),
        };

        int best = 0;
        double bestScore = at(0);
        for (int v = 1; v < vocab; v++)
        {
            double s = at(v);
            if (s > bestScore) { bestScore = s; best = v; }
        }
        return best;
    }
}
