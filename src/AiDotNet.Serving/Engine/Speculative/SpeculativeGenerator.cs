using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Serving.Engine.Speculative;

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
/// Greedy speculative decoding over any <see cref="ICausalLmModel{T}"/> (Leviathan 2023 / Chen 2023): a cheap
/// <see cref="ISpeculativeDrafter"/> proposes several next tokens, the target model verifies them all in one
/// forward pass (its logits already cover every drafted position), and the longest correct prefix is accepted —
/// plus one guaranteed correction or bonus token per round. The emitted sequence is <b>bit-identical</b> to
/// plain greedy decoding; only the number of expensive target passes drops.
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
        if (!sampling.IsGreedy)
            throw new NotSupportedException(
                "SpeculativeGenerator currently supports greedy decoding only (temperature 0). Use the " +
                "standard generation path for stochastic sampling.");

        statistics = new SpeculationStatistics();
        var context = new List<int>(promptTokenIds);
        var generated = new List<int>();
        int vocab = _target.VocabularySize;

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
            bool stopped = false;

            // Accept the longest prefix of the draft that matches the target's greedy choice.
            int i = 0;
            for (; i < d; i++)
            {
                int targetToken = ArgMaxAt(logits, basePos + i, vocab);
                if (targetToken != draft[i])
                {
                    // Reject here: emit the target's own token instead, then end this round.
                    if (Emit(targetToken, context, generated, statistics, sampling)) stopped = true;
                    break;
                }
                statistics.AcceptedTokens++;
                if (Emit(draft[i], context, generated, statistics, sampling)) { stopped = true; i++; break; }
            }

            if (!stopped && i == d)
            {
                // Whole draft accepted (or empty): also emit the bonus token the target predicts after it.
                int bonus = ArgMaxAt(logits, basePos + d, vocab);
                Emit(bonus, context, generated, statistics, sampling);
                // A stop here simply ends the outer loop on the next check.
                if (IsStop(bonus, generated, sampling)) stopped = true;
            }

            if (stopped) break;
        }

        return generated;
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
