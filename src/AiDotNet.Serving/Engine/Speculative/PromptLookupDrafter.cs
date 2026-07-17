using System;
using System.Collections.Generic;

namespace AiDotNet.Serving.Engine.Speculative;

/// <summary>
/// A model-free drafter (prompt-lookup / n-gram decoding, Saxena 2023): it proposes the continuation by finding
/// the most recent earlier occurrence of the current trailing n-gram in the context and copying the tokens that
/// followed it. It needs no second model, so it delivers speculative speedups "for free" on repetitive text —
/// code, structured output, summarization that echoes the source.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> lots of generated text repeats phrases that already appeared (variable names, a
/// quoted sentence, boilerplate). This drafter simply says "last time I saw these few tokens, these came next —
/// let's guess those again." It costs almost nothing and is often right on repetitive content; when it is
/// wrong, the model corrects it, so quality never suffers.</para>
/// </remarks>
public sealed class PromptLookupDrafter : ISpeculativeDrafter
{
    private readonly int _maxNgram;
    private readonly int _minNgram;

    /// <summary>Creates a prompt-lookup drafter.</summary>
    /// <param name="maxNgram">Longest trailing n-gram to match (tried first for higher-confidence guesses).</param>
    /// <param name="minNgram">Shortest trailing n-gram to fall back to.</param>
    public PromptLookupDrafter(int maxNgram = 3, int minNgram = 1)
    {
        if (minNgram < 1) throw new ArgumentOutOfRangeException(nameof(minNgram), "minNgram must be >= 1.");
        if (maxNgram < minNgram) throw new ArgumentOutOfRangeException(nameof(maxNgram), "maxNgram must be >= minNgram.");
        _maxNgram = maxNgram;
        _minNgram = minNgram;
    }

    /// <inheritdoc/>
    public IReadOnlyList<int> Draft(IReadOnlyList<int> contextTokenIds, int maxDraftTokens)
    {
        if (contextTokenIds is null) throw new ArgumentNullException(nameof(contextTokenIds));
        if (maxDraftTokens < 1) return Array.Empty<int>();

        int len = contextTokenIds.Count;
        // Try the longest n-gram first: a longer match is a more specific (more confident) prediction.
        for (int n = Math.Min(_maxNgram, len - 1); n >= _minNgram; n--)
        {
            var draft = TryMatch(contextTokenIds, n, maxDraftTokens);
            if (draft.Count > 0) return draft;
        }
        return Array.Empty<int>();
    }

    private static IReadOnlyList<int> TryMatch(IReadOnlyList<int> ctx, int n, int maxDraftTokens)
    {
        int len = ctx.Count;
        // Search backwards for the most recent prior occurrence of the trailing n-gram ctx[len-n .. len-1].
        // The candidate window must end before the trailing n-gram itself (start + n <= len - 1) so we copy
        // tokens that genuinely followed an earlier occurrence.
        for (int start = len - n - 1; start >= 0; start--)
        {
            if (!MatchesTrailing(ctx, start, n)) continue;

            int followStart = start + n;
            int available = (len - 1) - followStart + 1; // tokens between the match and the trailing n-gram
            int take = Math.Min(maxDraftTokens, available);
            if (take <= 0) continue;

            var draft = new int[take];
            for (int i = 0; i < take; i++) draft[i] = ctx[followStart + i];
            return draft;
        }
        return Array.Empty<int>();
    }

    // True if the n tokens starting at `start` equal the trailing n tokens ctx[len-n .. len-1].
    private static bool MatchesTrailing(IReadOnlyList<int> ctx, int start, int n)
    {
        int tail = ctx.Count - n;
        for (int i = 0; i < n; i++)
            if (ctx[start + i] != ctx[tail + i]) return false;
        return true;
    }
}
