namespace AiDotNet.Serving.ContinuousBatching;

/// <summary>
/// A single (token id, log-probability) pair, used to report the log-probabilities of candidate tokens at
/// a decode position (OpenAI <c>logprobs</c> / <c>top_logprobs</c>).
/// </summary>
public sealed class TokenLogProb
{
    /// <summary>The vocabulary token id.</summary>
    public int TokenId { get; set; }

    /// <summary>The natural-log probability of this token under the (temperature-scaled) model distribution.</summary>
    public float LogProb { get; set; }
}

/// <summary>
/// Log-probability information for one generated position: the chosen token's log-probability plus, when
/// requested, the top-N most likely alternatives at that step.
/// </summary>
public sealed class PositionLogProbs
{
    /// <summary>The token id that was actually emitted at this position.</summary>
    public int TokenId { get; set; }

    /// <summary>The natural-log probability of the emitted token.</summary>
    public float LogProb { get; set; }

    /// <summary>The most likely tokens at this position (highest log-prob first), including the chosen token.
    /// Empty when only the chosen token's log-probability was requested.</summary>
    public List<TokenLogProb> TopLogProbs { get; set; } = new();
}
