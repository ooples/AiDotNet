namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Statistics for a single decoding step.
/// </summary>
internal class StepStatistics
{
    /// <summary>Number of draft tokens generated.</summary>
    public int DraftTokens { get; set; }

    /// <summary>Number of draft tokens accepted.</summary>
    public int AcceptedTokens { get; set; }

    /// <summary>Whether a token was resampled due to rejection.</summary>
    public bool ResampledToken { get; set; }

    /// <summary>Whether a bonus token was sampled after full acceptance.</summary>
    public bool BonusToken { get; set; }
}
