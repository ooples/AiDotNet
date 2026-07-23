namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Overall statistics for speculative decoding.
/// </summary>
internal class SpeculativeDecodingStats
{
    /// <summary>Total tokens generated.</summary>
    public long TotalTokensGenerated { get; set; }

    /// <summary>Total draft tokens proposed.</summary>
    public long TotalDraftTokens { get; set; }

    /// <summary>Draft tokens that were accepted.</summary>
    public long AcceptedDraftTokens { get; set; }

    /// <summary>Total verification calls to target model.</summary>
    public long TotalVerificationCalls { get; set; }

    /// <summary>Draft acceptance rate.</summary>
    public double AcceptanceRate { get; set; }

    /// <summary>Average tokens per verification.</summary>
    public double TokensPerVerification { get; set; }

    /// <summary>Estimated speedup factor.</summary>
    public double SpeedupEstimate { get; set; }
}
