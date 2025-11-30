namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of tree-based speculative decoding.
/// </summary>
public class TreeSpeculativeResult
{
    /// <summary>All tokens.</summary>
    public int[] Tokens { get; set; } = [];

    /// <summary>Newly generated tokens.</summary>
    public int[] NewTokens { get; set; } = [];

    /// <summary>Number of new tokens.</summary>
    public int NumGenerated { get; set; }

    /// <summary>Acceptance rate.</summary>
    public double AcceptanceRate { get; set; }

    /// <summary>Step statistics.</summary>
    public List<TreeStepStatistics> StepStatistics { get; set; } = [];
}
