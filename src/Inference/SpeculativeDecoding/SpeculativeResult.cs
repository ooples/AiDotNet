using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of speculative decoding generation.
/// </summary>
internal class SpeculativeResult
{
    /// <summary>
    /// All tokens (input + generated).
    /// </summary>
    public Vector<int> Tokens { get; set; } = new Vector<int>(0);

    /// <summary>
    /// Only the newly generated tokens.
    /// </summary>
    public Vector<int> NewTokens { get; set; } = new Vector<int>(0);

    /// <summary>
    /// Number of tokens generated.
    /// </summary>
    public int NumGenerated { get; set; }

    /// <summary>
    /// Overall draft acceptance rate.
    /// </summary>
    public double AcceptanceRate { get; set; }

    /// <summary>
    /// Average tokens generated per verification call.
    /// </summary>
    public double TokensPerVerification { get; set; }

    /// <summary>
    /// Statistics for each decoding step.
    /// </summary>
    public List<StepStatistics> StepStatistics { get; set; } = new();
}
