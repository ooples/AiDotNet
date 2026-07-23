using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of tree-based speculative decoding.
/// </summary>
internal class TreeSpeculativeResult
{
    /// <summary>
    /// All tokens (input + generated).
    /// </summary>
    public Vector<int> Tokens { get; set; } = new Vector<int>(0);

    /// <summary>
    /// Newly generated tokens.
    /// </summary>
    public Vector<int> NewTokens { get; set; } = new Vector<int>(0);

    /// <summary>
    /// Number of new tokens generated.
    /// </summary>
    public int NumGenerated { get; set; }

    /// <summary>
    /// Node acceptance rate.
    /// </summary>
    public double AcceptanceRate { get; set; }

    /// <summary>
    /// Statistics for each step.
    /// </summary>
    public List<TreeStepStatistics> StepStatistics { get; set; } = new();
}
