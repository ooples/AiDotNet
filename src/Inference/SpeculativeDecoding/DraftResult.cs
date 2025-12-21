using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of draft token generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class DraftResult<T>
{
    /// <summary>
    /// Gets the generated draft tokens.
    /// </summary>
    public Vector<int> Tokens { get; set; } = new Vector<int>(0);

    /// <summary>
    /// Gets the probability distributions for each draft position.
    /// Shape: [num_draft_tokens, vocab_size]
    /// </summary>
    public Matrix<T> Probabilities { get; set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the sampled token probabilities (p(token) for each drafted token).
    /// </summary>
    public Vector<T> TokenProbabilities { get; set; } = new Vector<T>(0);

    /// <summary>
    /// Gets the number of draft tokens generated.
    /// </summary>
    public int NumTokens => Tokens.Length;
}
