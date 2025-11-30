namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of draft token generation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DraftResult<T>
{
    /// <summary>
    /// Gets the generated draft tokens.
    /// </summary>
    public int[] Tokens { get; set; } = [];

    /// <summary>
    /// Gets the probability distributions for each draft position.
    /// Shape: [num_draft_tokens, vocab_size]
    /// </summary>
    public T[,] Probabilities { get; set; } = new T[0, 0];

    /// <summary>
    /// Gets the sampled token probabilities (p(token) for each drafted token).
    /// </summary>
    public float[] TokenProbabilities { get; set; } = [];

    /// <summary>
    /// Gets the number of draft tokens generated.
    /// </summary>
    public int NumTokens => Tokens.Length;
}
