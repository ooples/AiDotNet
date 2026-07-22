using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Result of draft token generation returned by an <see cref="IDraftModel{T}"/>: the guessed tokens and the
/// probability distributions the draft assigned to them (used by the target model to accept/reject drafts).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a draft model guesses the next few tokens, it returns them here along with
/// how confident it was about each. The main model uses that information to decide which guesses to keep.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type.</typeparam>
public class DraftResult<T>
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
