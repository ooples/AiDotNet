namespace AiDotNet.Document;

/// <summary>
/// Represents the result of text recognition from a cropped image.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Text recognition reads the actual characters from an image
/// of text. This result contains the recognized text string along with confidence
/// scores for each character.
/// </para>
/// </remarks>
public class TextRecognitionResult<T>
{
    /// <summary>
    /// Gets the recognized text string.
    /// </summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>
    /// Gets the overall confidence score (0-1).
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the per-character confidence scores.
    /// </summary>
    public IReadOnlyList<CharacterRecognition<T>> Characters { get; init; } = [];

    /// <summary>
    /// Gets the character-level probability distribution (shape: [seq_len, vocab_size]).
    /// </summary>
    public Tensor<T>? CharacterProbabilities { get; init; }

    /// <summary>
    /// Gets the attention weights for visualization (if available).
    /// </summary>
    public Tensor<T>? AttentionWeights { get; init; }

    /// <summary>
    /// Gets the processing time in milliseconds.
    /// </summary>
    public double ProcessingTimeMs { get; init; }

    /// <summary>
    /// Gets alternative recognition hypotheses with their scores.
    /// </summary>
    public IReadOnlyList<(string Text, double Score)> Alternatives { get; init; } = [];
}

/// <summary>
/// Represents a single recognized character with its confidence.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CharacterRecognition<T>
{
    /// <summary>
    /// Gets the recognized character.
    /// </summary>
    public char Character { get; init; }

    /// <summary>
    /// Gets the confidence score for this character.
    /// </summary>
    public T Confidence { get; init; } = default!;

    /// <summary>
    /// Gets the confidence as a double value.
    /// </summary>
    public double ConfidenceValue { get; init; }

    /// <summary>
    /// Gets the position in the sequence (0-based index).
    /// </summary>
    public int Position { get; init; }

    /// <summary>
    /// Gets alternative characters with their probabilities.
    /// </summary>
    public IReadOnlyList<(char Character, double Probability)> Alternatives { get; init; } = [];
}
