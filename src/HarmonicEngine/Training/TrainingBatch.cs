namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// A mini-batch of training data for <see cref="ITrainingStrategy{T}"/>.
/// Holds the input token sequences, the target token sequences (usually the
/// inputs shifted by one for language modeling), and optional metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for tensor entries.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A "batch" in machine learning is a small group of training
/// examples processed together to make training faster and more stable than one at
/// a time. For a language model, each example is a sequence of token IDs, and the
/// target is typically the same sequence shifted forward by one position — the
/// model learns to predict each next token from the preceding ones.
/// </para>
/// </remarks>
public class TrainingBatch<T>
{
    /// <summary>
    /// Input tokens, shape <c>[batchSize, seqLen]</c>. Each entry is a token ID
    /// in <c>[0, vocabSize)</c>.
    /// </summary>
    public Tensor<T> InputTokens { get; }

    /// <summary>
    /// Target tokens, shape <c>[batchSize, seqLen]</c>. For causal language
    /// modeling, this is typically <c>InputTokens</c> shifted forward by one.
    /// </summary>
    public Tensor<T> TargetTokens { get; }

    /// <summary>
    /// Gets the batch size (number of sequences in this batch).
    /// </summary>
    public int BatchSize => InputTokens._shape[0];

    /// <summary>
    /// Gets the sequence length of each example in the batch.
    /// </summary>
    public int SequenceLength => InputTokens._shape[1];

    /// <summary>
    /// Creates a new training batch.
    /// </summary>
    /// <param name="inputTokens">Input token tensor, shape <c>[B, S]</c>.</param>
    /// <param name="targetTokens">Target token tensor, shape <c>[B, S]</c>.</param>
    public TrainingBatch(Tensor<T> inputTokens, Tensor<T> targetTokens)
    {
        if (inputTokens._shape.Length != 2)
            throw new ArgumentException(
                $"Input tokens must have shape [B, S], got {inputTokens._shape.Length}D.", nameof(inputTokens));
        if (targetTokens._shape.Length != 2)
            throw new ArgumentException(
                $"Target tokens must have shape [B, S], got {targetTokens._shape.Length}D.", nameof(targetTokens));
        if (inputTokens._shape[0] != targetTokens._shape[0] ||
            inputTokens._shape[1] != targetTokens._shape[1])
            throw new ArgumentException(
                $"Input and target tokens must have the same shape. Input: " +
                $"[{inputTokens._shape[0]},{inputTokens._shape[1]}], " +
                $"Target: [{targetTokens._shape[0]},{targetTokens._shape[1]}].");

        InputTokens = inputTokens;
        TargetTokens = targetTokens;
    }
}
