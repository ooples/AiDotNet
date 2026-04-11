namespace AiDotNet.HarmonicEngine.Training;

/// <summary>
/// A mini-batch of token sequences for language-model training. Thin wrapper
/// around the <see cref="Tensor{T}"/>-tuple format produced by the standard
/// AiDotNet data loaders (<see cref="Data.Loaders.InputOutputDataLoaderBase{T, TInput, TOutput}"/>),
/// adding shape validation and convenience accessors for <see cref="ITrainingStrategy{T}"/>
/// implementations.
/// </summary>
/// <typeparam name="T">The numeric type used for tensor entries.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A "batch" is a small group of training examples processed
/// together. For a language model, each example is a sequence of token IDs, and the
/// target is typically the same sequence shifted forward by one position — the
/// model learns to predict each next token from the preceding ones.
/// </para>
/// <para>
/// You'll usually get <see cref="TrainingBatch{T}"/> instances from a standard
/// AiDotNet data loader's <c>GetNextBatch()</c> method via the implicit tuple
/// conversion: <c>TrainingBatch&lt;T&gt;.From(loader.GetNextBatch())</c>.
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
    /// modeling, this is typically <see cref="InputTokens"/> shifted forward by one.
    /// </summary>
    public Tensor<T> TargetTokens { get; }

    /// <summary>
    /// Gets the batch size (number of sequences in this batch).
    /// </summary>
    public int BatchSize => InputTokens.Shape[0];

    /// <summary>
    /// Gets the sequence length of each example in the batch.
    /// </summary>
    public int SequenceLength => InputTokens.Shape[1];

    /// <summary>
    /// Creates a new training batch.
    /// </summary>
    /// <param name="inputTokens">Input token tensor, shape <c>[B, S]</c>.</param>
    /// <param name="targetTokens">Target token tensor, shape <c>[B, S]</c>.</param>
    public TrainingBatch(Tensor<T> inputTokens, Tensor<T> targetTokens)
    {
        if (inputTokens.Shape.Length != 2)
            throw new ArgumentException(
                $"Input tokens must have shape [B, S], got {inputTokens.Shape.Length}D.", nameof(inputTokens));
        if (targetTokens.Shape.Length != 2)
            throw new ArgumentException(
                $"Target tokens must have shape [B, S], got {targetTokens.Shape.Length}D.", nameof(targetTokens));
        if (inputTokens.Shape[0] != targetTokens.Shape[0] ||
            inputTokens.Shape[1] != targetTokens.Shape[1])
            throw new ArgumentException(
                $"Input and target tokens must have the same shape. Input: " +
                $"[{inputTokens.Shape[0]},{inputTokens.Shape[1]}], " +
                $"Target: [{targetTokens.Shape[0]},{targetTokens.Shape[1]}].");

        InputTokens = inputTokens;
        TargetTokens = targetTokens;
    }

    /// <summary>
    /// Creates a <see cref="TrainingBatch{T}"/> from a standard AiDotNet data
    /// loader's (Features, Labels) tuple. Use this to bridge between the
    /// library's loader conventions and the HRE training strategies.
    /// </summary>
    public static TrainingBatch<T> From((Tensor<T> Features, Tensor<T> Labels) batch)
    {
        return new TrainingBatch<T>(batch.Features, batch.Labels);
    }
}
