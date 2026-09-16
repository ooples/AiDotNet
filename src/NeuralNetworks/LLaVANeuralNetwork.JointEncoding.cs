using AiDotNet.Validation;

namespace AiDotNet.NeuralNetworks;

public partial class LLaVANeuralNetwork<T>
{
    /// <summary>
    /// Encodes image patches, instruction tokens and optional trailing learned tokens through
    /// the native language stack, returning hidden states rather than vocabulary logits.
    /// </summary>
    /// <param name="image">An image [channels,height,width] or batch [batch,channels,height,width].</param>
    /// <param name="tokenIds">Token IDs from the tokenizer associated with this model.</param>
    /// <param name="trailingTokenEmbeddings">Optional learned [tokens,embeddingDimension] edit/query tokens.</param>
    /// <returns>[tokens,embeddingDimension], or [batch,tokens,embeddingDimension] for a batch.</returns>
    /// <remarks>
    /// The route uses this model's real patch encoder, projector, embeddings and language layers.
    /// It preserves tape connections and the active engine; the caller chooses training or inference
    /// scope. It does not load pretrained weights or infer special-token IDs from a checkpoint.
    /// The existing ONNX generation route remains separate: arbitrary providers need not expose hidden states.
    /// </remarks>
    public Tensor<T> EncodeJointHiddenStates(Tensor<T> image, IReadOnlyList<int> tokenIds,
        Tensor<T>? trailingTokenEmbeddings = null)
    {
        Guard.NotNull(image);
        Guard.NotNull(tokenIds);
        if (!_useNativeMode)
            throw new NotSupportedException("Joint hidden-state encoding requires the native multimodal layers.");
        if (image.Rank is not (3 or 4))
            throw new ArgumentException("An image must have rank three or four.", nameof(image));
        if (tokenIds.Count == 0)
            throw new ArgumentException("At least one instruction token is required.", nameof(tokenIds));
        if (trailingTokenEmbeddings is not null &&
            (trailingTokenEmbeddings.Rank != 2 || trailingTokenEmbeddings.Shape[0] == 0 ||
             trailingTokenEmbeddings.Shape[1] != _lmHiddenDim))
            throw new ArgumentException("Trailing tokens must have the model's embedding width.", nameof(trailingTokenEmbeddings));
        int queryCount = trailingTokenEmbeddings?.Shape[0] ?? 0;
        long jointLength = (long)_numVisualTokens + 1 + tokenIds.Count + queryCount;
        if (jointLength > _maxSequenceLength)
            throw new ArgumentException("The image, instruction and trailing tokens exceed the model's sequence limit.", nameof(tokenIds));
        foreach (int token in tokenIds)
            if (token < 0 || token >= _vocabularySize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), "An instruction token is outside the model vocabulary.");

        EnsureLayerRandomSeedsWired();
        var visual = ProjectToLanguageSpace(ExtractVisualFeaturesNative(image));
        var text = EmbedTextTokens(tokenIds.ToArray());
        var joint = ConcatenateSequences(visual, text);
        if (trailingTokenEmbeddings is not null)
            joint = ConcatenateSequences(joint, trailingTokenEmbeddings);
        return ForwardLLM(joint);
    }

    /// <summary>
    /// Positions that precede the instruction tokens in <see cref="EncodeJointHiddenStates"/>: the patch tokens and
    /// the vision class token.
    /// </summary>
    internal int JointVisualTokenCount => _numVisualTokens + 1;

    /// <summary>Projects language-model hidden states [rows, width] to vocabulary logits [rows, vocabulary].</summary>
    /// <remarks>Uses the model's live LM head and keeps its tape connection, for training language objectives.</remarks>
    internal Tensor<T> ProjectToVocabulary(Tensor<T> hiddenRows)
    {
        Guard.NotNull(hiddenRows);
        if (!_useNativeMode || _outputProjection is null)
            throw new NotSupportedException("Vocabulary projection requires the native language-model head.");
        if (hiddenRows.Rank != 2 || hiddenRows.Shape[1] != _lmHiddenDim)
            throw new ArgumentException("Hidden rows must be [rows, embedding width].", nameof(hiddenRows));
        return _outputProjection.Forward(hiddenRows);
    }

    /// <summary>Tokenizes an instruction using this model's configured tokenizer.</summary>
    public IReadOnlyList<int> EncodeInstructionTokens(string instruction)
    {
        if (string.IsNullOrWhiteSpace(instruction))
            throw new ArgumentException("An instruction cannot be empty.", nameof(instruction));
        return _tokenizer.Encode(instruction).TokenIds.ToArray();
    }

    private Tensor<T> BroadcastTokenSequence(Tensor<T> sequence, int batchSize)
    {
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        var row = Engine.Reshape(sequence, new[] { 1, sequence.Shape[0], sequence.Shape[1] });
        if (batchSize == 1) return row;
        // Repeated references accumulate each batch row's gradient into the shared token tensor.
        // No CPU readback or detached constant copy is introduced.
        var rows = new Tensor<T>[batchSize];
        for (int index = 0; index < rows.Length; index++) rows[index] = row;
        return Engine.TensorConcatenate(rows, axis: 0);
    }
}
