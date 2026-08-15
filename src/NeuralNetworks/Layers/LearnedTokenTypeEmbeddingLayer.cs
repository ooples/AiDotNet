using AiDotNet.Attributes;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Adds a learned BERT token-type embedding to <c>[S,C]</c> or <c>[B,S,C]</c> states.
/// </summary>
/// <remarks>
/// The single-report MedCLIP text contract uses segment zero for every token, while retaining
/// BERT's complete two-row token-type parameter table for checkpoint compatibility.
/// </remarks>
[LayerCategory(LayerCategory.Embedding)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4, 8", TestConstructorArgs = "2, 8")]
public sealed class LearnedTokenTypeEmbeddingLayer<T> : LayerBase<T>
{
    private readonly int _tokenTypeCount;
    private readonly int _embeddingSize;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private readonly Tensor<T> _embeddings;

    /// <summary>Gets the number of token-type rows retained in the checkpoint.</summary>
    public int TokenTypeCount => _tokenTypeCount;

    /// <summary>Gets the embedding width.</summary>
    public int EmbeddingSize => _embeddingSize;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override long ParameterCount => _embeddings.Length;

    /// <summary>Creates a BERT-compatible token-type embedding table.</summary>
    public LearnedTokenTypeEmbeddingLayer(int tokenTypeCount, int embeddingSize)
        : base([1, embeddingSize], [1, embeddingSize])
    {
        if (tokenTypeCount <= 0) throw new ArgumentOutOfRangeException(nameof(tokenTypeCount));
        if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
        _tokenTypeCount = tokenTypeCount;
        _embeddingSize = embeddingSize;
        _embeddings = new Tensor<T>([tokenTypeCount, embeddingSize]);
        InitializeLayerWeights(_embeddings, embeddingSize, embeddingSize);
        RegisterTrainableParameter(_embeddings, PersistentTensorRole.Embeddings);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[^1] != _embeddingSize)
            throw new ArgumentException(
                $"Expected [S,{_embeddingSize}] or [B,S,{_embeddingSize}], got [{string.Join(',', input.Shape)}].",
                nameof(input));

        var segmentZero = Engine.TensorSlice(_embeddings, [0, 0], [1, _embeddingSize]);
        if (input.Rank == 3)
            segmentZero = Engine.Reshape(segmentZero, [1, 1, _embeddingSize]);
        return Engine.TensorBroadcastAdd(input, segmentZero);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters() => Vector<T>.FromMemory(_embeddings.Data);

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _embeddings.Length)
            throw new ArgumentException(
                $"Expected {_embeddings.Length} parameters, got {parameters.Length}.", nameof(parameters));
        parameters.AsSpan().CopyTo(_embeddings.Data.Span);
        Engine.InvalidatePersistentTensor(_embeddings);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _embeddings.Length) return;
        for (int i = 0; i < _embeddings.Length; i++)
            _embeddings[i] = NumOps.Subtract(
                _embeddings[i], NumOps.Multiply(learningRate, gradients[i]));
        Engine.InvalidatePersistentTensor(_embeddings);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["TokenTypeCount"] = _tokenTypeCount.ToString(
            System.Globalization.CultureInfo.InvariantCulture);
        metadata["EmbeddingSize"] = _embeddingSize.ToString(
            System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
