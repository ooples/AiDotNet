using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Adds a learned absolute position embedding to [S,C] or [B,S,C] sequences.</summary>
[LayerCategory(LayerCategory.Positional)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4, 8", TestConstructorArgs = "4, 8")]
public class LearnedPositionalEmbeddingLayer<T> : LayerBase<T>
{
    private readonly int _maxSequenceLength;
    private readonly int _embeddingSize;

    [TrainableParameter(Role = PersistentTensorRole.Embeddings)]
    private readonly Tensor<T> _embeddings;

    public int MaxSequenceLength => _maxSequenceLength;
    public int EmbeddingSize => _embeddingSize;
    public override bool SupportsTraining => true;
    public override long ParameterCount => _embeddings.Length;

    public LearnedPositionalEmbeddingLayer(int maxSequenceLength, int embeddingSize)
        : base([maxSequenceLength, embeddingSize], [maxSequenceLength, embeddingSize])
    {
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (embeddingSize <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingSize));
        _maxSequenceLength = maxSequenceLength;
        _embeddingSize = embeddingSize;
        _embeddings = new Tensor<T>([maxSequenceLength, embeddingSize]);
        InitializeLayerWeights(_embeddings, embeddingSize, embeddingSize);
        RegisterTrainableParameter(_embeddings, PersistentTensorRole.Embeddings);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank is not (2 or 3) || input.Shape[^1] != _embeddingSize)
            throw new ArgumentException(
                $"Expected [S,{_embeddingSize}] or [B,S,{_embeddingSize}], got [{string.Join(',', input.Shape)}].",
                nameof(input));
        int sequence = input.Shape[^2];
        if (sequence > _maxSequenceLength)
            throw new ArgumentException($"Sequence length {sequence} exceeds {_maxSequenceLength}.", nameof(input));

        var positions = Engine.TensorSlice(_embeddings, [0, 0], [sequence, _embeddingSize]);
        if (input.Rank == 3)
            positions = Engine.Reshape(positions, [1, sequence, _embeddingSize]);
        return Engine.TensorBroadcastAdd(input, positions);
    }

    public override Vector<T> GetParameters() => Vector<T>.FromMemory(_embeddings.Data);

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _embeddings.Length)
            throw new ArgumentException($"Expected {_embeddings.Length} parameters, got {parameters.Length}.");
        parameters.AsSpan().CopyTo(_embeddings.Data.Span);
        Engine.InvalidatePersistentTensor(_embeddings);
    }

    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != _embeddings.Length) return;
        for (int i = 0; i < _embeddings.Length; i++)
            _embeddings[i] = NumOps.Subtract(_embeddings[i], NumOps.Multiply(learningRate, gradients[i]));
        Engine.InvalidatePersistentTensor(_embeddings);
    }

    public override void ResetState()
    {
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["MaxSequenceLength"] = _maxSequenceLength.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["EmbeddingSize"] = _embeddingSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
