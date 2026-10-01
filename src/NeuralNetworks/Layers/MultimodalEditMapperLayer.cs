using AiDotNet.Attributes;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Maps joint image/instruction hidden states to a fixed-length diffusion context using learned
/// queries and an encoder-decoder transformer with true cross-attention.
/// </summary>
/// <remarks>
/// The default geometry follows the released MGIE mapper: eight 4096-wide edit states, a
/// 512-wide four-head transformer with four encoder and four decoder blocks, 77 learned queries,
/// and a 768-wide output. This is a trainable native component, not a pretrained checkpoint loader.
/// Its edit-token embeddings are both the appended MLLM input tokens and the residual embeddings
/// added to their contextual hidden states before the mapper's input projection.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3,
    Cost = ComputeCost.High, TestInputShape = "1, 2, 8", TestConstructorArgs = "8, 8, 8, 2, 3, 2, 1")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class MultimodalEditMapperLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly int _editTokenCount;
    private readonly int _queryCount;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly double _dropoutRate;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _editTokenEmbeddings;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _queryEmbeddings;
    [SubLayerInput("1, _inputDim")]
    private readonly DenseLayer<T> _inputProjection;
    [SubLayerInput("1, _hiddenDim")]
    private readonly DenseLayer<T> _outputProjection;
    private readonly List<TransformerEncoderBlock<T>> _encoders = new();
    private readonly List<TransformerDecoderBlock<T>> _decoders = new();
    private readonly LayerNormalizationLayer<T> _encoderNorm;
    private readonly LayerNormalizationLayer<T> _decoderNorm;

    /// <summary>Creates a registered, trainable fixed-query multimodal mapper.</summary>
    public MultimodalEditMapperLayer(
        [LayerState] int inputDim = 4096,
        [LayerState] int hiddenDim = 512,
        [LayerState] int outputDim = 768,
        [LayerState] int editTokenCount = 8,
        [LayerState] int queryCount = 77,
        [LayerState] int numHeads = 4,
        [LayerState] int numLayers = 4,
        [LayerState] double dropoutRate = 0)
        : base(new[] { -1, editTokenCount, inputDim }, new[] { -1, queryCount, outputDim })
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (hiddenDim <= 0 || hiddenDim > int.MaxValue / 4) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));
        if (editTokenCount <= 0) throw new ArgumentOutOfRangeException(nameof(editTokenCount));
        if (queryCount <= 0) throw new ArgumentOutOfRangeException(nameof(queryCount));
        if (numHeads <= 0 || hiddenDim % numHeads != 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (double.IsNaN(dropoutRate) || double.IsInfinity(dropoutRate) || dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        _inputDim = inputDim;
        _hiddenDim = hiddenDim;
        _outputDim = outputDim;
        _editTokenCount = editTokenCount;
        _queryCount = queryCount;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _dropoutRate = dropoutRate;

        _editTokenEmbeddings = InitializeEmbeddings(editTokenCount, inputDim);
        _queryEmbeddings = InitializeEmbeddings(queryCount, hiddenDim);
        RegisterTrainableParameter(_editTokenEmbeddings, PersistentTensorRole.Weights);
        AppendTrainableParameter(_queryEmbeddings, PersistentTensorRole.Weights);
        _inputProjection = new DenseLayer<T>(hiddenDim);
        _outputProjection = new DenseLayer<T>(outputDim);
        RegisterSubLayer(_inputProjection);
        for (int i = 0; i < numLayers; i++)
        {
            var encoder = new TransformerEncoderBlock<T>(hiddenDim, numHeads, hiddenDim * 4, dropoutRate);
            var decoder = new TransformerDecoderBlock<T>(hiddenDim, numHeads, hiddenDim * 4, dropoutRate);
            _encoders.Add(encoder);
            _decoders.Add(decoder);
            RegisterSubLayer(encoder);
            RegisterSubLayer(decoder);
        }
        _encoderNorm = new LayerNormalizationLayer<T>(hiddenDim);
        _decoderNorm = new LayerNormalizationLayer<T>(hiddenDim);
        RegisterSubLayer(_encoderNorm);
        RegisterSubLayer(_decoderNorm);
        RegisterSubLayer(_outputProjection);
    }

    /// <summary>The live edit-token input embeddings. Internal callers do not detach their tape edge.</summary>
    internal Tensor<T> EditTokenEmbeddings => _editTokenEmbeddings;

    internal bool TrainingMode => IsTrainingMode;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 3
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Fixed(_queryCount)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputDim))
        }
        : null;

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in GetSubLayers()) layer.ResetState();
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[0] <= 0 || input.Shape[1] != _editTokenCount || input.Shape[2] != _inputDim)
            throw new ArgumentException($"Expected [batch, {_editTokenCount}, {_inputDim}] edit hidden states.", nameof(input));

        int batch = input.Shape[0];
        var embeddings = Engine.Reshape(_editTokenEmbeddings, new[] { 1, _editTokenCount, _inputDim });
        var enriched = Engine.TensorAdd(input, embeddings);
        var memory = _inputProjection.Forward(Engine.Reshape(enriched, new[] { batch * _editTokenCount, _inputDim }));
        memory = Engine.Reshape(memory, new[] { batch, _editTokenCount, _hiddenDim });
        foreach (var encoder in _encoders) memory = encoder.Forward(memory);
        memory = _encoderNorm.Forward(memory);

        var queries = Engine.Reshape(_queryEmbeddings, new[] { 1, _queryCount, _hiddenDim });
        if (batch > 1)
        {
            var copies = new Tensor<T>[batch];
            for (int i = 0; i < batch; i++) copies[i] = queries;
            queries = Engine.TensorConcatenate(copies, axis: 0);
        }
        foreach (var decoder in _decoders) queries = decoder.Forward(queries, memory);
        queries = _decoderNorm.Forward(queries);
        var projected = _outputProjection.Forward(Engine.Reshape(queries, new[] { batch * _queryCount, _hiddenDim }));
        return Engine.Reshape(projected, new[] { batch, _queryCount, _outputDim });
    }

    private Tensor<T> InitializeEmbeddings(int rows, int width)
    {
        var embeddings = new Tensor<T>(new[] { rows, width });
        double scale = 1.0 / Math.Sqrt(width);
        for (int i = 0; i < embeddings.Length; i++)
            embeddings[i] = NumOps.FromDouble((Random.NextDouble() * 2.0 - 1.0) * scale);
        return embeddings;
    }
}
