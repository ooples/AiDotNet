using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Predicts one log-duration per token with same-padded temporal convolutions.
/// </summary>
/// <remarks>
/// Input is [batch, tokens, encoder features]; output is [batch, tokens, 1].
/// The caller supplies unpadded sequences and decides whether to detach encoder features.
/// Each block applies a width-three convolution, ReLU, channel LayerNorm, and dropout.
/// The final width-one convolution is a parallel duration head, never a decoder bottleneck.
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3,
    Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 8, 2, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TokenDurationPredictorLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _encoderDim;
    private readonly int _hiddenDim;
    private readonly int _numLayers;
    private readonly double _dropoutRate;
    private readonly List<Conv1DLayer<T>> _convolutions = new();
    private readonly List<LayerNormalizationLayer<T>> _normalizations = new();
    private readonly List<DropoutLayer<T>> _dropouts = new();
    private readonly Conv1DLayer<T> _projection;

    /// <summary>Creates a trainable duration branch whose dimensions are known at construction.</summary>
    public TokenDurationPredictorLayer(
        [LayerState] int encoderDim,
        [LayerState] int hiddenDim = 256,
        [LayerState] int numLayers = 2,
        [LayerState] double dropoutRate = 0.1)
        : base(new[] { -1, -1, encoderDim }, new[] { -1, -1, 1 })
    {
        if (encoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(encoderDim));
        if (hiddenDim <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenDim));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (double.IsNaN(dropoutRate) || double.IsInfinity(dropoutRate) || dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate));

        _encoderDim = encoderDim;
        _hiddenDim = hiddenDim;
        _numLayers = numLayers;
        _dropoutRate = dropoutRate;
        for (int i = 0; i < numLayers; i++)
        {
            var convolution = new Conv1DLayer<T>(inputChannels: i == 0 ? encoderDim : hiddenDim,
                outputChannels: hiddenDim, kernelSize: 3, padding: 1,
                activation: new ReLUActivation<T>());
            var normalization = new LayerNormalizationLayer<T>(hiddenDim);
            _convolutions.Add(convolution);
            _normalizations.Add(normalization);
            RegisterSubLayer(convolution);
            RegisterSubLayer(normalization);
            if (dropoutRate > 0)
            {
                var dropout = new DropoutLayer<T>(dropoutRate);
                _dropouts.Add(dropout);
                RegisterSubLayer(dropout);
            }
        }

        _projection = new Conv1DLayer<T>(inputChannels: hiddenDim, outputChannels: 1, kernelSize: 1);
        RegisterSubLayer(_projection);
    }

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => inputRank == 3
        ? new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Time, AxisRelation.Same(TensorAxis.Time)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(1))
        }
        : null;

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 3 || input.Shape[2] != _encoderDim || input.Shape[1] <= 0)
            throw new ArgumentException($"Expected [batch, tokens, {_encoderDim}] with nonempty tokens.", nameof(input));

        var current = input;
        for (int i = 0; i < _numLayers; i++)
        {
            current = Engine.TensorPermute(current, new[] { 0, 2, 1 });
            current = _convolutions[i].Forward(current);
            current = Engine.TensorPermute(current, new[] { 0, 2, 1 });
            current = _normalizations[i].Forward(current);
            if (_dropoutRate > 0) current = _dropouts[i].Forward(current);
        }

        current = _projection.Forward(Engine.TensorPermute(current, new[] { 0, 2, 1 }));
        return Engine.TensorPermute(current, new[] { 0, 2, 1 });
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        try
        {
            if (disposing)
                foreach (var child in GetSubLayers())
                    if (child is IDisposable disposable) AiDotNet.Helpers.DisposeOnceGuard.TryDispose(disposable);
        }
        finally { base.Dispose(disposing); }
    }
}
