using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The variance predictor of FastSpeech 2: a 2-layer 1D convolutional network, each layer followed by ReLU,
/// layer normalization and dropout, then a linear projection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FastSpeech 2 (Ren et al. 2021, §2.3, Fig. 1c) uses this structure, with separate parameters, for its duration,
/// pitch and energy predictors: "a 2-layer 1D-convolutional network with ReLU activation, each followed by the
/// layer normalization and the dropout layer, and an extra linear layer to project the hidden states into the
/// output sequence". Appendix A: kernel 3, 256 filters for both layers, dropout 0.5.
/// </para>
/// <para>Input <c>[batch, time, hidden]</c>; output <c>[batch, time, outputSize]</c> (1 for duration and energy, the
/// number of CWT scales for the pitch spectrogram).</para>
/// <para><b>For Beginners:</b> A small network that reads the phoneme (or frame) features and guesses one number per
/// position — how long the phoneme lasts, how high or loud it is.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, Cost = ComputeCost.Medium, TestInputShape = "1, 6, 16", TestConstructorArgs = "16, 16, 1, 3, 0.0")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class VariancePredictorLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputSize;
    private readonly int _filterSize;
    private readonly int _outputSize;
    private readonly int _kernelSize;
    private readonly double _dropoutRate;

    [SubLayerInput("1, _inputSize, 1")]
    private readonly Conv1DLayer<T> _conv1;
    [SubLayerInput("_filterSize")]
    private readonly LayerNormalizationLayer<T> _norm1;
    [SubLayerInput("1, _filterSize, 1")]
    private readonly Conv1DLayer<T> _conv2;
    [SubLayerInput("_filterSize")]
    private readonly LayerNormalizationLayer<T> _norm2;
    [SubLayerInput("1, _filterSize")]
    private readonly DenseLayer<T> _projection;
    private readonly DropoutLayer<T>? _dropout1;
    private readonly DropoutLayer<T>? _dropout2;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a variance predictor.
    /// </summary>
    /// <param name="inputSize">Width of the incoming hidden sequence (256 in FastSpeech 2).</param>
    /// <param name="filterSize">Channels of both convolutions (256 in FastSpeech 2).</param>
    /// <param name="outputSize">Values predicted per position.</param>
    /// <param name="kernelSize">Kernel of both convolutions (3 in FastSpeech 2); odd, so the length is kept.</param>
    /// <param name="dropoutRate">Dropout after each normalization (0.5 in FastSpeech 2).</param>
    public VariancePredictorLayer(
        [LayerState] int inputSize,
        [LayerState] int filterSize,
        [LayerState] int outputSize = 1,
        [LayerState] int kernelSize = 3,
        [LayerState] double dropoutRate = 0.5)
        : base(new[] { inputSize }, new[] { outputSize })
    {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (filterSize <= 0) throw new ArgumentOutOfRangeException(nameof(filterSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "The kernel must be odd so 'same' padding keeps the length.");
        if (dropoutRate < 0 || dropoutRate >= 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));

        _inputSize = inputSize;
        _filterSize = filterSize;
        _outputSize = outputSize;
        _kernelSize = kernelSize;
        _dropoutRate = dropoutRate;

        _conv1 = new Conv1DLayer<T>(inputChannels: inputSize, outputChannels: filterSize, kernelSize: kernelSize,
            activation: new ReLUActivation<T>());
        _norm1 = new LayerNormalizationLayer<T>(filterSize);
        _conv2 = new Conv1DLayer<T>(inputChannels: filterSize, outputChannels: filterSize, kernelSize: kernelSize,
            activation: new ReLUActivation<T>());
        _norm2 = new LayerNormalizationLayer<T>(filterSize);
        _projection = new DenseLayer<T>(outputSize, new IdentityActivation<T>() as IActivationFunction<T>);
        if (dropoutRate > 0)
        {
            _dropout1 = new DropoutLayer<T>(dropoutRate);
            _dropout2 = new DropoutLayer<T>(dropoutRate);
        }

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_norm1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_norm2);
        RegisterSubLayer(_projection);
        if (_dropout1 is not null) RegisterSubLayer(_dropout1);
        if (_dropout2 is not null) RegisterSubLayer(_dropout2);
    }

    /// <summary>Width of the incoming hidden sequence.</summary>
    public int InputSize => _inputSize;
    /// <summary>Channels of both convolutions.</summary>
    public int FilterSize => _filterSize;
    /// <summary>Values predicted per position.</summary>
    public int OutputSize => _outputSize;
    /// <summary>Kernel of both convolutions.</summary>
    public int KernelSize => _kernelSize;
    /// <summary>Dropout probability.</summary>
    public double DropoutRate => _dropoutRate;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input) => ForwardWithHidden(input, out _);

    /// <summary>
    /// Runs the predictor and also returns the convolutional hidden sequence <c>[batch, time, filterSize]</c> that
    /// feeds the final projection.
    /// </summary>
    /// <remarks>FastSpeech 2's pitch predictor averages this hidden sequence over time to predict the utterance's
    /// pitch mean and variance (Ren et al. 2021, App. C.2).</remarks>
    public Tensor<T> ForwardWithHidden(Tensor<T> input, out Tensor<T> hidden)
    {
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] }) : input;

        var h = ConvBlock(x, _conv1, _norm1, _dropout1);
        h = ConvBlock(h, _conv2, _norm2, _dropout2);
        hidden = h;

        int batch = h.Shape[0], time = h.Shape[1];
        var flat = Engine.Reshape(h, new[] { batch * time, _filterSize });
        var projected = Engine.Reshape(_projection.Forward(flat), new[] { batch, time, _outputSize });

        if (!unbatched) return projected;
        hidden = Engine.Reshape(h, new[] { time, _filterSize });
        return Engine.Reshape(projected, new[] { time, _outputSize });
    }

    private Tensor<T> ConvBlock(Tensor<T> x, Conv1DLayer<T> conv, LayerNormalizationLayer<T> norm,
        DropoutLayer<T>? dropout)
    {
        var channelsFirst = Engine.TensorPermute(x, new[] { 0, 2, 1 }).Contiguous();
        var convolved = Engine.TensorPermute(conv.Forward(channelsFirst), new[] { 0, 2, 1 }).Contiguous();
        var normed = norm.Forward(convolved);
        return dropout is null ? normed : dropout.Forward(normed);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var child in GetSubLayers()) child.ResetState();
    }

    /// <summary>Persists the constructor arguments for deserialization.</summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputSize"] = _inputSize.ToString(inv);
        metadata["FilterSize"] = _filterSize.ToString(inv);
        metadata["OutputSize"] = _outputSize.ToString(inv);
        metadata["KernelSize"] = _kernelSize.ToString(inv);
        metadata["DropoutRate"] = _dropoutRate.ToString(inv);
        return metadata;
    }
}
