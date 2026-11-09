using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The Conformer's convolution subsampling front end (Gulati et al. 2020, §2): two 3x3 convolutions with stride 2,
/// each followed by ReLU, over the [time, mel] plane, then a linear projection of the flattened channels and
/// frequencies to the model width. It shortens the sequence four times before the attention blocks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The layout follows the reference implementation the paper's encoder is distributed in (ESPnet's
/// <c>Conv2dSubsampling</c>): unpadded kernels, so <c>T</c> frames become <c>((T - 1) / 2 - 1) / 2</c> and the
/// <c>F</c> mel bins likewise, and the projection takes <c>channels * F'</c> features. Axes before the last two
/// are batch axes and are kept.
/// </para>
/// <para>
/// <b>For Beginners:</b> Speech features arrive 100 times a second. Attention over every one of those frames is
/// slow and unnecessary, so the encoder first learns a convolutional summary of each 40 ms of audio and attends
/// over those instead.
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ConvSubsamplingLayer<T> : LayerBase<T>, IShapeContract
{
    /// <summary>The smallest time or frequency extent two unpadded stride-2 3x3 convolutions can reduce.</summary>
    public const int MinimumExtent = 7;

    private readonly int _inputFeatures;
    private readonly int _modelDim;
    private readonly int _subsampledFeatures;

    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly FullyConnectedLayer<T> _projection;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates a convolution subsampling front end.
    /// </summary>
    /// <param name="inputFeatures">Feature (mel) bins per input frame.</param>
    /// <param name="modelDim">The encoder width; also the convolutions' channel count, as in the paper.</param>
    public ConvSubsamplingLayer([LayerState] int inputFeatures, [LayerState] int modelDim)
        : base(new[] { -1, inputFeatures }, new[] { -1, modelDim })
    {
        if (inputFeatures < MinimumExtent)
            throw new ArgumentOutOfRangeException(nameof(inputFeatures),
                $"Two stride-2 3x3 convolutions need at least {MinimumExtent} feature bins; got {inputFeatures}.");
        if (modelDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelDim), "modelDim must be positive.");

        _inputFeatures = inputFeatures;
        _modelDim = modelDim;
        _subsampledFeatures = Subsampled(inputFeatures);

        var relu = (Interfaces.IActivationFunction<T>)new ReLUActivation<T>();
        // Explicit channel counts: the kernels do not depend on the time or frequency extent, so the parameter
        // count is known at construction and a clone slices the flat parameter vector correctly.
        _conv1 = ConvolutionalLayer<T>.WithInputDepth(1, modelDim, kernelSize: 3, stride: 2, padding: 0,
            activationFunction: relu);
        _conv2 = ConvolutionalLayer<T>.WithInputDepth(modelDim, modelDim, kernelSize: 3, stride: 2, padding: 0,
            activationFunction: relu);
        _projection = new FullyConnectedLayer<T>(modelDim * _subsampledFeatures, modelDim,
            (Interfaces.IActivationFunction<T>)new IdentityActivation<T>());

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_projection);
    }

    /// <summary>The length a time or frequency extent has after the two convolutions.</summary>
    public static int Subsampled(int extent) => ((extent - 1) / 2 - 1) / 2;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank < 2)
            throw new ArgumentException(
                $"Convolution subsampling takes [..., time, features]; got rank {input.Rank}.", nameof(input));

        // Leading axes (batch, or a singleton channel before it) fold into one batch axis and come back after.
        int rank = input.Rank;
        int batch = 1;
        for (int i = 0; i < rank - 2; i++) batch *= input.Shape[i];
        int time = input.Shape[rank - 2];
        int features = input.Shape[rank - 1];
        if (features != _inputFeatures)
            throw new ArgumentException(
                $"Convolution subsampling was built for {_inputFeatures} feature bins; got {features}.", nameof(input));
        if (time < MinimumExtent)
            throw new ArgumentException(
                $"Convolution subsampling needs at least {MinimumExtent} frames; got {time}.", nameof(input));

        // [B, T, F] -> [B, 1, T, F] -> conv -> [B, C, T', F'] -> [B, T', C, F'] -> [B, T', C * F'] -> [B, T', D].
        var image = Engine.Reshape(input, new[] { batch, 1, time, features });
        var convolved = _conv2.Forward(_conv1.Forward(image));
        int subsampledTime = convolved.Shape[2];
        var timeMajor = Engine.TensorPermute(convolved, new[] { 0, 2, 1, 3 });
        var flattened = Engine.Reshape(timeMajor, new[] { batch, subsampledTime, _modelDim * _subsampledFeatures });
        var projected = _projection.Forward(flattened);
        if (rank == 3) return projected;
        var outputShape = input.Shape.ToArray();
        outputShape[rank - 2] = subsampledTime;
        outputShape[rank - 1] = _modelDim;
        return Engine.Reshape(projected, outputShape);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _projection.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _conv1.ResetState();
        _conv2.ResetState();
        _projection.ResetState();
    }
}
