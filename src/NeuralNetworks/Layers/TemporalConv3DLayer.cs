using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A trainable anisotropic 3D convolution for video tensors in NCFHW layout.
/// </summary>
/// <remarks>
/// Unlike <see cref="Conv3DLayer{T}"/>, this layer permits different temporal and
/// spatial kernel sizes. It is intended for paper architectures whose temporal
/// kernels are <c>(K,1,1)</c>. Forward execution is composed from
/// <see cref="IEngine.Conv3D{T}(Tensor{T}, Tensor{T}, int[], int[], int[])"/>, so
/// CPU execution, all direct-GPU backends, and gradient-tape registration share
/// the same implementation.
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 5)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public sealed partial class TemporalConv3DLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputChannels <= 0) return null;
        OutputAxisContract Window(TensorAxis axis, int kernel, int padding) =>
            new(axis, AxisRelation.Window(axis, kernel, stride: 1, padding: padding));
        var channels = new OutputAxisContract(
            TensorAxis.Channels, AxisRelation.Fixed(_outputChannels));
        var time = Window(TensorAxis.Time, _kernelDepth, _paddingDepth);
        var height = Window(TensorAxis.Height, _kernelHeight, _paddingHeight);
        var width = Window(TensorAxis.Width, _kernelWidth, _paddingWidth);
        return inputRank switch
        {
            5 =>
            [
                new(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, time, height, width
            ],
            _ => null
        };
    }

    private readonly int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelDepth;
    private readonly int _kernelHeight;
    private readonly int _kernelWidth;
    private readonly int _paddingDepth;
    private readonly int _paddingHeight;
    private readonly int _paddingWidth;
    private readonly bool _zeroInitialize;

    [TrainableParameter(Role = PersistentTensorRole.Weights,
        Shape = "_outputChannels, _inputChannels, _kernelDepth, _kernelHeight, _kernelWidth")]
    private Tensor<T> _kernels = new([0, 0, 0, 0, 0]);

    [TrainableParameter(Role = PersistentTensorRole.Biases, Shape = "_outputChannels")]
    private Tensor<T> _biases = new([0]);

    /// <summary>Gets the input channel count.</summary>
    public int InputChannels => _inputChannels;

    /// <summary>Gets the output channel count.</summary>
    public int OutputChannels => _outputChannels;

    /// <summary>Gets the temporal kernel extent.</summary>
    public int KernelDepth => _kernelDepth;

    /// <summary>Gets the spatial kernel height.</summary>
    public int KernelHeight => _kernelHeight;

    /// <summary>Gets the spatial kernel width.</summary>
    public int KernelWidth => _kernelWidth;

    /// <summary>Gets whether the layer is initialized as an exact zero projection.</summary>
    public bool IsZeroInitialized => _zeroInitialize;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>Creates an anisotropic video convolution.</summary>
    public TemporalConv3DLayer(
        int inputChannels,
        int outputChannels,
        int kernelDepth,
        int kernelHeight = 1,
        int kernelWidth = 1,
        int? paddingDepth = null,
        int? paddingHeight = null,
        int? paddingWidth = null,
        bool zeroInitialize = false)
        : base(
            [inputChannels, -1, -1, -1],
            [outputChannels, -1, -1, -1])
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelDepth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelDepth));
        if (kernelHeight <= 0) throw new ArgumentOutOfRangeException(nameof(kernelHeight));
        if (kernelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(kernelWidth));

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelDepth = kernelDepth;
        _kernelHeight = kernelHeight;
        _kernelWidth = kernelWidth;
        _paddingDepth = paddingDepth ?? (kernelDepth - 1) / 2;
        _paddingHeight = paddingHeight ?? (kernelHeight - 1) / 2;
        _paddingWidth = paddingWidth ?? (kernelWidth - 1) / 2;
        _zeroInitialize = zeroInitialize;

        if (_paddingDepth < 0) throw new ArgumentOutOfRangeException(nameof(paddingDepth));
        if (_paddingHeight < 0) throw new ArgumentOutOfRangeException(nameof(paddingHeight));
        if (_paddingWidth < 0) throw new ArgumentOutOfRangeException(nameof(paddingWidth));
    }

    /// <inheritdoc />
    protected override void OnFirstForward(Tensor<T> input)
    {
        if (input.Rank != 5)
            throw new ArgumentException(
                $"TemporalConv3DLayer requires [B,C,F,H,W], got rank {input.Rank}.", nameof(input));
        if (input.Shape[1] != _inputChannels)
            throw new ArgumentException(
                $"Expected {_inputChannels} input channels, got {input.Shape[1]}.", nameof(input));

        MaterializeParameters();
        int outF = input.Shape[2] + 2 * _paddingDepth - _kernelDepth + 1;
        int outH = input.Shape[3] + 2 * _paddingHeight - _kernelHeight + 1;
        int outW = input.Shape[4] + 2 * _paddingWidth - _kernelWidth + 1;
        if (outF <= 0 || outH <= 0 || outW <= 0)
            throw new ArgumentException("Kernel and padding produce an empty video output.", nameof(input));
        ResolveShapes(
            [_inputChannels, input.Shape[2], input.Shape[3], input.Shape[4]],
            [_outputChannels, outF, outH, outW]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        var convolved = Engine.Conv3D(
            input,
            _kernels,
            [1, 1, 1],
            [_paddingDepth, _paddingHeight, _paddingWidth],
            [1, 1, 1]);
        var bias = Engine.Reshape(_biases, [1, _outputChannels, 1, 1, 1]);
        return Engine.TensorBroadcastAdd(convolved, bias);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        var gradients = GetParameterGradients();
        if (gradients.Length != ParameterCount || _kernels.Length == 0) return;
        int index = 0;
        for (int i = 0; i < _kernels.Length; i++, index++)
            _kernels[i] = NumOps.Subtract(_kernels[i], NumOps.Multiply(learningRate, gradients[index]));
        for (int i = 0; i < _biases.Length; i++, index++)
            _biases[i] = NumOps.Subtract(_biases[i], NumOps.Multiply(learningRate, gradients[index]));
        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
    }

    /// <inheritdoc />
    public override void ResetState()
    {
    }

    /// <inheritdoc />
    public override LayerBase<T> Clone()
    {
        var clone = new TemporalConv3DLayer<T>(
            _inputChannels, _outputChannels,
            _kernelDepth, _kernelHeight, _kernelWidth,
            _paddingDepth, _paddingHeight, _paddingWidth,
            _zeroInitialize);
        if (_kernels.Length > 0) clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = _inputChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["OutputChannels"] = _outputChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KernelDepth"] = _kernelDepth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KernelHeight"] = _kernelHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["KernelWidth"] = _kernelWidth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PaddingDepth"] = _paddingDepth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PaddingHeight"] = _paddingHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["PaddingWidth"] = _paddingWidth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["ZeroInitialize"] = _zeroInitialize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    protected override void EnsureParametersMaterialized()
    {
        if (_kernels.Length > 0) return;

        int[] kernelShape =
            [_outputChannels, _inputChannels, _kernelDepth, _kernelHeight, _kernelWidth];
        _kernels = AllocateLazyWeight(kernelShape);
        _biases = AllocateLazyWeight([_outputChannels]);
        if (_zeroInitialize)
        {
            Engine.TensorFill(_kernels, NumOps.Zero);
            Engine.TensorFill(_biases, NumOps.Zero);
        }
        else
        {
            int fanIn = _inputChannels * _kernelDepth * _kernelHeight * _kernelWidth;
            T bound = NumOps.FromDouble(1.0 / System.Math.Sqrt(fanIn));
            var initialized = Engine.TensorRandomUniformRange<T>(
                kernelShape, NumOps.Negate(bound), bound);
            initialized.AsSpan().CopyTo(_kernels.Data.Span);
            var initializedBias = Engine.TensorRandomUniformRange<T>(
                [_outputChannels], NumOps.Negate(bound), bound);
            initializedBias.AsSpan().CopyTo(_biases.Data.Span);
        }

        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }
}
