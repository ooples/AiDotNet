using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// 1D transposed convolution ("deconvolution") for sequence / waveform data —
/// the learnable temporal-upsampling primitive used by HiFi-GAN (Kong et al.
/// 2020) and the GAN-vocoder family. Operates on rank-3 input
/// <c>[B, C_in, T]</c> and produces rank-3 output <c>[B, C_out, T_out]</c>
/// where, matching PyTorch <c>nn.ConvTranspose1d</c> exactly:
/// <code>
/// T_out = (T - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// PyTorch parity: the weight layout is <c>[C_in, C_out, kernelSize]</c> (the
/// transposed-convolution convention — input channels first, opposite of the
/// forward <see cref="Conv1DLayer{T}"/>'s <c>[C_out, C_in, K]</c>), and the
/// <c>T_out</c> formula above is bit-identical to <c>nn.ConvTranspose1d</c>.
/// </para>
/// <para>
/// Implemented by delegating to <c>Engine.ConvTranspose2D</c> with the time axis
/// expanded to a degenerate 2D layout — input <c>[B, C, T]</c> is reshaped to
/// <c>[B, C, 1, T]</c>, kernel shape is <c>[C_in, C_out, 1, kernelSize]</c>,
/// stride is <c>(1, stride)</c>, padding <c>(0, padding)</c>, output padding
/// <c>(0, outputPadding)</c>. This reuses the engine's transposed-conv kernel
/// (including the fused GPU path) and keeps the tape autodiff backward identical
/// to <see cref="DeconvolutionalLayer{T}"/> — no hand-written backward needed.
/// We exceed the stock PyTorch op by routing through the engine's fused
/// conv-transpose + bias (+ activation) kernel when available.
/// </para>
/// <para>
/// Used by <c>LayerHelper.CreateDefaultHiFiGANLayers</c>: each upsample stage is a
/// <c>ConvTranspose1d(ch, ch/2, kernel=2*rate, stride=rate, padding=rate/2)</c>
/// matching the official <c>jik876/hifi-gan</c> generator
/// (<c>upsample_rates=[8,8,2,2]</c>, <c>upsample_kernel_sizes=[16,16,4,4]</c>).
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "4, 2, 4, 2, 0, 1, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
// Roles come from this layer's own guard in OnFirstForward - "requires rank-3 [B, C, T] input" - so
// rank 3 is the ONLY declared form and Batch is NOT optional: a rank-2 input is rejected outright,
// and BatchOptional would advertise a form the layer throws on.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Time,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class Conv1DTransposeLayer<T> : LayerBase<T>, IShapeContract
{
    private int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _outputPadding;
    private readonly int _dilation;

    private Tensor<T> _kernels;
    private Tensor<T> _biases;
    private int[]? _originalInputShape;

    /// <inheritdoc />
    /// <remarks>
    /// Paired with <see cref="ParameterCount"/> and <see cref="GetParameters"/>, which both report
    /// nothing until the input channel count arrives. Without this the layer said "I have no
    /// parameters" AND "nothing is pending" at once -- both false, since it certainly gains weights
    /// on the first forward. The model-family non-empty invariant accepts either a positive count
    /// or a pending flag, so MusicSourceSeparator failed it the moment the count became honest.
    /// </remarks>
    public override bool HasUninitializedParameters => !IsShapeResolved;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Lazy-input-channel constructor (mirrors PyTorch's lazy conv semantics). The
    /// kernel/bias tensors are allocated on the first <see cref="Forward"/>.
    /// </summary>
    /// <param name="outputChannels">Number of output feature maps (<c>C_out</c>).</param>
    /// <param name="kernelSize">Kernel width along the time axis.</param>
    /// <param name="stride">Upsampling stride along the time axis (the temporal expansion factor). Defaults to 1.</param>
    /// <param name="padding">Zero padding subtracted from each end of the output. Defaults to <c>(kernelSize - stride) / 2</c> (the HiFi-GAN convention that keeps <c>T_out ≈ T·stride</c>).</param>
    /// <param name="outputPadding">Extra size added to one side of the output to disambiguate the stride's fractional output length. Defaults to 0.</param>
    /// <param name="dilation">Dilation factor. Defaults to 1 (HiFi-GAN upsampling uses 1).</param>
    /// <param name="activation">Optional scalar activation.</param>
    /// <param name="initializationStrategy">Optional weight initialization (defaults to He).</param>
    public Conv1DTransposeLayer(
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int? padding = null,
        int outputPadding = 0,
        int dilation = 1,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, -1 }, new[] { outputChannels, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        // The engine's transposed-conv kernel (Engine.ConvTranspose2D) takes no
        // dilation argument and does not dilate, so honouring dilation > 1 here is
        // impossible — reject it at the boundary rather than silently ignore it.
        // HiFi-GAN upsampling is always dilation=1; the dilated convolutions in the
        // MRF use the FORWARD Conv1DLayer (which does support dilation).
        if (dilation != 1) throw new ArgumentOutOfRangeException(nameof(dilation),
            "Conv1DTransposeLayer supports only dilation == 1; use Conv1DLayer for dilated (non-transposed) convolutions.");
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (outputPadding < 0) throw new ArgumentOutOfRangeException(nameof(outputPadding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = -1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        // HiFi-GAN convention: padding = (kernel - stride) / 2 keeps T_out = T * stride.
        // Clamp to 0: when stride > kernelSize the symmetric formula goes negative,
        // which is not a valid padding (PyTorch rejects it) — 0 is the only sane default.
        _padding = padding ?? System.Math.Max(0, (kernelSize - stride) / 2);
        _outputPadding = outputPadding;
        _dilation = dilation;

        _kernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
    }

    /// <summary>
    /// Eager-init constructor — pre-allocates kernel/bias at construction when the
    /// input channel count is known up-front (the HiFi-GAN generator stack has
    /// fixed per-stage channel counts), so <see cref="ParameterCount"/> and
    /// <see cref="GetParameters"/> agree before the first Forward (Clone round-trip).
    /// </summary>
    public Conv1DTransposeLayer(
        [LayerState] int inputChannels,
        [LayerState] int outputChannels,
        [LayerState] int kernelSize,
        [LayerState] int stride = 1,
        [LayerState] int? padding = null,
        [LayerState] int outputPadding = 0,
        [LayerState] int dilation = 1,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { inputChannels, -1 }, new[] { outputChannels, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        // See lazy ctor: the engine's transposed-conv path does not dilate.
        if (dilation != 1) throw new ArgumentOutOfRangeException(nameof(dilation),
            "Conv1DTransposeLayer supports only dilation == 1; use Conv1DLayer for dilated (non-transposed) convolutions.");
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (outputPadding < 0) throw new ArgumentOutOfRangeException(nameof(outputPadding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        // Clamp the symmetric default to 0 (negative padding is invalid; see lazy ctor).
        _padding = padding ?? System.Math.Max(0, (kernelSize - stride) / 2);
        _outputPadding = outputPadding;
        _dilation = dilation;

        // Transposed-conv weight layout is [C_in, C_out, 1, K] (input channels first).
        _kernels = AllocateLazyWeight([inputChannels, outputChannels, 1, kernelSize]);
        _biases = AllocateLazyWeight([outputChannels]);
        InitializeLayerWeights(_kernels, inputChannels * kernelSize, outputChannels);
        InitializeLayerBiases(_biases);
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        // The time axis is a function of the input length, so it is not known here. Declaring
        // ComputeOutputLength(MinValidInputLength()) -- the shortest input this configuration
        // accepts -- published that placeholder as the layer's contract: the layer advertised
        // [outputChannels, 8] and then produced [outputChannels, 128] for real audio. Only the
        // channel count is fixed at construction; the length stays dynamic until a forward runs.
        int minTime = MinValidInputLength();
        ResolveShapes(new[] { inputChannels, minTime }, new[] { outputChannels, LayerShape.Dynamic });
    }

    /// <summary>PyTorch <c>nn.ConvTranspose1d</c> output-length formula.</summary>
    private int ComputeOutputLength(int tIn)
        => (tIn - 1) * _stride - 2 * _padding + _dilation * (_kernelSize - 1) + _outputPadding + 1;

    /// <summary>
    /// Smallest input length whose transposed-convolution output is still at least one frame.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Placeholder shapes must not be built from a hard-coded <c>tIn = 1</c>. Transposed convolution
    /// SUBTRACTS <c>2·padding</c>, so whenever the padding exceeds the kernel's reach the length
    /// formula goes non-positive: with this layer's own test configuration (kernelSize 2, stride 4,
    /// padding 2) <c>ComputeOutputLength(1)</c> is -2, which resolved an invalid output shape and
    /// surfaced as "Resolved output shape still contains a -1 placeholder". Solve
    /// <c>(tIn-1)·stride ≥ 2·padding - dilation·(K-1) - outputPadding</c> for the smallest valid
    /// <c>tIn</c> instead.
    /// </para>
    /// </remarks>
    private int MinValidInputLength()
    {
        int deficit = 2 * _padding - _dilation * (_kernelSize - 1) - _outputPadding;
        if (deficit <= 0) return 1;
        return 1 + (deficit + _stride - 1) / _stride;
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Hand-written because the time axis follows <see cref="ComputeOutputLength"/>, which is the
    /// INVERSE of a sliding window: <c>T_out = (T-1)*stride - 2*padding + dilation*(K-1) +
    /// outputPadding + 1</c>. <c>Window</c> models the forward direction only, so it cannot express
    /// this - a transposed convolution GROWS its axis by roughly <c>stride</c>, and reading the
    /// window formula backwards would understate the output by that whole factor.
    /// </para>
    /// <para>
    /// Rewriting the formula as <c>T_out = T*stride + C</c> with
    /// <c>C = dilation*(K-1) + outputPadding + 1 - stride - 2*padding</c> shows the one case that IS
    /// in the vocabulary: when <c>C</c> is zero the layer is exactly <c>Scaled(Time, stride)</c>.
    /// That is not an edge case - it is the configuration HiFi-GAN uses
    /// (<c>kernel = 2*rate, stride = rate, padding = rate/2</c>), so the vocoder stacks this
    /// annotation is written for resolve their lengths precisely rather than stopping at Unknown.
    /// When <c>C</c> is non-zero the length is a genuine affine offset and no relation carries it,
    /// so it is declared Unknown WITH the offset in the reason rather than approximated.
    /// </para>
    /// <para>
    /// The channel count is <c>Fixed</c> from <c>_outputChannels</c> - that is the one output axis
    /// <c>OnFirstForward</c> pins: <c>ResolveShapes(new[] { cIn, tIn }, new[] { _outputChannels,
    /// LayerShape.Dynamic })</c>. The <c>Dynamic</c> on the second axis there is the same statement
    /// this contract makes, expressed as a relation instead of a placeholder.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3) return null;

        int offset = _dilation * (_kernelSize - 1) + _outputPadding + 1 - _stride - 2 * _padding;
        var time = offset == 0
            ? AxisRelation.Scaled(TensorAxis.Time, _stride)
            : AxisRelation.Unknown(
                $"Transposed-convolution length is T*{_stride} + {offset}; an affine offset on a "
                + "scaled axis is not in the relation vocabulary.");

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outputChannels)),
            new OutputAxisContract(TensorAxis.Time, time),
        };
    }

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank != 3)
        {
            throw new ArgumentException(
                $"Conv1DTransposeLayer requires rank-3 [B, C, T] input; got rank {rank}.",
                nameof(input));
        }

        int cIn = input.Shape[1];
        int tIn = input.Shape[2];
        int tOut = ComputeOutputLength(tIn);

        _inputChannels = cIn;
        // Idempotent: don't re-init weights a clone/deserialize already installed (#1221). See Conv1DLayer.
        if (!WeightsAlreadyAllocated(_kernels, cIn, _outputChannels, 1, _kernelSize))
        {
            _kernels = AllocateLazyWeight([cIn, _outputChannels, 1, _kernelSize]);
            _biases = AllocateLazyWeight([_outputChannels]);
            InitializeLayerWeights(_kernels, cIn * _kernelSize, _outputChannels);
            InitializeLayerBiases(_biases);
            RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
        }

        // tOut describes THIS input, not the layer. Publishing it means the value is saved and
        // restored, and the restored layer then disagrees with the next input of a different
        // length. The channel count is the only output axis this layer fixes.
        _ = tOut;
        ResolveShapes(new[] { cIn, tIn }, new[] { _outputChannels, LayerShape.Dynamic });
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _originalInputShape = input._shape;

        // [B, C, T] -> [B, C, 1, T] for the degenerate-2D transposed conv. Kernel is
        // [C_in, C_out, 1, K]; ConvTranspose2D yields [B, C_out, 1, T_out].
        var input4D = Engine.Reshape(input,
            new[] { input.Shape[0], input.Shape[1], 1, input.Shape[2] });

        var deconv = Engine.ConvTranspose2D(
            input4D, _kernels,
            new[] { 1, _stride },
            new[] { 0, _padding },
            new[] { 0, _outputPadding });

        var biasReshaped = Engine.Reshape(_biases, new[] { 1, _outputChannels, 1, 1 });
        var withBias = Engine.TensorAdd(deconv, biasReshaped);
        var activated = ApplyActivation(withBias);

        // [B, C_out, 1, T_out] -> [B, C_out, T_out]
        return Engine.Reshape(activated,
            new[] { activated.Shape[0], activated.Shape[1], activated.Shape[3] });
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _originalInputShape = null;
    }

    /// <summary>
    /// Serialization metadata — the transposed-conv hyper-parameters aren't
    /// recoverable from input/output shapes, so they round-trip here for
    /// <c>CreateLayerFromType</c> to rebuild an identically-shaped layer on
    /// Clone/Deserialize.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["OutputChannels"] = _outputChannels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["Stride"] = _stride.ToString();
        metadata["Padding"] = _padding.ToString();
        metadata["OutputPadding"] = _outputPadding.ToString();
        metadata["Dilation"] = _dilation.ToString();
        if (_inputChannels > 0)
            metadata["InputChannels"] = _inputChannels.ToString();
        if (ScalarActivation is not null)
        {
            metadata["ScalarActivationType"] = ScalarActivation.GetType().AssemblyQualifiedName
                ?? ScalarActivation.GetType().FullName ?? string.Empty;
        }
        return metadata;
    }
}
