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
public partial class Conv1DTransposeLayer<T> : LayerBase<T>
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

    /// <summary>
    /// Live parameter count: <c>(C_in·C_out·K) + C_out</c> once input channels are
    /// resolved; before that, falls back to a 1-input-channel estimate so a
    /// freshly-constructed model still reports a non-zero <c>ParameterCount</c>.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            int effectiveInputChannels = _inputChannels > 0 ? _inputChannels : 1;
            return ((long)effectiveInputChannels * _outputChannels * _kernelSize) + _outputChannels;
        }
    }

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
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (outputPadding < 0) throw new ArgumentOutOfRangeException(nameof(outputPadding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = -1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        // HiFi-GAN convention: padding = (kernel - stride) / 2 keeps T_out = T * stride.
        _padding = padding ?? ((kernelSize - stride) / 2);
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
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int? padding = null,
        int outputPadding = 0,
        int dilation = 1,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { inputChannels, -1 }, new[] { outputChannels, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        if (outputPadding < 0) throw new ArgumentOutOfRangeException(nameof(outputPadding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding ?? ((kernelSize - stride) / 2);
        _outputPadding = outputPadding;
        _dilation = dilation;

        // Transposed-conv weight layout is [C_in, C_out, 1, K] (input channels first).
        _kernels = AllocateLazyWeight([inputChannels, outputChannels, 1, kernelSize]);
        _biases = AllocateLazyWeight([outputChannels]);
        InitializeLayerWeights(_kernels, inputChannels * kernelSize, outputChannels);
        InitializeLayerBiases(_biases);
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        int minTime = 1;
        int outTime = ComputeOutputLength(minTime);
        ResolveShapes(new[] { inputChannels, minTime }, new[] { outputChannels, outTime });
    }

    /// <summary>PyTorch <c>nn.ConvTranspose1d</c> output-length formula.</summary>
    private int ComputeOutputLength(int tIn)
        => (tIn - 1) * _stride - 2 * _padding + _dilation * (_kernelSize - 1) + _outputPadding + 1;

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
        _kernels = AllocateLazyWeight([cIn, _outputChannels, 1, _kernelSize]);
        _biases = AllocateLazyWeight([_outputChannels]);
        InitializeLayerWeights(_kernels, cIn * _kernelSize, _outputChannels);
        InitializeLayerBiases(_biases);
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        ResolveShapes(new[] { cIn, tIn }, new[] { _outputChannels, tOut });
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
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
        var withBias = Engine.TensorBroadcastAdd(deconv, biasReshaped);
        var activated = ApplyActivation(withBias);

        // [B, C_out, 1, T_out] -> [B, C_out, T_out]
        return Engine.Reshape(activated,
            new[] { activated.Shape[0], activated.Shape[1], activated.Shape[3] });
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Tape autodiff drives updates through the registered trainable parameters;
        // this manual hook is a no-op (parity with Conv1DLayer / DeconvolutionalLayer).
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        if (!IsShapeResolved)
        {
            return new Vector<T>(0);
        }
        return Vector<T>.Concatenate(
            new Vector<T>(_kernels.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (!IsShapeResolved)
        {
            // Layout: kernels [C_in, C_out, 1, K] + biases [C_out]. Solve for C_in.
            int candidateInputChannels = (parameters.Length - _outputChannels) /
                                         (_outputChannels * _kernelSize);
            if (candidateInputChannels <= 0
                || candidateInputChannels * _outputChannels * _kernelSize + _outputChannels != parameters.Length)
            {
                throw new ArgumentException(
                    $"Cannot infer inputChannels for Conv1DTransposeLayer from {parameters.Length} parameters " +
                    $"(outputChannels={_outputChannels}, kernelSize={_kernelSize}).");
            }
            _inputChannels = candidateInputChannels;
            ResolveFromShape(new[] { candidateInputChannels, 1 });
            _kernels = AllocateLazyWeight([candidateInputChannels, _outputChannels, 1, _kernelSize]);
            _biases = AllocateLazyWeight([_outputChannels]);
            RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
        }

        int expectedLength = _kernels.Length + _biases.Length;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        // In-place copy preserves the persistent-tensor identities registered above
        // (same pattern as Conv1DLayer.SetParameters).
        parameters.AsSpan().Slice(0, _kernels.Length).CopyTo(_kernels.Data.Span);
        parameters.AsSpan().Slice(_kernels.Length, _biases.Length).CopyTo(_biases.Data.Span);

        Engine.InvalidatePersistentTensor(_kernels);
        Engine.InvalidatePersistentTensor(_biases);
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
