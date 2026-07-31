using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// 1D convolutional layer for sequence / waveform data, with optional
/// dilation. Operates on rank-3 input <c>[B, C_in, T]</c> and produces
/// rank-3 output <c>[B, C_out, T_out]</c>, where
/// <c>T_out = (T + 2*padding - dilation*(kernelSize - 1) - 1) / stride + 1</c>
/// per the standard 1D convolution formula (PyTorch <c>nn.Conv1d</c>
/// convention).
/// </summary>
/// <remarks>
/// <para>
/// Implemented by delegating to <c>Engine.Conv2D</c> with the time axis
/// expanded to a degenerate 2D layout — input <c>[B, C, T]</c> is
/// reshaped to <c>[B, C, 1, T]</c>, kernel shape is
/// <c>[C_out, C_in, 1, kernelSize]</c>, dilation is <c>(1, dilation)</c>,
/// and padding is <c>(0, padding)</c>. This avoids duplicating the conv
/// kernel inside the layer and keeps the tape autodiff path identical to
/// every other Conv layer in the codebase. The degenerate height axis is
/// reshaped away on return.
/// </para>
/// <para>
/// Used by <see cref="AiDotNet.Diffusion.Audio.DiffWaveModel{T}"/> for the
/// dilated convolution stack from Kong et al. 2020 "DiffWave" §3 — kernel
/// size 3, dilation <c>2^(i % dilation_cycle)</c>. Also valid as a 1×1
/// channel mixer (<c>kernelSize=1</c>) — the same shape used by DiffWave
/// for the input/skip/output projections.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 2, 8", TestConstructorArgs = "4, 3, 1, 1, 1, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
public partial class Conv1DLayer<T> : LayerBase<T>
{
    private int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;
    private readonly int _padding;
    private readonly int _dilation;

    private Tensor<T> _kernels;
    private Tensor<T> _biases;
    private int[]? _originalInputShape;

    /// <summary>
    /// Live parameter count. Returns the eventual <c>(C_out·C_in·K) + C_out</c>
    /// formula once <see cref="OnFirstForward"/> has resolved input
    /// channels; before that, falls back to <c>(C_out·1·K) + C_out</c>
    /// (assumes a 1-channel input until proven otherwise) so a
    /// freshly-constructed model still reports a non-zero
    /// <c>ParameterCount</c> for the
    /// <see cref="AiDotNet.Tests.ModelFamilyTests.Base.NeuralNetworkModelTestBase.Parameters_ShouldBeNonEmpty"/>
    /// invariant — without locking the lazy shape resolution to a wrong
    /// input channel count.
    /// </summary>
    public override long ParameterCount
    {
        get
        {
            int effectiveInputChannels = _inputChannels > 0 ? _inputChannels : 1;
            return ((long)_outputChannels * effectiveInputChannels * _kernelSize) + _outputChannels;
        }
    }

    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new 1D convolutional layer with lazy input-channel
    /// resolution. The kernel and bias tensors aren't allocated until
    /// <see cref="OnFirstForward"/> sees the first real input — mirrors
    /// PyTorch's lazy <c>nn.LazyConv1d</c> semantics so DiffWave-style
    /// models can be constructed without knowing audio rate at compile
    /// time.
    /// </summary>
    /// <param name="outputChannels">Number of output feature maps (<c>C_out</c>).</param>
    /// <param name="kernelSize">Kernel width along the time axis.</param>
    /// <param name="dilation">Dilation factor (Kong 2020 §3 uses <c>2^i</c>).</param>
    /// <param name="stride">Stride along the time axis. Defaults to 1.</param>
    /// <param name="padding">Zero padding along the time axis. Defaults to <c>(kernelSize-1)*dilation/2</c> for "same" output length when stride=1.</param>
    /// <param name="activation">Optional scalar activation.</param>
    /// <param name="initializationStrategy">Optional weight initialization (defaults to He).</param>
    public Conv1DLayer(
        int outputChannels,
        int kernelSize,
        int dilation = 1,
        int stride = 1,
        int? padding = null,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, -1 }, new[] { outputChannels, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = -1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        // Default "same" padding when stride=1 — preserves the input
        // sequence length, which is what DiffWave residual blocks rely on
        // (each block must return tensors with the same T as its input
        // for the residual add). Caller can override for downsampling.
        _padding = padding ?? ((kernelSize - 1) * dilation / 2);
        _dilation = dilation;

        _kernels = new Tensor<T>([0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
    }

    /// <summary>
    /// Eager-init constructor — pre-allocates kernel and bias tensors at
    /// construction time when the input channel count is known up-front
    /// (DiffWave / WaveNet style architectures with fixed per-block
    /// channel counts). Avoids the lazy-init disagreement between
    /// <see cref="ParameterCount"/> (placeholder estimate) and
    /// <see cref="GetParameters"/> (empty until first Forward) that
    /// breaks Clone-via-SetParameters round-trips.
    /// </summary>
    public Conv1DLayer(
        [LayerState] int inputChannels,
        [LayerState] int outputChannels,
        [LayerState] int kernelSize,
        [LayerState] int dilation = 1,
        [LayerState] int stride = 1,
        // MUST be persisted. Padding is not recoverable from any saved tensor: the kernel shape
        // carries outputChannels/inputChannels/kernelSize, but padding only shifts WHERE the
        // kernel lands, so a rebuild that omits it produces a layer with byte-identical weights
        // that computes a different convolution. Left unmarked, this parameter was merely
        // "optional" to the LayerStateGenerator, which silently rebuilt every Conv1DLayer with
        // padding = null => (kernelSize-1)*dilation/2. MusicSourceSeparator's Demucs encoder
        // passes kernel 8 / stride 4 / padding 2 (chosen so encoder and decoder lengths stay
        // aligned for the skip-add), and the fallback yields 3 — with L=64 both give output
        // length 16, so the mismatch cleared every shape assertion and surfaced only as clone
        // output diverging by ~2.7e-01 with provably identical parameters.
        // LayerStateGenerator strips Nullable<T> to match the int _padding field, so what
        // round-trips is the EFFECTIVE padding, which reproduces either spelling exactly.
        [LayerState(Key = "Padding")] int? padding = null,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { inputChannels, -1 }, new[] { outputChannels, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (padding.HasValue && padding.Value < 0) throw new ArgumentOutOfRangeException(nameof(padding));

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = inputChannels;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;
        _padding = padding ?? ((kernelSize - 1) * dilation / 2);
        _dilation = dilation;

        _kernels = AllocateLazyWeight([outputChannels, inputChannels, 1, kernelSize]);
        _biases = AllocateLazyWeight([outputChannels]);
        InitializeLayerWeights(_kernels, inputChannels * kernelSize, outputChannels);
        InitializeLayerBiases(_biases);
        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);

        // Resolve against a placeholder T = required minimum for the dilated kernel to fit; the
        // real T is bound on first Forward via EnsureInitializedFromInput and doesn't change the
        // parameter count (Conv1D is translation-invariant in T). The OUTPUT length must stay
        // dynamic: publishing the placeholder's length made the layer advertise
        // [outputChannels, 5] and then produce [outputChannels, 32].
        int minTime = _dilation * (_kernelSize - 1) + 1;
        ResolveShapes(new[] { inputChannels, minTime }, new[] { outputChannels, LayerShape.Dynamic });
    }

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank != 3)
        {
            throw new ArgumentException(
                $"Conv1DLayer requires rank-3 [B, C, T] input; got rank {rank}.",
                nameof(input));
        }

        int cIn = input.Shape[1];
        int tIn = input.Shape[2];
        int tOut = (tIn + 2 * _padding - _dilation * (_kernelSize - 1) - 1) / _stride + 1;

        _inputChannels = cIn;

        // Idempotent weight allocation. If this layer ALREADY holds correctly-shaped weights — because
        // they were installed by SetParameters / SetTrainableParameters (deserialize) or shared by a
        // copy-on-write clone before the first forward — do NOT re-allocate and re-initialize them.
        // Re-initializing here would overwrite restored/shared TRAINED weights with fresh random ones
        // the first time a cloned or deserialized model runs a forward, silently dropping the training
        // (the #1221 Clone_AfterTraining failure class for lazy conv/attention stacks). Only allocate
        // when the weights are missing or their input-channel count no longer matches the input.
        bool weightsAlreadyValid = _kernels is { Rank: 4 } k
            && k.Shape[0] == _outputChannels && k.Shape[1] == cIn && k.Shape[3] == _kernelSize
            && _biases is { } b && b.Length == _outputChannels;

        if (!weightsAlreadyValid)
        {
            _kernels = AllocateLazyWeight([_outputChannels, cIn, 1, _kernelSize]);
            _biases = AllocateLazyWeight([_outputChannels]);
            InitializeLayerWeights(_kernels, cIn * _kernelSize, _outputChannels);
            InitializeLayerBiases(_biases);
            RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
        }

        // Apply any parameters handed to SetParameters before the shape was known. This is the second
        // half of the deferral: geometry is now resolved from the REAL input, so the restored weights
        // land on exactly the tensors the original had, and a clone reproduces the original bit-for-bit.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyResolvedParameters(pending);
        }

        // Same reasoning as the lazy constructor: the output length follows the input length, so
        // it is not part of this layer's contract and must not be frozen into the declaration.
        _ = tOut;
        ResolveShapes(new[] { cIn, tIn }, new[] { _outputChannels, LayerShape.Dynamic });
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        _originalInputShape = input._shape;

        // Reshape [B, C, T] -> [B, C, 1, T] for the Engine.Conv2D call.
        // The kernel is already [C_out, C_in, 1, K] from OnFirstForward,
        // so Conv2D produces [B, C_out, 1, T_out]; we strip the H=1 axis
        // on return.
        var input4D = Engine.Reshape(input,
            new[] { input.Shape[0], input.Shape[1], 1, input.Shape[2] });

        var conv = Engine.Conv2D(
            input4D, _kernels,
            new[] { 1, _stride },
            new[] { 0, _padding },
            new[] { 1, _dilation });

        // Broadcast bias [C_out] -> [1, C_out, 1, 1] and add.
        var biasReshaped = Engine.Reshape(_biases, new[] { 1, _outputChannels, 1, 1 });
        var withBias = Engine.TensorBroadcastAdd(conv, biasReshaped);
        var activated = ApplyActivation(withBias);

        // [B, C_out, 1, T_out] -> [B, C_out, T_out]
        return Engine.Reshape(activated,
            new[] { activated.Shape[0], activated.Shape[1], activated.Shape[3] });
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Tape-based autodiff drives parameter updates through the
        // engine's optimizer integration; manual UpdateParameters is a
        // legacy hook kept only for API completeness. No-op here — the
        // tape's Backward pass already accumulated and applied gradients
        // to _kernels / _biases via the registered trainable parameters.
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        if (!IsShapeResolved)
        {
            // Hand back whatever SetParameters deferred, so a save -> load -> save round trip is
            // LOSSLESS even when no forward has run in between. Returning an empty vector here would
            // silently drop restored weights for a model that is cloned twice before use. PyTorch has no
            // equivalent: state_dict() on a lazy module reports uninitialized parameters.
            if (_pendingParameters is not null) return _pendingParameters.Clone();

            // Caller asked for parameters before first Forward — return
            // an empty vector that round-trips with SetParameters'
            // pre-resolved branch below. This matches DenseLayer's
            // pre-init contract.
            return new Vector<T>(0);
        }
        return Vector<T>.Concatenate(
            new Vector<T>(_kernels.ToArray()),
            new Vector<T>(_biases.ToArray()));
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // Infer input channels from the parameter layout when the layer
        // hasn't seen a Forward yet — needed for Clone() paths that
        // SetParameters(GetParameters()) on a fresh clone before
        // PredictNoise has run. (C_out * C_in * K) + C_out = params.Length,
        // solve for C_in.
        if (!IsShapeResolved)
        {
            // DEFER instead of guessing. Inferring inputChannels here also forces a resolution, and the
            // only length available is a PLACEHOLDER (MinValidInputLength()) rather than the shape the
            // original layer actually resolved against. The weights then land correctly — a restored
            // model's flat parameter vector compares bit-identical — while the layer computes a
            // different function, measured on a MusicSourceSeparator clone as Encoder_0 diverging by
            // 7.93e-01 on the FIRST layer with 0 of 5892 parameters differing.
            //
            // Holding the parameters until the first real Forward keeps the layer fully lazy and
            // resolves geometry from the ACTUAL input, so a restore reproduces the original exactly.
            // PyTorch cannot do this at all: nn.Conv1d requires in_channels up front and
            // load_state_dict refuses a lazy module until a forward has run.
            _pendingParameters = parameters.Clone();
            return;
        }

        ApplyResolvedParameters(parameters);
    }

    /// <summary>Parameters handed to <see cref="SetParameters"/> before the shape was known.</summary>
    private Vector<T>? _pendingParameters;

    /// <summary>Applies a parameter vector to the already-resolved kernel and bias tensors.</summary>
    private void ApplyResolvedParameters(Vector<T> parameters)
    {
        int expectedLength = _kernels.Length + _biases.Length;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Expected {expectedLength} parameters, but got {parameters.Length}");
        }

        // Copy in place via Data.Span to preserve the persistent-tensor identities
        // registered above (RegisterTrainableParameter at lines 284-285). Replacing
        // _kernels/_biases with `new Tensor<T>(...)` would leave the registry
        // pointing at the old tensors — Forward would use the old (un-loaded)
        // weights while the new ones sat unreferenced — so a Clone restored via
        // SetParameters would silently lose the loaded values on the next step.
        // Same pattern DenseLayer.SetParameters uses (DenseLayer.cs:1379-1385).
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
    /// Returns layer-specific metadata for serialization. The convolution
    /// hyper-parameters (channel counts, kernel width, dilation, stride,
    /// padding) are NOT recoverable from the input/output shapes alone, so they
    /// must round-trip through metadata for <c>CreateLayerFromType</c> to
    /// reconstruct an identically-shaped layer on Clone/Deserialize.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["OutputChannels"] = _outputChannels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["Dilation"] = _dilation.ToString();
        metadata["Stride"] = _stride.ToString();
        metadata["Padding"] = _padding.ToString();
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
