using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements an Inverted Residual Block (MBConv) used in MobileNetV2 and MobileNetV3.
/// </summary>
/// <remarks>
/// <para>
/// The Inverted Residual Block is the core building block for MobileNet architectures.
/// Unlike traditional residual blocks that narrow then widen, inverted residual blocks
/// expand then narrow, hence the name "inverted."
/// </para>
/// <para>
/// Architecture:
/// <code>
/// Input ────────────────────────────────────────────────────────────────┐
///   │                                                                   │
///   └─► [Expand 1x1 ─► BN ─► Act] ─► DW 3x3 ─► BN ─► Act ─► [SE] ─►    │
///                                                              │        │
///                                           Project 1x1 ─► BN ─┘─► (+) ─► Output
///                                                                    ↑
///                                                             (if skip connection)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The Inverted Residual Block is designed for efficient mobile inference.
///
/// Key innovations:
/// - Expansion: First EXPANDS the channels (opposite of traditional bottlenecks)
/// - Depthwise separable convolution: Filters each channel independently, then mixes
/// - Linear bottleneck: The final projection has NO activation (preserves information)
/// - Skip connection: Only when input and output dimensions match
///
/// This design reduces computational cost while maintaining model accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "8")]
public class InvertedResidualBlock<T> : LayerBase<T>, ILayerSerializationExtras<T>
{
    // Non-readonly: lazy ctor leaves these null until OnFirstForward
    // observes input.Shape and allocates each against the resolved
    // hiddenDim = inChannels × expansionRatio.
    private ConvolutionalLayer<T>? _expandConv;
    private BatchNormalizationLayer<T>? _expandBn;
    private ConvolutionalLayer<T>? _dwConv;
    private BatchNormalizationLayer<T>? _dwBn;
    private SqueezeAndExcitationLayer<T>? _se;
    private ConvolutionalLayer<T>? _projectConv;
    private BatchNormalizationLayer<T>? _projectBn;

    // Buffered parameters for the pre-Forward Deserialize → SetParameters
    // path: sub-layers are still null then, so we stash the vector here
    // and replay it inside OnFirstForward once the channel-count-driven
    // layout is known and sub-layers exist.
    private Vector<T>? _pendingParameters;

    // Non-readonly: lazy ctor leaves _useResidual = false until
    // OnFirstForward observes input channel count.
    private bool _useResidual;
    private readonly bool _hasExpansion;
    private readonly bool _useSE;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastExpandOut;
    private Tensor<T>? _lastExpandBnOut;
    private Tensor<T>? _lastExpandActOut;
    private Tensor<T>? _lastDwOut;
    private Tensor<T>? _lastDwBnOut;
    private Tensor<T>? _lastDwActOut;
    private Tensor<T>? _lastSeOut;
    private Tensor<T>? _lastProjectOut;
    private Tensor<T>? _lastProjectBnOut;

    /// <summary>
    /// Sum of trainable parameters across all sub-layers. The non-optional
    /// <c>_dwConv</c>, <c>_dwBn</c>, <c>_projectConv</c>, <c>_projectBn</c>
    /// stay null until <see cref="OnFirstForward"/> resolves the input
    /// channel count and allocates them — null-guard each so this property
    /// returns 0 in the pre-Forward state instead of throwing.
    /// </summary>
    public override long ParameterCount =>
        (_expandConv?.ParameterCount ?? 0) + (_expandBn?.ParameterCount ?? 0) +
        (_dwConv?.ParameterCount ?? 0) + (_dwBn?.ParameterCount ?? 0) +
        (_se?.ParameterCount ?? 0) +
        (_projectConv?.ParameterCount ?? 0) + (_projectBn?.ParameterCount ?? 0);

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    public override Vector<T> GetParameterGradients()
    {
        // All sub-layers stay null until OnFirstForward resolves input
        // channel count; null-guard each to keep this query side-effect-
        // free in the pre-Forward state.
        var grads = new List<T>();
        if (_expandConv is not null) grads.AddRange(_expandConv.GetParameterGradients().ToArray());
        if (_expandBn is not null) grads.AddRange(_expandBn.GetParameterGradients().ToArray());
        if (_dwConv is not null) grads.AddRange(_dwConv.GetParameterGradients().ToArray());
        if (_dwBn is not null) grads.AddRange(_dwBn.GetParameterGradients().ToArray());
        if (_se is not null) grads.AddRange(_se.GetParameterGradients().ToArray());
        if (_projectConv is not null) grads.AddRange(_projectConv.GetParameterGradients().ToArray());
        if (_projectBn is not null) grads.AddRange(_projectBn.GetParameterGradients().ToArray());
        return new Vector<T>([.. grads]);
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _expandConv?.ClearGradients(); _expandBn?.ClearGradients();
        _dwConv?.ClearGradients(); _dwBn?.ClearGradients();
        _se?.ClearGradients();
        _projectConv?.ClearGradients(); _projectBn?.ClearGradients();
    }

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InChannels { get; private set; }

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutChannels { get; }

    /// <summary>
    /// Gets the expansion ratio.
    /// </summary>
    public int ExpansionRatio { get; }

    /// <summary>
    /// Gets the stride used in the depthwise convolution.
    /// </summary>
    public int Stride { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="InvertedResidualBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="outChannels">The number of output channels.</param>
    /// <param name="inputHeight">The height of the input feature map.</param>
    /// <param name="inputWidth">The width of the input feature map.</param>
    /// <param name="expansionRatio">The expansion ratio for the hidden layer (default: 6).</param>
    /// <param name="stride">The stride for the depthwise convolution (default: 1).</param>
    /// <param name="useSE">Whether to use Squeeze-and-Excitation (for MobileNetV3).</param>
    /// <param name="seRatio">The reduction ratio for SE block (default: 4).</param>
    /// <param name="activation">The activation function to use. Default is ReLU6.</param>
    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>); only
    /// <c>outChannels</c>, <c>expansionRatio</c>, <c>stride</c>, and SE
    /// configuration are required at construction. The expansion conv's
    /// <c>hiddenDim = inChannels × expansionRatio</c> isn't known until
    /// input.Shape is observed, so the expansion / depthwise / SE
    /// sub-layers are allocated in <see cref="OnFirstForward"/>.
    /// </summary>
    public InvertedResidualBlock(
        int outChannels,
        int expansionRatio = 6,
        int stride = 1,
        bool useSE = false,
        int seRatio = 4,
        IActivationFunction<T>? activationFunction = null)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [outChannels, -1, -1],
            scalarActivation: activationFunction ?? new ReLU6Activation<T>())
    {
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (expansionRatio <= 0) throw new ArgumentOutOfRangeException(nameof(expansionRatio));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        InChannels = -1; // resolved in OnFirstForward
        OutChannels = outChannels;
        ExpansionRatio = expansionRatio;
        Stride = stride;
        _seRatio = seRatio;

        _hasExpansion = expansionRatio != 1;
        _useSE = useSE;

        // _expandConv, _expandBn, _dwConv, _dwBn, _se, _projectConv,
        // _projectBn are allocated in OnFirstForward — their channel
        // dimensions all derive from the resolved inChannels.
        _projectConv = null!;
        _projectBn = null!;
        _dwConv = null!;
        _dwBn = null!;
    }

    private readonly int _seRatio;

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves input channels + spatial dims, allocates expansion /
    /// depthwise / SE / projection sub-layers against the resolved
    /// <c>hiddenDim = inChannels × expansionRatio</c>, and propagates
    /// shape to each so ParameterCount reports the real weight count
    /// before any sub-layer's first Forward fires.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inChannels, inputHeight, inputWidth;
        if (s.Length == 3) { inChannels = s[0]; inputHeight = s[1]; inputWidth = s[2]; }
        else if (s.Length == 4) { inChannels = s[1]; inputHeight = s[2]; inputWidth = s[3]; }
        else
            throw new ArgumentException(
                $"InvertedResidualBlock requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        InChannels = inChannels;
        int hiddenDim = inChannels * ExpansionRatio;
        _useResidual = Stride == 1 && inChannels == OutChannels;

        // Expansion layer (1×1 conv) — only if expansion ratio > 1.
        if (_hasExpansion)
        {
            var expandConv = new ConvolutionalLayer<T>(
                outputDepth: hiddenDim,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
            var expandBn = new BatchNormalizationLayer<T>();
            _expandConv = expandConv;
            _expandBn = expandBn;
        }

        int dwInputChannels = _hasExpansion ? hiddenDim : inChannels;
        _dwConv = new ConvolutionalLayer<T>(
            outputDepth: dwInputChannels,
            kernelSize: 3,
            stride: Stride,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
        _dwBn = new BatchNormalizationLayer<T>();

        if (_useSE)
        {
            _se = new SqueezeAndExcitationLayer<T>(
                dwInputChannels,
                _seRatio,
                firstActivation: (IActivationFunction<T>?)null,
                secondActivation: (IActivationFunction<T>?)null);
        }

        _projectConv = new ConvolutionalLayer<T>(
            outputDepth: OutChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
        _projectBn = new BatchNormalizationLayer<T>();

        RegisterSubLayer(_dwConv);
        RegisterSubLayer(_dwBn);
        RegisterSubLayer(_projectConv);
        RegisterSubLayer(_projectBn);
        if (_expandConv is not null) RegisterSubLayer(_expandConv);
        if (_expandBn is not null) RegisterSubLayer(_expandBn);
        if (_se is not null) RegisterSubLayer(_se);

        // Propagate the parent's current training mode to the sub-layers
        // we just allocated — without this, a layer toggled to eval mode
        // before its first Forward sees fresh sub-layers in default
        // training mode and BN collapses to zero on batch=1 inputs.
        _expandConv?.SetTrainingMode(IsTrainingMode);
        _expandBn?.SetTrainingMode(IsTrainingMode);
        _dwConv.SetTrainingMode(IsTrainingMode);
        _dwBn.SetTrainingMode(IsTrainingMode);
        _se?.SetTrainingMode(IsTrainingMode);
        _projectConv.SetTrainingMode(IsTrainingMode);
        _projectBn.SetTrainingMode(IsTrainingMode);

        int dwOutH = (inputHeight + 2 - 3) / Stride + 1; // padding=1, kernel=3
        int dwOutW = (inputWidth + 2 - 3) / Stride + 1;

        // Use ResolveFromShape so weights are allocated up front — needed
        // for buffered Deserialize parameters to slice correctly.
        if (_hasExpansion)
        {
            _expandConv!.ResolveFromShape(new[] { inChannels, inputHeight, inputWidth });
            _expandBn!.ResolveFromShape(new[] { 1, hiddenDim, inputHeight, inputWidth });
        }
        _dwConv.ResolveFromShape(new[] { dwInputChannels, inputHeight, inputWidth });
        _dwBn.ResolveFromShape(new[] { 1, dwInputChannels, dwOutH, dwOutW });
        _projectConv.ResolveFromShape(new[] { dwInputChannels, dwOutH, dwOutW });
        _projectBn.ResolveFromShape(new[] { 1, OutChannels, dwOutH, dwOutW });

        ResolveShapes(
            new[] { inChannels, inputHeight, inputWidth },
            new[] { OutChannels, dwOutH, dwOutW });

        // Replay parameters that arrived via Deserialize → SetParameters
        // before sub-layers existed. Doing this AFTER ResolveShapes (which
        // also forces ResolveFromShape on the sub-layers above when called
        // through EnsureInitializedFromInput) means each sub-layer's
        // GetParameters().Length now reports the correct count.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyParameters(pending);
        }

        // Replay BN running-state extras buffered by
        // ILayerSerializationExtras.SetExtraParameters before _expandBn /
        // _dwBn / _projectBn were allocated. Without this, BN state for
        // a deserialized checkpoint stays at zero and inference diverges.
        if (_pendingExtraParameters is not null)
        {
            var pendingExtras = _pendingExtraParameters;
            _pendingExtraParameters = null;
            ApplyExtraParametersUnsafe(pendingExtras);
        }
    }

    private void ApplyParameters(Vector<T> parameters)
    {
        int offset = 0;
        if (_expandConv is not null)
        {
            int count = _expandConv.GetParameters().Length;
            _expandConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_expandBn is not null)
        {
            int count = _expandBn.GetParameters().Length;
            _expandBn.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_dwConv is not null)
        {
            int count = _dwConv.GetParameters().Length;
            _dwConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_dwBn is not null)
        {
            int count = _dwBn.GetParameters().Length;
            _dwBn.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_se is not null)
        {
            int count = _se.GetParameters().Length;
            _se.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_projectConv is not null)
        {
            int count = _projectConv.GetParameters().Length;
            _projectConv.SetParameters(parameters.SubVector(offset, count));
            offset += count;
        }
        if (_projectBn is not null)
        {
            int count = _projectBn.GetParameters().Length;
            _projectBn.SetParameters(parameters.SubVector(offset, count));
        }
    }

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        // Propagate to internal layers allocated lazily on first forward.
        // Null-guard for the pre-Forward state where sub-layers haven't been allocated yet.
        _expandConv?.SetTrainingMode(isTraining);
        _expandBn?.SetTrainingMode(isTraining);
        _dwConv?.SetTrainingMode(isTraining);
        _dwBn?.SetTrainingMode(isTraining);
        _se?.SetTrainingMode(isTraining);
        _projectConv?.SetTrainingMode(isTraining);
        _projectBn?.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Performs the forward pass of the Inverted Residual Block.
    /// </summary>
    /// <param name="input">The input tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The output tensor after the inverted residual computation.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy ctor leaves all sub-layers null until OnFirstForward
        // resolves the input channel count and allocates them. Subsequent
        // calls short-circuit via IsShapeResolved.
        if (!IsShapeResolved) OnFirstForward(input);

        // Post-OnFirstForward contract: the four mandatory sub-layers are
        // non-null. Capture as locals so flow analysis stops complaining
        // and the body reads cleanly.
        var dwConv = _dwConv ?? throw new InvalidOperationException("OnFirstForward did not allocate _dwConv.");
        var dwBn = _dwBn ?? throw new InvalidOperationException("OnFirstForward did not allocate _dwBn.");
        var projectConv = _projectConv ?? throw new InvalidOperationException("OnFirstForward did not allocate _projectConv.");
        var projectBn = _projectBn ?? throw new InvalidOperationException("OnFirstForward did not allocate _projectBn.");

        _lastInput = input;
        Tensor<T> x = input;

        // Expansion phase (if expansion > 1)
        if (_hasExpansion && _expandConv is not null && _expandBn is not null)
        {
            _lastExpandOut = _expandConv.Forward(x);
            _lastExpandBnOut = _expandBn.Forward(_lastExpandOut);
            _lastExpandActOut = ApplyBlockActivation(_lastExpandBnOut);
            x = _lastExpandActOut;
        }

        // Depthwise convolution phase
        _lastDwOut = dwConv.Forward(x);
        _lastDwBnOut = dwBn.Forward(_lastDwOut);
        _lastDwActOut = ApplyBlockActivation(_lastDwBnOut);
        x = _lastDwActOut;

        // Squeeze-and-Excitation phase (optional)
        // Note: SE layer expects NHWC format, but our tensors are in NCHW format
        if (_useSE && _se is not null)
        {
            // Transpose NCHW [B, C, H, W] -> NHWC [B, H, W, C]
            var seInput = TransposeNCHWToNHWC(x);
            var seOutput = _se.Forward(seInput);
            // Transpose NHWC [B, H, W, C] -> NCHW [B, C, H, W]
            _lastSeOut = TransposeNHWCToNCHW(seOutput);
            x = _lastSeOut;
        }

        // Projection phase (LINEAR - no activation)
        _lastProjectOut = projectConv.Forward(x);
        _lastProjectBnOut = projectBn.Forward(_lastProjectOut);

        // Residual connection (only if dimensions match)
        if (_useResidual)
        {
            return AddTensors(_lastProjectBnOut, input);
        }

        return _lastProjectBnOut;
    }

    /// <summary>
    /// Performs the forward pass on GPU, keeping data GPU-resident.
    /// </summary>
    /// <param name="inputs">The input tensors (expects single input).</param>
    /// <returns>The output tensor on GPU.</returns>
    public override Tensor<T> ForwardGpu(params Tensor<T>[] inputs)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("At least one input tensor is required.", nameof(inputs));

        if (Engine is not DirectGpuTensorEngine gpuEngine)
            throw new InvalidOperationException("ForwardGpu requires a DirectGpuTensorEngine.");

        var input = inputs[0];

        if (!IsShapeResolved) OnFirstForward(input);
        var dwConv = _dwConv ?? throw new InvalidOperationException("OnFirstForward did not allocate _dwConv.");
        var dwBn = _dwBn ?? throw new InvalidOperationException("OnFirstForward did not allocate _dwBn.");
        var projectConv = _projectConv ?? throw new InvalidOperationException("OnFirstForward did not allocate _projectConv.");
        var projectBn = _projectBn ?? throw new InvalidOperationException("OnFirstForward did not allocate _projectBn.");

        Tensor<T> x = input;

        // Expansion phase (if expansion > 1)
        if (_hasExpansion && _expandConv is not null && _expandBn is not null)
        {
            var expandOut = _expandConv.ForwardGpu(x);
            var expandBnOut = _expandBn.ForwardGpu(expandOut);
            expandOut.Dispose(); // Dispose intermediate tensor
            var expandAct = gpuEngine.ActivationGpu(expandBnOut, GetFusedActivationType());
            expandBnOut.Dispose(); // Dispose intermediate tensor
            x = expandAct;
        }

        // Depthwise convolution phase
        var dwOut = dwConv.ForwardGpu(x);
        // Dispose expansion output if we had expansion phase
        if (_hasExpansion && x != input)
            x.Dispose();
        var dwBnOut = dwBn.ForwardGpu(dwOut);
        dwOut.Dispose(); // Dispose intermediate tensor
        var dwAct = gpuEngine.ActivationGpu(dwBnOut, GetFusedActivationType());
        dwBnOut.Dispose(); // Dispose intermediate tensor
        x = dwAct;

        // Squeeze-and-Excitation phase (optional)
        if (_useSE && _se is not null)
        {
            // Permute from NCHW [B, C, H, W] to NHWC [B, H, W, C] for SE layer
            var xNhwc = gpuEngine.PermuteGpu(x, [0, 2, 3, 1]);
            x.Dispose(); // Dispose dwAct before SE

            // Apply SE layer (expects NHWC format)
            var seOut = _se.ForwardGpu(xNhwc);
            xNhwc.Dispose(); // Dispose intermediate tensor

            // Permute back from NHWC [B, H, W, C] to NCHW [B, C, H, W]
            var seOutNchw = gpuEngine.PermuteGpu(seOut, [0, 3, 1, 2]);
            seOut.Dispose(); // Dispose intermediate tensor

            x = seOutNchw;
        }

        // Projection phase (LINEAR - no activation)
        var projectOut = projectConv.ForwardGpu(x);
        x.Dispose(); // Dispose SE output (or dwAct if no SE)
        var projectBnOut = projectBn.ForwardGpu(projectOut);
        projectOut.Dispose(); // Dispose intermediate tensor

        // Residual connection (only if dimensions match)
        if (_useResidual)
        {
            var result = gpuEngine.AddGpu(projectBnOut, input);
            projectBnOut.Dispose(); // Dispose intermediate tensor
            return result;
        }

        return projectBnOut;
    }

    /// <summary>
    /// Updates the parameters of all sub-layers.
    /// </summary>
    /// <param name="learningRate">The learning rate for parameter updates.</param>
    public override void UpdateParameters(T learningRate)
    {
        _expandConv?.UpdateParameters(learningRate);
        _expandBn?.UpdateParameters(learningRate);
        _dwConv?.UpdateParameters(learningRate);
        _dwBn?.UpdateParameters(learningRate);
        _se?.UpdateParameters(learningRate);
        _projectConv?.UpdateParameters(learningRate);
        _projectBn?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters from the block.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        if (_expandConv is not null)
            parameters.AddRange(_expandConv.GetParameters().ToArray());
        if (_expandBn is not null)
            parameters.AddRange(_expandBn.GetParameters().ToArray());

        if (_dwConv is not null)
            parameters.AddRange(_dwConv.GetParameters().ToArray());
        if (_dwBn is not null)
            parameters.AddRange(_dwBn.GetParameters().ToArray());

        if (_se is not null)
            parameters.AddRange(_se.GetParameters().ToArray());

        if (_projectConv is not null)
            parameters.AddRange(_projectConv.GetParameters().ToArray());
        if (_projectBn is not null)
            parameters.AddRange(_projectBn.GetParameters().ToArray());

        return new Vector<T>(parameters.ToArray());
    }

    /// <summary>
    /// Sets all trainable parameters from the given parameter vector.
    /// </summary>
    /// <param name="parameters">The parameter vector containing all layer parameters.</param>
    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward path: sub-layers are still null because their
        // channel-count layout depends on the input we haven't seen
        // yet. Buffer the vector and replay it from OnFirstForward,
        // after sub-layers exist with their resolved shapes.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        ApplyParameters(parameters);
    }

    // --- ILayerSerializationExtras: serialize internal BN running stats ---

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            if (_expandBn is ILayerSerializationExtras<T> eb) count += eb.ExtraParameterCount;
            if (_dwBn is ILayerSerializationExtras<T> db) count += db.ExtraParameterCount;
            if (_projectBn is ILayerSerializationExtras<T> pb) count += pb.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var parts = new List<T>();
        if (_expandBn is ILayerSerializationExtras<T> eb)
            parts.AddRange(eb.GetExtraParameters().ToArray());
        if (_dwBn is ILayerSerializationExtras<T> db)
            parts.AddRange(db.GetExtraParameters().ToArray());
        if (_projectBn is ILayerSerializationExtras<T> pb)
            parts.AddRange(pb.GetExtraParameters().ToArray());
        return new Vector<T>(parts.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        // Buffer until OnFirstForward resolves shapes when arriving pre-
        // resolution: _expandBn/_dwBn/_projectBn are null at construction
        // (lazy ctor; allocated in OnFirstForward once inChannels is
        // observed), so the type-checks below all skip and the extras
        // are silently dropped. Without buffering, BN running mean/var
        // for a deserialized checkpoint stay at zero and inference
        // diverges from the trained model.
        if (!IsShapeResolved)
        {
            _pendingExtraParameters = extraParameters;
            return;
        }
        ApplyExtraParametersUnsafe(extraParameters);
    }

    /// <summary>
    /// Buffer for ILayerSerializationExtras.SetExtraParameters when
    /// called pre-OnFirstForward. Replayed inside OnFirstForward once
    /// _expandBn/_dwBn/_projectBn are allocated.
    /// </summary>
    private Vector<T>? _pendingExtraParameters;

    private void ApplyExtraParametersUnsafe(Vector<T> extraParameters)
    {
        int offset = 0;
        if (_expandBn is ILayerSerializationExtras<T> eb)
        {
            int count = eb.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for expand BN: need {offset + count} but got {extraParameters.Length}.");
            eb.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
        if (_dwBn is ILayerSerializationExtras<T> db)
        {
            int count = db.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for depthwise BN: need {offset + count} but got {extraParameters.Length}.");
            db.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
        if (_projectBn is ILayerSerializationExtras<T> pb)
        {
            int count = pb.ExtraParameterCount;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated extra-parameters for project BN: need {offset + count} but got {extraParameters.Length}.");
            pb.SetExtraParameters(extraParameters.SubVector(offset, count));
        }
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastExpandOut = null;
        _lastExpandBnOut = null;
        _lastExpandActOut = null;
        _lastDwOut = null;
        _lastDwBnOut = null;
        _lastDwActOut = null;
        _lastSeOut = null;
        _lastProjectOut = null;
        _lastProjectBnOut = null;

        // All sub-layers stay null until OnFirstForward resolves the
        // input channel count; null-guard so ResetState before any
        // forward is a no-op rather than throwing.
        _expandConv?.ResetState();
        _expandBn?.ResetState();
        _dwConv?.ResetState();
        _dwBn?.ResetState();
        _se?.ResetState();
        _projectConv?.ResetState();
        _projectBn?.ResetState();
    }



    #region Helper Methods

    private Tensor<T> ApplyBlockActivation(Tensor<T> input)
    {
        if (ScalarActivation is null)
            return input;
        return ScalarActivation.Activate(input);
    }

    private Tensor<T> ApplyBlockActivationDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        if (ScalarActivation is null)
            return gradient;
        var derivative = ScalarActivation.Derivative(input);
        return MultiplyTensors(gradient, derivative);
    }

    /// <summary>
    /// Multiplies two tensors element-wise using vectorized Engine operations.
    /// </summary>
    private Tensor<T> MultiplyTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorMultiply(a, b);
    }

    /// <summary>
    /// Adds two tensors element-wise using vectorized Engine operations.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    /// <summary>
    /// Transposes a tensor from NCHW to NHWC format.
    /// Supports both 3D [C, H, W] and 4D [B, C, H, W] inputs.
    /// Uses vectorized Engine.TensorPermute operation.
    /// </summary>
    private Tensor<T> TransposeNCHWToNHWC(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank == 3)
        {
            // 3D input [C, H, W] -> add batch dimension -> [1, C, H, W]
            var input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
            // NCHW [0, 1, 2, 3] -> NHWC [0, 2, 3, 1]
            var result4D = Engine.TensorPermute(input4D, [0, 2, 3, 1]);
            // Remove batch dimension -> [H, W, C]
            return result4D.Reshape([result4D.Shape[1], result4D.Shape[2], result4D.Shape[3]]);
        }
        else if (rank == 4)
        {
            // 4D input: NCHW [0, 1, 2, 3] -> NHWC [0, 2, 3, 1]
            return Engine.TensorPermute(input, [0, 2, 3, 1]);
        }
        else
        {
            // Higher rank: flatten leading dims, transpose, restore
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            var input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
            var result4D = Engine.TensorPermute(input4D, [0, 2, 3, 1]);
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 3; d++)
                outputShape[d] = input.Shape[d];
            outputShape[rank - 3] = result4D.Shape[1];
            outputShape[rank - 2] = result4D.Shape[2];
            outputShape[rank - 1] = result4D.Shape[3];
            return result4D.Reshape(outputShape);
        }
    }

    /// <summary>
    /// Transposes a tensor from NHWC to NCHW format.
    /// Supports both 3D [H, W, C] and 4D [B, H, W, C] inputs.
    /// Uses vectorized Engine.TensorPermute operation.
    /// </summary>
    private Tensor<T> TransposeNHWCToNCHW(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank == 3)
        {
            // 3D input [H, W, C] -> add batch dimension -> [1, H, W, C]
            var input4D = input.Reshape([1, input.Shape[0], input.Shape[1], input.Shape[2]]);
            // NHWC [0, 1, 2, 3] -> NCHW [0, 3, 1, 2]
            var result4D = Engine.TensorPermute(input4D, [0, 3, 1, 2]);
            // Remove batch dimension -> [C, H, W]
            return result4D.Reshape([result4D.Shape[1], result4D.Shape[2], result4D.Shape[3]]);
        }
        else if (rank == 4)
        {
            // 4D input: NHWC [0, 1, 2, 3] -> NCHW [0, 3, 1, 2]
            return Engine.TensorPermute(input, [0, 3, 1, 2]);
        }
        else
        {
            // Higher rank: flatten leading dims, transpose, restore
            int flatBatch = 1;
            for (int d = 0; d < rank - 3; d++)
                flatBatch *= input.Shape[d];
            var input4D = input.Reshape([flatBatch, input.Shape[rank - 3], input.Shape[rank - 2], input.Shape[rank - 1]]);
            var result4D = Engine.TensorPermute(input4D, [0, 3, 1, 2]);
            var outputShape = new int[rank];
            for (int d = 0; d < rank - 3; d++)
                outputShape[d] = input.Shape[d];
            outputShape[rank - 3] = result4D.Shape[1];
            outputShape[rank - 2] = result4D.Shape[2];
            outputShape[rank - 1] = result4D.Shape[3];
            return result4D.Reshape(outputShape);
        }
    }

    #endregion

    /// <summary>
    /// Returns layer-specific metadata for serialization purposes.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InChannels"] = InChannels.ToString();
        metadata["OutChannels"] = OutChannels.ToString();
        metadata["ExpansionRatio"] = ExpansionRatio.ToString();
        metadata["Stride"] = Stride.ToString();
        metadata["UseSE"] = _useSE.ToString();
        return metadata;
    }

}
