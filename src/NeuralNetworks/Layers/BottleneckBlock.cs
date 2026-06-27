using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the BottleneckBlock used in ResNet50, ResNet101, and ResNet152 architectures.
/// </summary>
/// <remarks>
/// <para>
/// The BottleneckBlock uses a 1x1-3x3-1x1 convolution pattern, where the 1x1 layers reduce and then
/// restore dimensions (with expansion), and the 3x3 layer is the bottleneck with smaller channels.
/// This design is more computationally efficient than stacking 3x3 convolutions for deep networks.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <code>
/// Input ─┬─ Conv1x1 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─ ReLU ─ Conv1x1 ─ BN ─┬─ (+) ─ ReLU ─ Output
///        │                                                             │
///        └────────────────────── [Downsample?] ────────────────────────┘
/// </code>
/// The first 1x1 conv reduces channels, the 3x3 processes at reduced channels,
/// and the final 1x1 expands channels by a factor of 4.
/// </para>
/// <para>
/// <b>For Beginners:</b> The BottleneckBlock is like a compressed processing pipeline.
///
/// Think of it as:
/// 1. First 1x1 conv: "Compress" - reduce the number of channels (like compressing a file)
/// 2. 3x3 conv: "Process" - do the heavy computation on the compressed representation
/// 3. Second 1x1 conv: "Expand" - restore and expand the channels
///
/// This is more efficient because:
/// - The expensive 3x3 convolution works on fewer channels
/// - The overall result has high capacity (4x expansion)
/// - Much fewer parameters than three 3x3 convolutions
///
/// The expansion factor of 4 means if the base channels is 64, the output will have 256 channels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 4, Cost = ComputeCost.High, TestInputShape = "1, 4, 8, 8", TestConstructorArgs = "4, 1")]
public class BottleneckBlock<T> : LayerBase<T>
{
    /// <summary>
    /// The expansion factor for BottleneckBlock. Output channels = base channels * 4.
    /// </summary>
    public const int Expansion = 4;

    private readonly ConvolutionalLayer<T> _conv1;  // 1x1 reduce
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv2;  // 3x3 bottleneck
    private readonly BatchNormalizationLayer<T> _bn2;
    private readonly ConvolutionalLayer<T> _conv3;  // 1x1 expand
    private readonly BatchNormalizationLayer<T> _bn3;
    // Non-readonly: lazy ctor leaves these null until OnFirstForward
    // observes the runtime input channel count.
    private ConvolutionalLayer<T>? _downsampleConv;
    private BatchNormalizationLayer<T>? _downsampleBn;
    private readonly IActivationFunction<T> _relu;
    // Non-readonly: lazy ctor leaves _hasDownsample = false until
    // OnFirstForward observes the runtime input channel count.
    private bool _hasDownsample;
    // Stored constructor args for serialization round-trip — without
    // these, DeserializationHelper defaults Stride=1 / ZeroInitResidual=true
    // which makes the cloned ResNet50 spatial dims diverge from the
    // original (224→56 instead of 224→7) and a single Predict call goes
    // from ~5s to ~80s due to convs running at 64× larger feature maps.
    // Non-readonly: lazy ctor leaves _inChannels = -1 until OnFirstForward
    // resolves it from the runtime input tensor's shape.
    private int _inChannels;
    private readonly int _baseChannels;
    private readonly int _stride;
    // Non-readonly: lazy ctor leaves _inputHeight/_inputWidth = -1 until
    // OnFirstForward resolves them from the runtime input tensor's shape.
    private int _inputHeight;
    private int _inputWidth;
    private readonly bool _zeroInitResidual;

    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastConv1Output;
    private Tensor<T>? _lastBn1Output;
    private Tensor<T>? _lastRelu1Output;
    private Tensor<T>? _lastConv2Output;
    private Tensor<T>? _lastBn2Output;
    private Tensor<T>? _lastRelu2Output;
    private Tensor<T>? _lastConv3Output;
    private Tensor<T>? _lastBn3Output;
    private Tensor<T>? _lastIdentity;
    private Tensor<T>? _lastPreActivation;

    // GPU cached tensors for backward pass
    private Tensor<T>? _gpuBn1Out;
    private Tensor<T>? _gpuBn2Out;
    private Tensor<T>? _gpuPreActivation;

    /// <summary>
    /// Gets a value indicating whether this layer supports training.
    /// </summary>
    public override long ParameterCount =>
        _conv1.ParameterCount + _bn1.ParameterCount + _conv2.ParameterCount + _bn2.ParameterCount +
        _conv3.ParameterCount + _bn3.ParameterCount +
        (_downsampleConv?.ParameterCount ?? 0) + (_downsampleBn?.ParameterCount ?? 0);
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="BottleneckBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="baseChannels">The base channel count (output will be baseChannels * 4).</param>
    /// <param name="stride">The stride for the 3x3 convolution (default: 1).</param>
    /// <param name="inputHeight">The input spatial height.</param>
    /// <param name="inputWidth">The input spatial width.</param>
    /// <param name="zeroInitResidual">If true, initialize the last BN to zero for better training stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The baseChannels parameter specifies the "bottleneck" width.
    /// The actual output channels will be baseChannels * 4 due to the expansion factor.
    /// For example, if baseChannels = 64, the output will have 256 channels.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>). Only
    /// <c>baseChannels</c> (kernel-sizing target) and <c>stride</c> are
    /// required at construction. The downsample shortcut's allocation
    /// is deferred to <see cref="OnFirstForward"/> because
    /// <c>_hasDownsample</c> depends on whether input channels match
    /// the expanded output channel count.
    /// </summary>
    public BottleneckBlock(
        int baseChannels,
        int stride = 1,
        bool zeroInitResidual = true)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [baseChannels * Expansion, -1, -1])
    {
        if (baseChannels <= 0) throw new ArgumentOutOfRangeException(nameof(baseChannels));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        _inChannels = -1; // resolved in OnFirstForward
        _baseChannels = baseChannels;
        _stride = stride;
        _inputHeight = -1;
        _inputWidth = -1;
        _zeroInitResidual = zeroInitResidual;
        int outChannels = baseChannels * Expansion;
        _relu = new ReLUActivation<T>();

        _conv1 = new ConvolutionalLayer<T>(
            outputDepth: baseChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
        _bn1 = new BatchNormalizationLayer<T>();
        _conv2 = new ConvolutionalLayer<T>(
            outputDepth: baseChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
        _bn2 = new BatchNormalizationLayer<T>();
        _conv3 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
        _bn3 = new BatchNormalizationLayer<T>();

        if (zeroInitResidual) _bn3.ZeroInitGamma();

        // Pre-resolve the obvious-true case for _hasDownsample. When
        // _stride != 1 the shortcut is mandatory regardless of runtime
        // channel count (spatial dims will differ), so we can flip the
        // flag at construction time. The actual sub-layer allocation
        // still happens in OnFirstForward because the conv kernel shape
        // depends on the runtime inChannels — but having the flag set
        // early makes pre-Forward queries (e.g. "does this block have a
        // skip branch?" inspection by NN-base or a debugger) report
        // accurately. The flag may also flip true in OnFirstForward for
        // the inChannels != outChannels path even when _stride == 1.
        if (_stride != 1) _hasDownsample = true;

        // Downsample sub-layer allocation deferred to OnFirstForward —
        // inChannels isn't known until input.Shape is observed.

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_bn1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_bn2);
        RegisterSubLayer(_conv3);
        RegisterSubLayer(_bn3);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves H/W from input.Shape and propagates to all sub-layers
    /// via ResolveShapesOnly so ParameterCount reports the real weight
    /// count before any sub-layer's first Forward fires.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inChannels, inputHeight, inputWidth;
        if (s.Length == 3) { inChannels = s[0]; inputHeight = s[1]; inputWidth = s[2]; }
        else if (s.Length == 4) { inChannels = s[1]; inputHeight = s[2]; inputWidth = s[3]; }
        else
            throw new ArgumentException(
                $"BottleneckBlock requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        _inChannels = inChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        // Conv output dim for the 3×3 / pad=1 / stride=_stride middle path:
        //   out = (in + 2·pad − kernel) / stride + 1
        // With pad=1, kernel=3 this is `(in − 1) / stride + 1`. Plain
        // floor division `in / stride` is wrong for odd inputs and
        // produces a 1-off mismatch between conv2's output and the
        // downsample BN's expected shape, breaking the residual add.
        int outH = (inputHeight - 1) / _stride + 1;
        int outW = (inputWidth - 1) / _stride + 1;
        int outChannels = _baseChannels * Expansion;

        // Downsample shortcut: needed when stride != 1 or channel counts differ.
        _hasDownsample = _stride != 1 || _inChannels != outChannels;
        if (_hasDownsample)
        {
            var downConv = new ConvolutionalLayer<T>(
                outputDepth: outChannels,
                kernelSize: 1,
                stride: _stride,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
            var downBn = new BatchNormalizationLayer<T>();
            _downsampleConv = downConv;
            _downsampleBn = downBn;
            RegisterSubLayer(downConv);
            RegisterSubLayer(downBn);
            // Propagate the parent's current training mode — without this,
            // a layer that was switched to eval mode before its first
            // Forward would still see fresh sub-layers in default training
            // mode, causing batch-1 BN to collapse to zeros.
            downConv.SetTrainingMode(IsTrainingMode);
            downBn.SetTrainingMode(IsTrainingMode);
        }

        // Use ResolveFromShape so weights are allocated up front — needed
        // for any buffered Deserialize parameters to slice correctly.
        _conv1.ResolveFromShape(new[] { _inChannels, inputHeight, inputWidth });
        _bn1.ResolveFromShape(new[] { 1, _baseChannels, inputHeight, inputWidth });
        _conv2.ResolveFromShape(new[] { _baseChannels, inputHeight, inputWidth });
        _bn2.ResolveFromShape(new[] { 1, _baseChannels, outH, outW });
        _conv3.ResolveFromShape(new[] { _baseChannels, outH, outW });
        _bn3.ResolveFromShape(new[] { 1, outChannels, outH, outW });
        _downsampleConv?.ResolveFromShape(new[] { _inChannels, inputHeight, inputWidth });
        _downsampleBn?.ResolveFromShape(new[] { 1, outChannels, outH, outW });

        ResolveShapes(
            new[] { _inChannels, inputHeight, inputWidth },
            new[] { outChannels, outH, outW });

        // Replay parameters that arrived via Deserialize → SetParameters
        // before any sub-layer shape was resolved. With weights now
        // allocated, slicing matches the GetParameters() ordering.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyParameters(pending);
        }
    }

    // Constructor args round-trip for serialization. DeserializationHelper
    // reads BaseChannels (via OutputChannels/Expansion), Stride, and
    // ZeroInitResidual to recreate an identically-configured block.
    // Without these, downsample blocks (stride=2 in ResNet50 stages 2/3/4)
    // reconstruct with stride=1 — the cloned network's spatial dims stay
    // 56×56 through every stage instead of shrinking to 7×7, so each
    // Predict call does 64× more work in the late stages and a 5s
    // Predict turns into 80+ seconds.
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ic = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _inChannels.ToString(ic);
        metadata["BaseChannels"] = _baseChannels.ToString(ic);
        metadata["Stride"] = _stride.ToString(ic);
        metadata["InputHeight"] = _inputHeight.ToString(ic);
        metadata["InputWidth"] = _inputWidth.ToString(ic);
        metadata["ZeroInitResidual"] = _zeroInitResidual.ToString(ic);
        return metadata;
    }

    /// <summary>
    /// Performs the forward pass through the BottleneckBlock.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after the residual connection.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Lazy ctor leaves _hasDownsample / _inChannels unresolved until
        // OnFirstForward observes input.Shape. Resolve before the
        // identity-branch decision below.
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3
        _lastConv1Output = _conv1.Forward(input);
        _lastBn1Output = _bn1.Forward(_lastConv1Output);
        _lastRelu1Output = ApplyReLU(_lastBn1Output);

        _lastConv2Output = _conv2.Forward(_lastRelu1Output);
        _lastBn2Output = _bn2.Forward(_lastConv2Output);
        _lastRelu2Output = ApplyReLU(_lastBn2Output);

        _lastConv3Output = _conv3.Forward(_lastRelu2Output);
        _lastBn3Output = _bn3.Forward(_lastConv3Output);

        // Identity/skip branch
        if (_hasDownsample && _downsampleConv is not null && _downsampleBn is not null)
        {
            var dsConvOut = _downsampleConv.Forward(input);
            _lastIdentity = _downsampleBn.Forward(dsConvOut);
        }
        else
        {
            _lastIdentity = input;
        }

        // Add residual connection
        _lastPreActivation = Engine.TensorAdd(_lastBn3Output, _lastIdentity);

        // Final ReLU
        return ApplyReLU(_lastPreActivation);
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

        // Mirror the lazy-init guard from Forward(): if this block's
        // first ever execution lands on the GPU path, _hasDownsample
        // stays at its construction default (only true when stride!=1
        // since #7) and _downsampleConv stays null even when the
        // channel-mismatch path was needed — silently dropping the skip
        // branch and producing a residual add against the wrong shape.
        // OnFirstForward resolves shapes and allocates _downsampleConv.
        if (!IsShapeResolved) OnFirstForward(input);

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> conv3 -> bn3
        var conv1Out = _conv1.ForwardGpu(input);
        var bn1Out = _bn1.ForwardGpu(conv1Out);

        // Cache bn1Out for backward pass (ReLU1 backward needs it)
        _gpuBn1Out = bn1Out;

        var relu1Out = gpuEngine.ReluGpu(bn1Out);
        var conv2Out = _conv2.ForwardGpu(relu1Out);
        var bn2Out = _bn2.ForwardGpu(conv2Out);

        // Cache bn2Out for backward pass (ReLU2 backward needs it)
        _gpuBn2Out = bn2Out;

        var relu2Out = gpuEngine.ReluGpu(bn2Out);
        var conv3Out = _conv3.ForwardGpu(relu2Out);
        var bn3Out = _bn3.ForwardGpu(conv3Out);

        // Identity/skip branch
        Tensor<T> identity;
        if (_hasDownsample && _downsampleConv is not null && _downsampleBn is not null)
        {
            var dsConvOut = _downsampleConv.ForwardGpu(input);
            identity = _downsampleBn.ForwardGpu(dsConvOut);
        }
        else
        {
            identity = input;
        }

        // Add residual connection
        var preActivation = gpuEngine.AddGpu(bn3Out, identity);

        // Cache preActivation for backward pass (final ReLU backward needs it)
        _gpuPreActivation = preActivation;

        // Final ReLU
        return gpuEngine.ReluGpu(preActivation);
    }

    /// <summary>
    /// Updates the parameters of all internal layers.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _bn1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _bn2.UpdateParameters(learningRate);
        _conv3.UpdateParameters(learningRate);
        _bn3.UpdateParameters(learningRate);
        _downsampleConv?.UpdateParameters(learningRate);
        _downsampleBn?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters.
    /// </summary>
    /// <returns>A vector containing all parameters.</returns>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        allParams.AddRange(_conv1.GetParameters().ToArray());
        allParams.AddRange(_bn1.GetParameters().ToArray());
        allParams.AddRange(_conv2.GetParameters().ToArray());
        allParams.AddRange(_bn2.GetParameters().ToArray());
        allParams.AddRange(_conv3.GetParameters().ToArray());
        allParams.AddRange(_bn3.GetParameters().ToArray());
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            allParams.AddRange(_downsampleConv.GetParameters().ToArray());
            allParams.AddRange(_downsampleBn.GetParameters().ToArray());
        }
        return new Vector<T>([.. allParams]);
    }

    public override Vector<T> GetParameterGradients()
    {
        var grads = new List<T>();
        grads.AddRange(_conv1.GetParameterGradients().ToArray());
        grads.AddRange(_bn1.GetParameterGradients().ToArray());
        grads.AddRange(_conv2.GetParameterGradients().ToArray());
        grads.AddRange(_bn2.GetParameterGradients().ToArray());
        grads.AddRange(_conv3.GetParameterGradients().ToArray());
        grads.AddRange(_bn3.GetParameterGradients().ToArray());
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            grads.AddRange(_downsampleConv.GetParameterGradients().ToArray());
            grads.AddRange(_downsampleBn.GetParameterGradients().ToArray());
        }
        return new Vector<T>([.. grads]);
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _conv1.ClearGradients(); _bn1.ClearGradients();
        _conv2.ClearGradients(); _bn2.ClearGradients();
        _conv3.ClearGradients(); _bn3.ClearGradients();
        _downsampleConv?.ClearGradients(); _downsampleBn?.ClearGradients();
    }

    public override void SetParameters(Vector<T> parameters)
    {
        // Pre-Forward: every sub-layer's shape is unresolved so each
        // ParameterCount returns 0 — slicing collapses. Buffer the
        // whole vector; OnFirstForward replays it after sub-layer
        // shapes (and any conditionally-allocated downsample) exist.
        if (!IsShapeResolved)
        {
            _pendingParameters = parameters;
            return;
        }

        ApplyParameters(parameters);
    }

    private Vector<T>? _pendingParameters;

    private void ApplyParameters(Vector<T> parameters)
    {
        int idx = 0;
        void Set(ILayer<T> layer)
        {
            int count = checked((int)layer.ParameterCount);
            layer.SetParameters(parameters.Slice(idx, count));
            idx += count;
        }
        Set(_conv1); Set(_bn1); Set(_conv2); Set(_bn2); Set(_conv3); Set(_bn3);
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            Set(_downsampleConv); Set(_downsampleBn);
        }
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _lastConv1Output = null;
        _lastBn1Output = null;
        _lastRelu1Output = null;
        _lastConv2Output = null;
        _lastBn2Output = null;
        _lastRelu2Output = null;
        _lastConv3Output = null;
        _lastBn3Output = null;
        _lastIdentity = null;
        _lastPreActivation = null;

        _conv1.ResetState();
        _bn1.ResetState();
        _conv2.ResetState();
        _bn2.ResetState();
        _conv3.ResetState();
        _bn3.ResetState();
        _downsampleConv?.ResetState();
        _downsampleBn?.ResetState();
    }

    private Tensor<T> ApplyReLU(Tensor<T> input)
    {
        return Engine.ReLU(input);
    }

    private Tensor<T> ApplyReLUDerivative(Tensor<T> preActivation, Tensor<T> gradient)
    {
        // ReLU derivative: 1 if x > 0, 0 otherwise
        var derivative = preActivation.Transform((x, _) => _relu.Derivative(x));
        return Engine.TensorMultiply(gradient, derivative);
    }

}
