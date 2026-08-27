using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements the BasicBlock used in ResNet18 and ResNet34 architectures.
/// </summary>
/// <remarks>
/// <para>
/// The BasicBlock contains two 3x3 convolutional layers with batch normalization and ReLU activation.
/// A skip connection adds the input directly to the output, enabling gradient flow through very deep networks.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <code>
/// Input ─┬─ Conv3x3 ─ BN ─ ReLU ─ Conv3x3 ─ BN ─┬─ (+) ─ ReLU ─ Output
///        │                                       │
///        └───────────── [Downsample?] ───────────┘
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The BasicBlock is like a "learning module" with a shortcut.
///
/// The key insight is:
/// - The two conv layers learn to predict what needs to be ADDED to the input (the "residual")
/// - The skip connection adds the original input back to this learned residual
/// - This makes it easier to train very deep networks because gradients can flow directly through the skip connection
///
/// When the input and output have different dimensions (due to stride or channel changes),
/// a downsample layer (1x1 conv + BN) is used to match the dimensions before adding.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 8, 8", TestConstructorArgs = "1")]
// Exactly the two ranks this block's own guard admits - OnFirstForward: "requires rank-3 [C,H,W] or
// rank-4 [B,C,H,W] input" - declared separately rather than as one BatchOptional layout, because a
// derived contract keys on the declared axis count and would leave the unbatched rank resolving to
// nothing. OutputAxesFor is hand-written: both spatial extents depend on the constructor's stride.
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class BasicBlock<T> : LayerBase<T>, ILayerSerializationExtras<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Taken from this block's own resolution, not from the residual add: <c>OnFirstForward</c> computes
    /// <c>outH = (inputHeight - 1) / _stride + 1</c> and resolves
    /// <c>ResolveShapes(new[] { _inChannels, inputHeight, inputWidth }, new[] { _outChannels, outH, outW })</c>.
    /// </para>
    /// <para>
    /// That expression IS the sliding window for the 3x3 / pad=1 main path, which is why it is declared as
    /// <c>Window(kernel: 3, stride: _stride, padding: 1)</c> rather than as a division:
    /// <c>floor((in + 2*1 - (3-1) - 1) / stride) + 1</c> reduces to <c>(in - 1) / stride + 1</c> exactly.
    /// The layer's own comment flags the alternative as a real defect - plain <c>in / stride</c> gives 3
    /// for in=7, stride=2 where the convolution produces 4, and the mismatch reaches the downsample
    /// branch's BatchNorm shape and breaks the residual add.
    /// </para>
    /// <para>
    /// Channels are <c>Fixed(_outChannels)</c> and not <c>Same</c>: the shortcut is 1x1-convolved and
    /// re-normalised precisely when <c>_stride != 1 || _inChannels != _outChannels</c>, so the block
    /// widens its input rather than carrying the channel count through.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (3 or 4) || _outChannels <= 0 || _stride <= 0) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels));
        var height = new OutputAxisContract(
            TensorAxis.Height, AxisRelation.Window(TensorAxis.Height, kernel: 3, stride: _stride, padding: 1));
        var width = new OutputAxisContract(
            TensorAxis.Width, AxisRelation.Window(TensorAxis.Width, kernel: 3, stride: _stride, padding: 1));

        return inputRank == 3
            ? new[] { channels, height, width }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, height, width,
            };
    }

    /// <summary>
    /// The expansion factor for BasicBlock. BasicBlock does not expand channels.
    /// </summary>
    public const int Expansion = 1;

    private readonly ConvolutionalLayer<T> _conv1;
    private readonly BatchNormalizationLayer<T> _bn1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly BatchNormalizationLayer<T> _bn2;
    // Non-readonly: lazy ctor leaves these null until OnFirstForward
    // observes the runtime input channel count and decides whether the
    // residual shortcut needs a 1×1 projection.
    private ConvolutionalLayer<T>? _downsampleConv;
    private BatchNormalizationLayer<T>? _downsampleBn;
    private readonly IActivationFunction<T> _relu;
    // Non-readonly: lazy ctor leaves _hasDownsample = false until
    // OnFirstForward observes the runtime input channel count and
    // decides whether the residual shortcut needs a 1×1 projection.
    private bool _hasDownsample;
    // Stored constructor args needed for serialization round-trip
    // (DeserializationHelper reads these from GetMetadata to reconstruct
    // an identically-configured block — without them, stride/inChannels/
    // zeroInitResidual all default to wrong values for downsample blocks
    // and the cloned ResNet's spatial dimensions diverge from the
    // original's, producing wrong inference output).
    // Non-readonly: lazy ctor leaves _inChannels = -1 until OnFirstForward
    // resolves it from the runtime input tensor's shape.
    private int _inChannels;
    private readonly int _outChannels;
    private readonly int _stride;
    // Non-readonly: lazy ctor leaves _inputHeight/_inputWidth = -1 until
    // OnFirstForward resolves them from the runtime input tensor's shape.
    private int _inputHeight;
    private int _inputWidth;
    private readonly bool _zeroInitResidual;

    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastConv1Output;
    [Scratch]
    private Tensor<T>? _lastBn1Output;
    [Scratch]
    private Tensor<T>? _lastRelu1Output;
    [Scratch]
    private Tensor<T>? _lastConv2Output;
    [Scratch]
    private Tensor<T>? _lastBn2Output;
    [Scratch]
    private Tensor<T>? _lastIdentity;
    [Scratch]
    private Tensor<T>? _lastPreActivation;

    // GPU cached tensors for backward pass
    [ExternalState]
    private Tensor<T>? _gpuBn1Out;
    [ExternalState]
    private Tensor<T>? _gpuBn2Out;
    [ExternalState]
    private Tensor<T>? _gpuPreActivation;

    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer has a GPU implementation.
    /// </summary>
    protected override bool SupportsGpuExecution => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="BasicBlock{T}"/> class.
    /// </summary>
    /// <param name="inChannels">The number of input channels.</param>
    /// <param name="outChannels">The number of output channels.</param>
    /// <param name="stride">The stride for the first convolution (default: 1).</param>
    /// <param name="inputHeight">The input spatial height.</param>
    /// <param name="inputWidth">The input spatial width.</param>
    /// <param name="zeroInitResidual">If true, initialize the last BN to zero for better training stability.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When stride > 1, the block will downsample the spatial dimensions.
    /// When inChannels != outChannels, a projection shortcut is used to match dimensions.
    /// </para>
    /// </remarks>
    /// <summary>
    /// Lazy ctor — input depth/height/width come from the first
    /// <see cref="Forward"/> call (<see cref="OnFirstForward"/>). Only
    /// <c>outChannels</c> (the conv kernel-sizing target) and
    /// <c>stride</c> are required at construction. The downsample
    /// shortcut's allocation is deferred to <see cref="OnFirstForward"/>
    /// because <c>_hasDownsample</c> depends on whether input channels
    /// match <c>outChannels</c>.
    /// </summary>
    public BasicBlock(
        int outChannels,
        int stride = 1,
        bool zeroInitResidual = true)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [outChannels, -1, -1])
    {
        if (outChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outChannels));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

        _inChannels = -1; // resolved in OnFirstForward
        _outChannels = outChannels;
        _stride = stride;
        _inputHeight = -1; // resolved in OnFirstForward
        _inputWidth = -1;
        _zeroInitResidual = zeroInitResidual;
        _relu = new ReLUActivation<T>();

        _conv1 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: stride,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
        _bn1 = new BatchNormalizationLayer<T>();
        _conv2 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
        _bn2 = new BatchNormalizationLayer<T>();

        if (zeroInitResidual) _bn2.ZeroInitGamma();

        // Downsample allocation deferred to OnFirstForward — _hasDownsample
        // depends on (stride != 1 || inChannels != outChannels), and
        // inChannels isn't known until input.Shape is observed.

        RegisterSubLayer(_conv1);
        RegisterSubLayer(_bn1);
        RegisterSubLayer(_conv2);
        RegisterSubLayer(_bn2);
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
                $"BasicBlock requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));

        _inChannels = inChannels;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        // Conv output dim for the 3×3 / pad=1 / stride=_stride main path:
        //   out = (in + 2·pad − kernel) / stride + 1
        // With pad=1, kernel=3 this is `(in − 1) / stride + 1`. Plain
        // floor division `in / stride` is wrong for odd inputs (e.g.
        // in=7, stride=2 gives 3 by floor-div but the conv actually
        // produces 4) — that mismatch propagates to the downsample
        // branch's BN shape and breaks the residual add.
        int outH = (inputHeight - 1) / _stride + 1;
        int outW = (inputWidth - 1) / _stride + 1;

        // Downsample shortcut: needed when stride != 1 or channel counts differ.
        _hasDownsample = _stride != 1 || _inChannels != _outChannels;
        if (_hasDownsample)
        {
            var downConv = new ConvolutionalLayer<T>(
                outputDepth: _outChannels,
                kernelSize: 1,
                stride: _stride,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
            var downBn = new BatchNormalizationLayer<T>();
            _downsampleConv = downConv;
            _downsampleBn = downBn;
            RegisterSubLayer(downConv);
            RegisterSubLayer(downBn);
            // Propagate the parent's training mode — see BottleneckBlock
            // for the same fix; otherwise batch-1 BN collapses to zero.
            downConv.SetTrainingMode(IsTrainingMode);
            downBn.SetTrainingMode(IsTrainingMode);
        }

        // Use ResolveFromShape so weights are allocated up front — needed
        // for any buffered Deserialize parameters to slice correctly.
        _conv1.ResolveFromShape(new[] { _inChannels, inputHeight, inputWidth });
        _bn1.ResolveFromShape(new[] { 1, _outChannels, outH, outW });
        _conv2.ResolveFromShape(new[] { _outChannels, outH, outW });
        _bn2.ResolveFromShape(new[] { 1, _outChannels, outH, outW });
        _downsampleConv?.ResolveFromShape(new[] { _inChannels, inputHeight, inputWidth });
        _downsampleBn?.ResolveFromShape(new[] { 1, _outChannels, outH, outW });

        ResolveShapes(
            new[] { _inChannels, inputHeight, inputWidth },
            new[] { _outChannels, outH, outW });

        // Replay parameters that arrived via Deserialize → SetParameters
        // before any sub-layer shape was resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            ApplyParameters(pending);
        }

        // Replay BN running-stats extras that arrived pre-resolution (see the
        // ILayerSerializationExtras implementation): without this a cloned block's
        // BN running mean/var stay at their reset defaults and eval-mode inference
        // diverges from the trained model even though weights are byte-identical
        // (Clone_AfterTraining).
        if (_pendingExtraParameters is not null)
        {
            var pendingExtras = _pendingExtraParameters;
            _pendingExtraParameters = null;
            ApplyExtraParametersUnsafe(pendingExtras);
        }
    }

    // Constructor args round-trip for serialization. DeserializationHelper
    // reads these to recreate an identically-configured block — without
    // them, downsample blocks (stride=2 in stage 2/3/4) reconstruct with
    // stride=1, keeping spatial dims unchanged through the network and
    // producing wrong inference output in the cloned model.
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ic = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InChannels"] = _inChannels.ToString(ic);
        metadata["OutChannels"] = _outChannels.ToString(ic);
        metadata["Stride"] = _stride.ToString(ic);
        metadata["InputHeight"] = _inputHeight.ToString(ic);
        metadata["InputWidth"] = _inputWidth.ToString(ic);
        metadata["ZeroInitResidual"] = _zeroInitResidual.ToString(ic);
        return metadata;
    }

    /// <summary>
    /// Performs the forward pass through the BasicBlock.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor after the residual connection.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // Lazy ctor leaves _hasDownsample / _inChannels unresolved until
        // OnFirstForward observes input.Shape.
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2
        _lastConv1Output = _conv1.Forward(input);
        _lastBn1Output = _bn1.Forward(_lastConv1Output);
        _lastRelu1Output = ApplyReLU(_lastBn1Output);
        _lastConv2Output = _conv2.Forward(_lastRelu1Output);
        _lastBn2Output = _bn2.Forward(_lastConv2Output);

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
        _lastPreActivation = Engine.TensorAdd(_lastBn2Output, _lastIdentity);

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
        // first ever execution is on the GPU path, _hasDownsample /
        // _downsampleConv stay false/null and the skip branch is
        // silently dropped (residual identity = raw input even when
        // the stage requires stride=2 downsampling). OnFirstForward
        // resolves shapes + allocates _downsampleConv / _downsampleBn.
        if (!IsShapeResolved) OnFirstForward(input);

        // Main branch: conv1 -> bn1 -> relu -> conv2 -> bn2
        var conv1Out = _conv1.ForwardGpu(input);
        var bn1Out = _bn1.ForwardGpu(conv1Out);

        // Cache bn1Out for backward pass (ReLU1 backward needs it)
        _gpuBn1Out = bn1Out;

        var relu1Out = gpuEngine.ReluGpu(bn1Out);
        var conv2Out = _conv2.ForwardGpu(relu1Out);
        var bn2Out = _bn2.ForwardGpu(conv2Out);

        // Cache bn2Out for backward pass (ReLU2/final backward needs it)
        _gpuBn2Out = bn2Out;

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
        var preActivation = gpuEngine.AddGpu(bn2Out, identity);

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
        _downsampleConv?.UpdateParameters(learningRate);
        _downsampleBn?.UpdateParameters(learningRate);
    }

    public override Vector<T> GetParameterGradients()
    {
        var grads = new List<T>();
        grads.AddRange(_conv1.GetParameterGradients().ToArray());
        grads.AddRange(_bn1.GetParameterGradients().ToArray());
        grads.AddRange(_conv2.GetParameterGradients().ToArray());
        grads.AddRange(_bn2.GetParameterGradients().ToArray());
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
        _downsampleConv?.ClearGradients(); _downsampleBn?.ClearGradients();
    }

    [Scratch]
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
        Set(_conv1); Set(_bn1); Set(_conv2); Set(_bn2);
        if (_downsampleConv is not null && _downsampleBn is not null)
        {
            Set(_downsampleConv); Set(_downsampleBn);
        }
    }

    // --- ILayerSerializationExtras: round-trip the internal BN layers' running
    // mean/variance (non-trainable state excluded from GetParameters). Without
    // this the block loses its BN statistics on serialize/clone and eval-mode
    // inference diverges (Clone_AfterTraining). BN order matches GetParameters:
    // bn1, bn2, then the optional downsample BN. ---

    int ILayerSerializationExtras<T>.ExtraParameterCount
    {
        get
        {
            int count = 0;
            if (_bn1 is ILayerSerializationExtras<T> b1) count += b1.ExtraParameterCount;
            if (_bn2 is ILayerSerializationExtras<T> b2) count += b2.ExtraParameterCount;
            if (_downsampleBn is ILayerSerializationExtras<T> db) count += db.ExtraParameterCount;
            return count;
        }
    }

    Vector<T> ILayerSerializationExtras<T>.GetExtraParameters()
    {
        var parts = new List<T>();
        if (_bn1 is ILayerSerializationExtras<T> b1) parts.AddRange(b1.GetExtraParameters().ToArray());
        if (_bn2 is ILayerSerializationExtras<T> b2) parts.AddRange(b2.GetExtraParameters().ToArray());
        if (_downsampleBn is ILayerSerializationExtras<T> db) parts.AddRange(db.GetExtraParameters().ToArray());
        return new Vector<T>(parts.ToArray());
    }

    void ILayerSerializationExtras<T>.SetExtraParameters(Vector<T> extraParameters)
    {
        if (!IsShapeResolved)
        {
            _pendingExtraParameters = extraParameters;
            return;
        }
        ApplyExtraParametersUnsafe(extraParameters);
    }

    [Scratch]
    private Vector<T>? _pendingExtraParameters;

    private void ApplyExtraParametersUnsafe(Vector<T> extraParameters)
    {
        int offset = 0;
        void Apply(BatchNormalizationLayer<T>? bn)
        {
            if (bn is not ILayerSerializationExtras<T> ex) return;
            int count = ex.ExtraParameterCount;
            if (count == 0) return;
            if (offset + count > extraParameters.Length)
                throw new ArgumentException(
                    $"Truncated BasicBlock extra-parameters: need {offset + count} but got {extraParameters.Length}.");
            ex.SetExtraParameters(extraParameters.SubVector(offset, count));
            offset += count;
        }
        Apply(_bn1); Apply(_bn2); Apply(_downsampleBn);
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
        _lastIdentity = null;
        _lastPreActivation = null;

        _conv1.ResetState();
        _bn1.ResetState();
        _conv2.ResetState();
        _bn2.ResetState();
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
