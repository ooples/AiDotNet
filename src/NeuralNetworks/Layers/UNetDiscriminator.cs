using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// U-Net Discriminator as used in Real-ESRGAN for improved perceptual quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the U-Net discriminator from the Real-ESRGAN paper (Wang et al., 2021).
/// Unlike traditional patch discriminators, U-Net discriminator provides pixel-level feedback
/// which helps the generator produce finer details.
/// </para>
/// <para>
/// The architecture has an encoder-decoder structure:
/// <code>
/// Input (3 channels, HR image)
///   ↓
/// Encoder (progressively downsample with skip connections)
///   ↓
/// Bottleneck
///   ↓
/// Decoder (progressively upsample, concat with skip connections)
///   ↓
/// Output (1 channel, per-pixel real/fake prediction)
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> The discriminator judges whether an image is real or fake.
///
/// Traditional discriminators output a single "real/fake" score for the whole image.
/// U-Net discriminator outputs a "real/fake" prediction for EVERY PIXEL, which:
/// - Provides more detailed feedback to the generator
/// - Helps produce sharper details and textures
/// - Enables better reconstruction of fine features
///
/// The U-Net architecture (encoder + decoder with skip connections) allows the
/// discriminator to consider both local details and global context.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
/// with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 3, 8, 8", TestConstructorArgs = "8, 2")]
// Both ranks are declared because OnFirstForward accepts both by name - "requires rank-3 [C,H,W] or
// rank-4 [B,C,H,W] input" - and the layer's own [LayerProperty] claims one of each (ExpectedInputRank
// = 3, TestInputShape rank 4). Written as two declarations rather than one BatchOptional layout so
// that BOTH ranks appear literally, which is what the rank cross-check reads.
//
// OutputAxesFor below is HAND-WRITTEN: the channel axis is not carried through, so the generator's
// Same(role)-per-axis derivation would be wrong for it.
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class UNetDiscriminator<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// SPATIAL DIMENSIONS SURVIVE THE PYRAMID, which is the whole point of a U-Net discriminator: it
    /// scores every pixel rather than the image as a whole. OnFirstForward states it directly -
    /// <c>ResolveShapes(new[] { inC, inH, inW }, new[] { 1, inH, inW })</c> - and the encoder/decoder
    /// walk in that same method shows why it is exact rather than approximate: each of the
    /// <c>_numBlocks</c> encoder stages does <c>currentH = (currentH + 1) / 2</c> and each decoder
    /// stage does <c>currentH *= 2</c>, and the guard above them rejects any input whose H or W is not
    /// divisible by <c>2^numBlocks</c>. Under that guard halving and doubling cancel exactly, so
    /// <c>Same</c> is the true relation and no <c>Window</c> is needed.
    /// </para>
    /// <para>
    /// The channel count is read off this layer's OWN declared output shape rather than written as the
    /// literal 1. The single channel is architectural - <c>_convLast</c> is built with
    /// <c>outputDepth: 1</c> and the constructor declares <c>base([-1,-1,-1], [1,-1,-1])</c>
    /// "Per-pixel output" - but reading it keeps the contract tied to the declaration instead of
    /// duplicating it, so the two cannot drift apart.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        var declaredOutput = GetOutputShape();
        if (declaredOutput is null || declaredOutput.Length != 3) return null;

        int outChannels = declaredOutput[0];
        if (outChannels <= 0) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(outChannels));
        var height = new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height));
        var width = new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width));

        return inputRank switch
        {
            3 => new[] { channels, height, width },
            4 => new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, height, width,
            },
            _ => null,
        };
    }

    #region Fields

    /// <summary>
    /// Encoder blocks (downsampling path).
    /// </summary>
    private readonly UNetConvBlock<T>[] _encoderBlocks;

    /// <summary>
    /// Decoder blocks (upsampling path).
    /// </summary>
    private readonly UNetUpBlock<T>[] _decoderBlocks;

    /// <summary>
    /// Initial convolution.
    /// </summary>
    private readonly ConvolutionalLayer<T> _convFirst;

    /// <summary>
    /// Final convolution (1x1 to output channels).
    /// </summary>
    private readonly ConvolutionalLayer<T> _convLast;

    /// <summary>
    /// Number of encoder/decoder blocks.
    /// </summary>
    private readonly int _numBlocks;

    /// <summary>
    /// Base number of channels.
    /// </summary>
    private readonly int _numChannels;

    /// <summary>
    /// LeakyReLU activation.
    /// </summary>
    private readonly LeakyReLUActivation<T> _leakyReLU;

    /// <summary>
    /// Skip connections stored during forward pass for concatenation.
    /// </summary>
    private Tensor<T>[]? _skipConnections;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastInput;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the number of encoder/decoder blocks.
    /// </summary>
    public int NumBlocks => _numBlocks;

    /// <summary>
    /// Gets the base number of channels.
    /// </summary>
    public int NumChannels => _numChannels;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets a value indicating whether this layer supports GPU execution.
    /// </summary>
    protected override bool SupportsGpuExecution => false;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new U-Net discriminator.
    /// </summary>
    /// <param name="inputHeight">Height of input image.</param>
    /// <param name="inputWidth">Width of input image.</param>
    /// <param name="inputChannels">Number of input channels (3 for RGB).</param>
    /// <param name="numChannels">Base number of channels. Default: 64.</param>
    /// <param name="numBlocks">Number of encoder/decoder blocks. Default: 4.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a U-Net discriminator for Real-ESRGAN:
    /// <code>
    /// var discriminator = new UNetDiscriminator&lt;float&gt;(
    ///     inputHeight: 256,
    ///     inputWidth: 256,
    ///     inputChannels: 3,
    ///     numChannels: 64,
    ///     numBlocks: 4
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public UNetDiscriminator(
        int numChannels = 64,
        int numBlocks = 4)
        : base(
            [-1, -1, -1],
            [1, -1, -1]) // Per-pixel output
    {
        if (numBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numBlocks), "Number of blocks must be positive.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");

        _numBlocks = numBlocks;
        _numChannels = numChannels;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        // Initial convolution: inputChannels (lazy) → numChannels.
        _convFirst = new ConvolutionalLayer<T>(
            outputDepth: numChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Encoder blocks (progressively downsample and double channels — capped at 8×).
        _encoderBlocks = new UNetConvBlock<T>[numBlocks];
        int currentChannels = numChannels;
        for (int i = 0; i < numBlocks; i++)
        {
            int outChannels = Math.Min(currentChannels * 2, numChannels * 8);
            _encoderBlocks[i] = new UNetConvBlock<T>(outChannels, downsample: true);
            currentChannels = outChannels;
        }

        // Decoder blocks (progressively upsample, use skip connections).
        _decoderBlocks = new UNetUpBlock<T>[numBlocks];
        for (int i = numBlocks - 1; i >= 0; i--)
        {
            int skipChannels = i == 0 ? numChannels : Math.Min(numChannels * (1 << i), numChannels * 8);
            int outChannels = i == 0 ? numChannels : Math.Min(numChannels * (1 << (i - 1)), numChannels * 8);
            _decoderBlocks[numBlocks - 1 - i] = new UNetUpBlock<T>(skipChannels, outChannels);
        }

        // Final convolution: numChannels → 1 (per-pixel prediction).
        _convLast = new ConvolutionalLayer<T>(
            outputDepth: 1,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        RegisterSubLayer(_convFirst);
        RegisterSubLayer(_convLast);
        foreach (var block in _encoderBlocks) RegisterSubLayer(block);
        foreach (var block in _decoderBlocks) RegisterSubLayer(block);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves spatial dims and per-input channel count, then drives
    /// each sub-layer's lazy resolution along the encoder→decoder spatial
    /// pyramid so weights are allocated up front.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inC, inH, inW;
        if (s.Length == 3) { inC = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inC = s[1]; inH = s[2]; inW = s[3]; }
        else throw new ArgumentException(
            $"UNetDiscriminator requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
            nameof(input));

        // U-Net pyramid contract: each encoder stage halves H/W (stride 2),
        // each decoder stage doubles them (×2 upsample). After N down-then-
        // up passes we must land on the same spatial dims to align with the
        // skip connections — that requires H and W divisible by 2^numBlocks.
        // Reject upfront with a clear message instead of producing
        // shape-mismatched skip-add later inside Forward.
        int divisor = 1 << _numBlocks;
        if (inH % divisor != 0 || inW % divisor != 0)
            throw new ArgumentException(
                $"UNetDiscriminator with numBlocks={_numBlocks} requires input H/W " +
                $"divisible by 2^{_numBlocks}={divisor}; got [{inH}, {inW}]. " +
                "Resize or pad the input so the encoder/decoder pyramid aligns.",
                nameof(input));

        _convFirst.ResolveFromShape(new[] { inC, inH, inW });
        _convFirst.SetTrainingMode(IsTrainingMode);

        int currentChannels = _numChannels;
        int currentH = inH;
        int currentW = inW;
        for (int i = 0; i < _numBlocks; i++)
        {
            _encoderBlocks[i].ResolveFromShape(new[] { currentChannels, currentH, currentW });
            _encoderBlocks[i].SetTrainingMode(IsTrainingMode);
            int outChannels = Math.Min(currentChannels * 2, _numChannels * 8);
            currentChannels = outChannels;
            currentH = (currentH + 1) / 2;
            currentW = (currentW + 1) / 2;
        }

        for (int i = 0; i < _numBlocks; i++)
        {
            _decoderBlocks[i].ResolveFromShape(new[] { currentChannels, currentH, currentW });
            _decoderBlocks[i].SetTrainingMode(IsTrainingMode);
            int decoderIndex = _numBlocks - 1 - i;
            int outChannels = decoderIndex == 0
                ? _numChannels
                : Math.Min(_numChannels * (1 << (decoderIndex - 1)), _numChannels * 8);
            currentChannels = outChannels;
            currentH *= 2;
            currentW *= 2;
        }

        _convLast.ResolveFromShape(new[] { currentChannels, currentH, currentW });
        _convLast.SetTrainingMode(IsTrainingMode);

        ResolveShapes(
            new[] { inC, inH, inW },
            new[] { 1, inH, inW });

        // Replay any Deserialize-buffered parameters now that block shapes are resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int offset = 0;
            offset = SetLayerParams(_convFirst, pending, offset);
            foreach (var block in _encoderBlocks)
                offset = SetLayerParams(block, pending, offset);
            foreach (var block in _decoderBlocks)
                offset = SetLayerParams(block, pending, offset);
            SetLayerParams(_convLast, pending, offset);
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)

        // Initial conv + activation
        var x = _convFirst.Forward(input);
        x = ApplyLeakyReLU(x);

        // Encoder path - store skip connections
        _skipConnections = new Tensor<T>[_numBlocks];
        for (int i = 0; i < _numBlocks; i++)
        {
            _skipConnections[i] = x; // Store before downsampling
            x = _encoderBlocks[i].Forward(x);
        }

        // Decoder path - use skip connections
        for (int i = 0; i < _numBlocks; i++)
        {
            int skipIdx = _numBlocks - 1 - i;
            x = _decoderBlocks[i].Forward(x, _skipConnections[skipIdx]);
        }

        // Final conv
        x = _convLast.Forward(x);

        return x;
    }

    #endregion

    #region Backward Pass

    #endregion

    #region Helper Methods

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        return Engine.LeakyReLU(input, _leakyReLU.Alpha);
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = TensorAllocator.Rent<T>(gradient._shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data.Span[i] = NumOps.Multiply(
                gradient.Data.Span[i],
                _leakyReLU.Derivative(forwardInput.Data.Span[i]));
        }
        return output;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _convFirst.UpdateParameters(learningRate);
        foreach (var block in _encoderBlocks)
        {
            block.UpdateParameters(learningRate);
        }
        foreach (var block in _decoderBlocks)
        {
            block.UpdateParameters(learningRate);
        }
        _convLast.UpdateParameters(learningRate);
    }

    [Scratch]
    private Vector<T>? _pendingParameters;

    private static void AddParamsToList(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    private static int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.GetParameters().Length;
        layer.SetParameters(parameters.SubVector(offset, count));
        return offset + count;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _skipConnections = null;

        _convFirst.ResetState();
        foreach (var block in _encoderBlocks)
        {
            block.ResetState();
        }
        foreach (var block in _decoderBlocks)
        {
            block.ResetState();
        }
        _convLast.ResetState();
    }

    #endregion


}

#region Helper Blocks

/// <summary>
/// Convolutional block for U-Net encoder with optional downsampling.
/// </summary>
// Rank 3 [C,H,W] or rank 4 [B,C,H,W] - the two forms OnFirstForward names explicitly, so one
// BatchOptional layout covers both.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class UNetConvBlock<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Restates <c>OnFirstForward</c>'s own arithmetic:
    /// <c>ResolveShapes([inC, inH, inW], [_outChannels, outH, outW])</c> where
    /// <c>outH = _downsample ? (inH + 1) / 2 : inH</c>. Channels become the configured width; the
    /// spatial axes either halve or pass through, depending on how this block was constructed.
    /// </para>
    /// <para>
    /// <c>(n + 1) / 2</c> in integer arithmetic is <c>ceil(n / 2)</c>, and that is exactly
    /// <c>Window(kernel: 1, stride: 2, padding: 0)</c>: <c>floor((n - 1) / 2) + 1</c>. Reaching for
    /// <see cref="AxisRelation.Scaled"/> would have been wrong - it refuses uneven division, so an
    /// odd height would resolve to nothing even though the layer handles it fine.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3 && inputRank != 4) return null;

        AxisRelation Spatial(TensorAxis axis) => _downsample
            ? AxisRelation.Window(axis, kernel: 1, stride: 2, padding: 0)
            : AxisRelation.Same(axis);

        var axes = new List<OutputAxisContract>(inputRank);
        if (inputRank == 4)
        {
            axes.Add(new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)));
        }

        axes.Add(new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)));
        axes.Add(new OutputAxisContract(TensorAxis.Height, Spatial(TensorAxis.Height)));
        axes.Add(new OutputAxisContract(TensorAxis.Width, Spatial(TensorAxis.Width)));
        return axes;
    }

    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly LeakyReLUActivation<T> _leakyReLU;
    private readonly bool _downsample;

    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _lastInput;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _conv1Output;      // After LeakyReLU (input to conv2)
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _conv1RawOutput;   // Before LeakyReLU (for backward)
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _conv2RawOutput;   // Before LeakyReLU (for backward)

    private readonly int _outChannels;

    public UNetConvBlock(int outChannels, bool downsample)
        : base([-1, -1, -1], [outChannels, -1, -1])
    {
        _outChannels = outChannels;
        _downsample = downsample;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        // First conv (with optional stride for downsampling)
        _conv1 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: downsample ? 2 : 1,
            padding: 1,
            activationFunction: null);

        // Second conv (always stride 1)
        _conv2 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // No manual RegisterSubLayer here — the source generator's
        // EnsureSubLayersRegistered (called from EnsureInitialized) auto-
        // discovers _conv1/_conv2. Manual registration would double-count
        // the convs in ParameterCount.
    }

    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inC, inH, inW;
        if (s.Length == 3) { inC = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inC = s[1]; inH = s[2]; inW = s[3]; }
        else throw new ArgumentException(
            $"UNetConvBlock requires rank-3 or rank-4 input; got rank {s.Length}.", nameof(input));

        int outH = _downsample ? (inH + 1) / 2 : inH;
        int outW = _downsample ? (inW + 1) / 2 : inW;
        _conv1.ResolveFromShape(new[] { inC, inH, inW });
        _conv2.ResolveFromShape(new[] { _outChannels, outH, outW });
        _conv1.SetTrainingMode(IsTrainingMode);
        _conv2.SetTrainingMode(IsTrainingMode);

        ResolveShapes(
            new[] { inC, inH, inW },
            new[] { _outChannels, outH, outW });

        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int count1 = _conv1.GetParameters().Length;
            _conv1.SetParameters(pending.SubVector(0, count1));
            _conv2.SetParameters(pending.SubVector(count1, pending.Length - count1));
        }
    }

    public override bool SupportsTraining => true;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        // #1668: chained per-stage outputs → locals; cache to fields only for backward.
        bool cacheBwd = ShouldCacheForBackward;
        _lastInput = cacheBwd ? input : null;

        // Conv1 + LeakyReLU
        var conv1Raw = _conv1.Forward(input);
        _conv1RawOutput = cacheBwd ? conv1Raw : null;
        var conv1Out = ApplyLeakyReLU(conv1Raw);
        _conv1Output = cacheBwd ? conv1Out : null;

        // Conv2 + LeakyReLU
        var conv2Raw = _conv2.Forward(conv1Out);
        _conv2RawOutput = cacheBwd ? conv2Raw : null;
        var output = ApplyLeakyReLU(conv2Raw);

        return output;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        return Engine.LeakyReLU(input, _leakyReLU.Alpha);
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = TensorAllocator.Rent<T>(gradient._shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data.Span[i] = NumOps.Multiply(
                gradient.Data.Span[i],
                _leakyReLU.Derivative(forwardInput.Data.Span[i]));
        }
        return output;
    }

    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
    }

    private Vector<T>? _pendingParameters;

    public override void ResetState()
    {
        _lastInput = null;
        _conv1Output = null;
        _conv1RawOutput = null;
        _conv2RawOutput = null;
        _conv1.ResetState();
        _conv2.ResetState();
    }


}

/// <summary>
/// Upsampling block for U-Net decoder with skip connection concatenation.
/// </summary>
// Rank 3 [C,H,W] or rank 4 [B,C,H,W] - the two forms OnFirstForward names explicitly, so one
// BatchOptional layout covers both.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class UNetUpBlock<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Restates <c>OnFirstForward</c>: <c>ResolveShapes([inC, inH, inW], [_outChannels, upH, upW])</c>
    /// with <c>upH = inH * 2</c>, the bilinear <c>UpsamplingLayer(scaleFactor: 2)</c>. Both convolutions
    /// are 3x3 stride 1 padding 1, so they preserve the upsampled extent and only the channel count
    /// changes.
    /// </para>
    /// <para>
    /// <see cref="AxisRelation.Scaled"/> is correct here where it was wrong for the encoder block:
    /// doubling is exact for every input, so there is no uneven division to refuse. The skip
    /// connection affects the CHANNEL count feeding conv1 (<c>inC + _skipChannels</c>) but not the
    /// output, which is pinned to <c>_outChannels</c> - so this stays a single-input contract.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3 && inputRank != 4) return null;

        var axes = new List<OutputAxisContract>(inputRank);
        if (inputRank == 4)
        {
            axes.Add(new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)));
        }

        axes.Add(new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)));
        axes.Add(new OutputAxisContract(
            TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, 2, 1)));
        axes.Add(new OutputAxisContract(
            TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, 2, 1)));
        return axes;
    }

    private readonly UpsamplingLayer<T> _upsample;
    private readonly ConvolutionalLayer<T> _conv1;
    private readonly ConvolutionalLayer<T> _conv2;
    private readonly LeakyReLUActivation<T> _leakyReLU;
    private readonly int _skipChannels;

    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastSkip;
    private Tensor<T>? _upsampledInput;
    private Tensor<T>? _concatenated;
    private Tensor<T>? _conv1Output;

    private readonly int _outChannels;

    public UNetUpBlock(int skipChannels, int outChannels)
        : base([-1, -1, -1], [outChannels, -1, -1])
    {
        _skipChannels = skipChannels;
        _outChannels = outChannels;
        _leakyReLU = new LeakyReLUActivation<T>(0.2);

        // Bilinear upsampling
        _upsample = new UpsamplingLayer<T>(scaleFactor: 2);

        // Conv after concatenation (inChannels + skipChannels -> outChannels)
        _conv1 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        _conv2 = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // No manual RegisterSubLayer — the source generator's
        // EnsureSubLayersRegistered (called from EnsureInitialized) auto-
        // discovers _upsample/_conv1/_conv2.
    }

    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inC, inH, inW;
        if (s.Length == 3) { inC = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inC = s[1]; inH = s[2]; inW = s[3]; }
        else throw new ArgumentException(
            $"UNetUpBlock requires rank-3 or rank-4 input; got rank {s.Length}.", nameof(input));

        int upH = inH * 2;
        int upW = inW * 2;
        _upsample.ResolveFromShape(new[] { inC, inH, inW });
        // After concat with skip the channel count is inC + skipChannels.
        _conv1.ResolveFromShape(new[] { inC + _skipChannels, upH, upW });
        _conv2.ResolveFromShape(new[] { _outChannels, upH, upW });
        _upsample.SetTrainingMode(IsTrainingMode);
        _conv1.SetTrainingMode(IsTrainingMode);
        _conv2.SetTrainingMode(IsTrainingMode);

        ResolveShapes(
            new[] { inC, inH, inW },
            new[] { _outChannels, upH, upW });

        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int count1 = _conv1.GetParameters().Length;
            _conv1.SetParameters(pending.SubVector(0, count1));
            _conv2.SetParameters(pending.SubVector(count1, pending.Length - count1));
        }
    }

    public override bool SupportsTraining => true;

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        return Forward(input, null);
    }

    public Tensor<T> Forward(Tensor<T> input, Tensor<T>? skip)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        // #1668: gate backward caches; _upsampledInput/_concatenated are read below
        // (chained), so forward uses locals and the fields cache only for backward.
        bool cacheBwd = ShouldCacheForBackward;
        _lastInput = cacheBwd ? input : null;
        _lastSkip = cacheBwd ? skip : null;

        // Upsample
        var upsampledInput = _upsample.Forward(input);
        _upsampledInput = cacheBwd ? upsampledInput : null;

        // Concatenate with skip connection
        Tensor<T> x;
        if (skip != null)
        {
            var concatenated = ConcatenateChannels(upsampledInput, skip);
            _concatenated = cacheBwd ? concatenated : null;
            x = concatenated;
        }
        else
        {
            _concatenated = null;
            x = upsampledInput;
        }

        // Conv1 + LeakyReLU
        x = _conv1.Forward(x);
        x = ApplyLeakyReLU(x);
        _conv1Output = cacheBwd ? x : null;

        // Conv2 + LeakyReLU
        x = _conv2.Forward(x);
        x = ApplyLeakyReLU(x);

        return x;
    }

    public (Tensor<T> mainGrad, Tensor<T> skipGrad) BackwardWithSkip(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _upsampledInput == null || _conv1Output == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through conv2 + LeakyReLU
        var grad = BackwardLeakyReLU(_conv2.Forward(_conv1Output), outputGradient);

        // Backward through conv1 + LeakyReLU
        grad = BackwardLeakyReLU(_conv1.Forward(_concatenated ?? _upsampledInput), grad);

        Tensor<T> skipGrad;
        Tensor<T> upsampleGrad;

        // Split gradient if skip connection was used
        if (_concatenated != null && _lastSkip != null)
        {
            // Handle both 3D [C, H, W] and 4D [N, C, H, W] tensors
            bool has4D = _upsampledInput.Shape.Length == 4;
            int mainChannels = has4D ? _upsampledInput.Shape[1] : _upsampledInput.Shape[0];
            int skipChannels = has4D ? _lastSkip.Shape[1] : _lastSkip.Shape[0];
            (upsampleGrad, skipGrad) = SplitGradient(grad, mainChannels, skipChannels);
        }
        else
        {
            upsampleGrad = grad;
            skipGrad = new Tensor<T>(_lastSkip?.Shape.ToArray() ?? new[] { 1 });
        }

        // Backward removed — tape handles gradients
        var inputGrad = new Tensor<T>(_lastInput?.Shape.ToArray() ?? new[] { 1 });
        return (inputGrad, skipGrad);
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Channel axis is 0 for 3D [C, H, W] and 1 for 4D [N, C, H, W]
        int channelAxis = a.Shape.Length == 4 ? 1 : 0;
        return Engine.TensorConcatenate([a, b], axis: channelAxis);
    }

    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        // Handle both 3D [C, H, W] and 4D [N, C, H, W] tensors
        bool has4D = grad.Shape.Length == 4;
        int batch = has4D ? grad.Shape[0] : 1;
        int height = has4D ? grad.Shape[2] : grad.Shape[1];
        int width = has4D ? grad.Shape[3] : grad.Shape[2];
        int spatialSize = height * width;

        var firstShape = has4D
            ? new int[] { batch, firstChannels, height, width }
            : new int[] { firstChannels, height, width };
        var secondShape = has4D
            ? new int[] { batch, secondChannels, height, width }
            : new int[] { secondChannels, height, width };

        var first = new Tensor<T>(firstShape);
        var second = new Tensor<T>(secondShape);
        int totalChannels = firstChannels + secondChannels;

        for (int n = 0; n < batch; n++)
        {
            int batchOffsetGrad = n * totalChannels * spatialSize;
            int batchOffsetFirst = n * firstChannels * spatialSize;
            int batchOffsetSecond = n * secondChannels * spatialSize;

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    first.Data.Span[batchOffsetFirst + c * spatialSize + hw] = grad.Data.Span[batchOffsetGrad + c * spatialSize + hw];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    second.Data.Span[batchOffsetSecond + c * spatialSize + hw] = grad.Data.Span[batchOffsetGrad + (firstChannels + c) * spatialSize + hw];
                }
            }
        }

        return (first, second);
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        return Engine.LeakyReLU(input, _leakyReLU.Alpha);
    }

    private Tensor<T> BackwardLeakyReLU(Tensor<T> forwardInput, Tensor<T> gradient)
    {
        var output = TensorAllocator.Rent<T>(gradient._shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data.Span[i] = NumOps.Multiply(
                gradient.Data.Span[i],
                _leakyReLU.Derivative(forwardInput.Data.Span[i]));
        }
        return output;
    }

    public override void UpdateParameters(T learningRate)
    {
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
    }

    private Vector<T>? _pendingParameters;

    public override void ResetState()
    {
        _lastInput = null;
        _lastSkip = null;
        _upsampledInput = null;
        _concatenated = null;
        _conv1Output = null;
        _upsample.ResetState();
        _conv1.ResetState();
        _conv2.ResetState();
    }



}

#endregion
