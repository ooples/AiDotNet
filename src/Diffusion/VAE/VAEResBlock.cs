using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Residual block for VAE encoder/decoder with GroupNorm and skip connections.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a proper VAE residual block following the Stable Diffusion VAE architecture:
/// - GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv
/// - Skip connection with optional 1x1 convolution when input/output channels differ
/// </para>
/// <para>
/// <b>For Beginners:</b> A residual block helps the network learn more effectively.
///
/// Think of it like taking notes during a lecture:
/// - The main path (two convolutions) learns new features
/// - The skip connection preserves the original information
/// - Adding them together means you learn the "difference" or "improvement"
///
/// The GroupNorm helps stabilize training by normalizing activations within groups
/// of channels, which works well even with small batch sizes commonly used in
/// image generation tasks.
///
/// Structure:
/// ```
///     input ─────────────────────────────────┐
///       │                                    │
///       ├─→ GroupNorm → SiLU → Conv3x3 ─→ h  │ (skip connection)
///       │                                    │
///       │        ↓                           │
///       │                                    │
///       │   GroupNorm → SiLU → Conv3x3 ─→ h  │
///       │                                    │
///       │        ↓                           ↓
///       │                                 [1x1 Conv if channels differ]
///       │        ↓                           ↓
///       └────────────────→ (+) ←─────────────┘
///                          │
///                       output
/// ```
/// </para>
/// </remarks>
// Roles from this block's own ForwardTraced doc - "Input tensor with shape [batch, channels, height,
// width]" / "Output tensor with shape [batch, outChannels, height, width]". Batch is NOT marked
// optional: nothing in this file establishes that the inner GroupNorm and 3x3 convolutions accept an
// unbatched [C,H,W], and the class carries no [LayerProperty(TestInputShape = ...)] pinning a lower
// rank, so declaring rank 3 would be a claim made on no evidence.
// OutputAxesFor below is HAND-WRITTEN: the channel count is a constructor argument.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class VAEResBlock<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Channels are the ONLY axis this block moves, and they move to a value the caller chose, so the
    /// relation is <c>Fixed(_outChannels)</c> read off the constructor argument rather than anything
    /// derived from the input. Both convolutions are built with
    /// <c>outputDepth: outChannels</c>, and the residual add at the end of <c>ForwardTraced</c>
    /// (<c>Engine.TensorAdd(_conv2Output, _skipOutput)</c>) forces the skip branch to agree - which is
    /// exactly why the 1x1 <c>_skipConv</c> exists at all, and only when <c>inChannels != outChannels</c>.
    /// </para>
    /// <para>
    /// The two spatial axes are <c>Same</c> because every convolution here is deliberately
    /// extent-preserving: <c>kernelSize: 3, stride: 1, padding: 1</c> on the main path (the constructor
    /// comment reads "3x3 with padding=1 preserves spatial dimensions") and <c>kernelSize: 1, stride: 1,
    /// padding: 0</c> on the skip. A <c>Window</c> relation would be technically correct and useless -
    /// it evaluates to the identity for both configurations, and stating it as <c>Same</c> is what the
    /// block actually guarantees, since the residual add would throw otherwise.
    /// </para>
    /// <para>
    /// The <c>spatialSize</c> constructor argument is NOT a claim about the input. It only sizes the
    /// placeholder shapes handed to the base constructor; no forward-path code reads it, and feeding a
    /// different resolution works.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 || _outChannels <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width)),
        };
    }

    /// <summary>
    /// First GroupNorm layer.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _norm1;

    /// <summary>
    /// Second GroupNorm layer.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _norm2;

    /// <summary>
    /// First convolution layer.
    /// </summary>
    private readonly ConvolutionalLayer<T> _conv1;

    /// <summary>
    /// Second convolution layer.
    /// </summary>
    private readonly ConvolutionalLayer<T> _conv2;

    /// <summary>
    /// Optional 1x1 convolution for skip connection when channels differ.
    /// </summary>
    private readonly ConvolutionalLayer<T>? _skipConv;

    /// <summary>
    /// SiLU activation function.
    /// </summary>
    private readonly IActivationFunction<T> _silu;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inChannels;

    /// <summary>
    /// Number of output channels.
    /// </summary>
    private readonly int _outChannels;

    /// <summary>
    /// Number of groups for GroupNorm.
    /// </summary>
    private readonly int _numGroups;

    /// <summary>
    /// Cached input from forward pass for backward.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached intermediate values for backward pass.
    /// </summary>
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _norm1Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _silu1Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _conv1Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _norm2Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _silu2Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _conv2Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _skipOutput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels => _inChannels;

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutputChannels => _outChannels;

    /// <summary>
    /// Gets the number of groups for GroupNorm.
    /// </summary>
    public int NumGroups => _numGroups;

    /// <summary>Construction state: the 'spatialSize' the layer was built with.</summary>
    private readonly int _spatialSize;

    /// <summary>
    /// Initializes a new instance of the VAEResBlock class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="spatialSize">Spatial dimensions (height/width) for conv layer setup.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a VAE residual block with the specified channel configuration.
    ///
    /// Typical configurations:
    /// - numGroups=32 for 256+ channels
    /// - numGroups=16 for 128 channels
    /// - numGroups=8 for 64 channels
    ///
    /// The numGroups should evenly divide the channel count for proper normalization.
    /// </para>
    /// </remarks>
    public VAEResBlock(int inChannels, int outChannels, int numGroups = 32, int spatialSize = 32)
        : base(CalculateInputShape(inChannels, spatialSize), CalculateOutputShape(outChannels, spatialSize))
    {
        _spatialSize = spatialSize;
        if (inChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inChannels), "Input channels must be positive.");
        if (outChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outChannels), "Output channels must be positive.");
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        // Adjust numGroups if it doesn't divide evenly
        numGroups = Math.Min(numGroups, Math.Min(inChannels, outChannels));
        while (inChannels % numGroups != 0 || outChannels % numGroups != 0)
        {
            numGroups--;
            if (numGroups <= 0) numGroups = 1;
        }

        _inChannels = inChannels;
        _outChannels = outChannels;
        _numGroups = numGroups;
        _silu = new SiLUActivation<T>();

        // GroupNorm layers
        _norm1 = new GroupNormalizationLayer<T>(numGroups, inChannels);
        _norm2 = new GroupNormalizationLayer<T>(numGroups, outChannels);

        // Convolutional layers (3x3 with padding=1 preserves spatial dimensions)
        _conv1 = ConvolutionalLayer<T>.WithInputDepth(
            inputDepth: inChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _conv2 = ConvolutionalLayer<T>.WithInputDepth(
            inputDepth: outChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Skip connection: 1x1 conv if channels differ
        if (inChannels != outChannels)
        {
            _skipConv = ConvolutionalLayer<T>.WithInputDepth(
                inputDepth: inChannels,
                outputDepth: outChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activationFunction: new IdentityActivation<T>());
        }
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    private static int[] CalculateOutputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    /// <summary>
    /// Performs the forward pass through the residual block.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, channels, height, width].</param>
    /// <returns>Output tensor with shape [batch, outChannels, height, width].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // LOCALS carry the dataflow; the fields are populated only when a manual backward will
        // actually read them (LayerBase.ShouldCacheForBackward is the canonical guard). Assigning
        // them unconditionally kept every stage of every block alive for the whole pass, so a
        // decoder holding ten of these had peak memory O(sum of activations) rather than
        // O(max live set) even during pure inference.
        bool cacheBwd = ShouldCacheForBackward;
        if (cacheBwd) _lastInput = input;

        // Main path: GroupNorm -> SiLU -> Conv -> GroupNorm -> SiLU -> Conv
        var norm1Output = _norm1.Forward(input);
        var silu1Output = ApplySiLU(norm1Output);
        var conv1Output = _conv1.Forward(silu1Output);

        var norm2Output = _norm2.Forward(conv1Output);
        var silu2Output = ApplySiLU(norm2Output);
        var conv2Output = _conv2.Forward(silu2Output);

        // Skip connection
        var skipOutput = _skipConv != null ? _skipConv.Forward(input) : input;

        if (cacheBwd)
        {
            _norm1Output = norm1Output;
            _silu1Output = silu1Output;
            _conv1Output = conv1Output;
            _norm2Output = norm2Output;
            _silu2Output = silu2Output;
            _conv2Output = conv2Output;
            _skipOutput = skipOutput;
        }

        // Add main path and skip connection
        return Engine.TensorAdd(conv2Output, skipOutput);
    }

    /// <summary>
    /// Applies SiLU activation to a tensor.
    /// </summary>
    private Tensor<T> ApplySiLU(Tensor<T> input)
    {
        return Engine.Swish(input);
    }

    /// <summary>
    /// Computes the SiLU derivative for a tensor.
    /// </summary>
    private Tensor<T> ApplySiLUDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var output = new Tensor<T>(input._shape);
        var inputSpan = input.AsSpan();
        var gradSpan = gradient.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = NumOps.Multiply(_silu.Derivative(inputSpan[i]), gradSpan[i]);
        }

        return output;
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    /// <param name="learningRate">The learning rate for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
        _conv1.UpdateParameters(learningRate);
        _conv2.UpdateParameters(learningRate);
        _skipConv?.UpdateParameters(learningRate);
    }

    private static void AddParameters(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    private static void SetLayerParams(ILayer<T> layer, Vector<T> parameters, ref int index)
    {
        var layerParams = layer.GetParameters();
        var newParams = new Vector<T>(layerParams.Length);

        for (int i = 0; i < layerParams.Length && index < parameters.Length; i++)
        {
            newParams[i] = parameters[index++];
        }

        layer.SetParameters(newParams);
    }

    /// <summary>
    /// Resets the internal state of the block.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _norm1Output = null;
        _silu1Output = null;
        _conv1Output = null;
        _norm2Output = null;
        _silu2Output = null;
        _conv2Output = null;
        _skipOutput = null;

        _norm1.ResetState();
        _norm2.ResetState();
        _conv1.ResetState();
        _conv2.ResetState();
        _skipConv?.ResetState();
    }
}
