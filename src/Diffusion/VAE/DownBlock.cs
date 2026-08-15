using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Downsampling block for VAE encoder with multiple ResBlocks and strided convolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements a downsampling block following the Stable Diffusion VAE architecture:
/// - Multiple VAEResBlocks to process features at the current resolution
/// - Strided convolution (stride=2) to reduce spatial dimensions by half
/// </para>
/// <para>
/// <b>For Beginners:</b> A DownBlock is like a compression stage in an encoder.
///
/// What it does:
/// 1. Processes the input through multiple residual blocks (learning features)
/// 2. Reduces spatial size by half using strided convolution (compression)
///
/// Example: 64x64 input -> 32x32 output (spatial dimensions halved)
///
/// Why use strided convolution instead of pooling?
/// - Strided conv is learnable (the network decides how to downsample)
/// - Max/Avg pooling has fixed behavior that may discard useful information
/// - Strided conv is the standard in modern generative models like VAEs and diffusion
///
/// Structure:
/// ```
///     input [B, C_in, H, W]
///           │
///           ├─→ ResBlock → ResBlock → ... (numLayers blocks)
///           │
///           ↓
///     [B, C_out, H, W]
///           │
///           ├─→ Conv3x3 (stride=2) ─→ downsample
///           │
///           ↓
///     output [B, C_out, H/2, W/2]
/// ```
/// </para>
/// </remarks>
// Roles from this block's own diagram above - input [B, C_in, H, W], output [B, C_out, H/2, W/2] -
// and from what it is made of: every stage is a VAEResBlock, which declares rank 4 only, so this
// block cannot accept less. Batch is NOT optional for the same reason it is not on VAEResBlock:
// nothing here establishes that the inner GroupNorm and 3x3 convolutions take an unbatched [C,H,W].
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class DownBlock<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Channels are <c>Fixed(_outChannels)</c>: the first <c>VAEResBlock</c> is constructed as
    /// <c>(inChannels, outChannels, ...)</c> and every later one as <c>(outChannels, outChannels, ...)</c>,
    /// so the width is settled by the constructor argument and never by the input.
    /// </para>
    /// <para>
    /// The spatial relation is CONDITIONAL, and that is the whole reason this is hand-written rather
    /// than probed: <c>ForwardTraced</c> applies <c>_downsample</c> only <c>if (_hasDownsample)</c>.
    /// The res blocks are extent-preserving, so with downsampling off the block is exactly
    /// <c>Same</c> on both axes - which is the configuration the last encoder block uses. With it on,
    /// the relation is the <c>_downsample</c> convolution's own arguments,
    /// <c>kernelSize: 3, stride: 2, padding: 1</c>, written as a <c>Window</c> rather than as
    /// <c>Scaled(axis, 1, 2)</c>: the two agree on even extents but the window is what the code
    /// actually computes, and it stays right on an odd one, where the halving would refuse.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 || _outChannels <= 0) return null;

        // Mirrors the constructor's own comment on _downsample:
        // "kernel=3, stride=2, padding=1 -> output_size = (input_size + 2*1 - 3) / 2 + 1 = input_size / 2".
        AxisRelation Spatial(TensorAxis axis) => _hasDownsample
            ? AxisRelation.Window(axis, kernel: 3, stride: 2, padding: 1)
            : AxisRelation.Same(axis);

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outChannels)),
            new OutputAxisContract(TensorAxis.Height, Spatial(TensorAxis.Height)),
            new OutputAxisContract(TensorAxis.Width, Spatial(TensorAxis.Width)),
        };
    }

    /// <summary>
    /// Residual blocks in this down block.
    /// </summary>
    private readonly VAEResBlock<T>[] _resBlocks;

    /// <summary>
    /// Strided convolution for downsampling.
    /// </summary>
    private readonly ConvolutionalLayer<T> _downsample;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inChannels;

    /// <summary>
    /// Number of output channels.
    /// </summary>
    private readonly int _outChannels;

    /// <summary>
    /// Number of residual blocks.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Number of groups for GroupNorm in ResBlocks.
    /// </summary>
    private readonly int _numGroups;

    /// <summary>
    /// Spatial size at input.
    /// </summary>
    private readonly int _inputSpatialSize;

    /// <summary>
    /// Whether this block includes downsampling (false for the last encoder block).
    /// </summary>
    private readonly bool _hasDownsample;

    /// <summary>
    /// Cached inputs and intermediate values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private readonly Tensor<T>?[] _resBlockOutputs;
    private Tensor<T>? _preDownsampleOutput;

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
    /// Gets the number of residual blocks.
    /// </summary>
    public int NumLayers => _numLayers;

    /// <summary>
    /// Gets whether this block performs downsampling.
    /// </summary>
    public bool HasDownsample => _hasDownsample;

    /// <summary>
    /// Initializes a new instance of the DownBlock class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="numLayers">Number of residual blocks (default: 2).</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="inputSpatialSize">Spatial dimensions at input (default: 64).</param>
    /// <param name="hasDownsample">Whether to include downsampling (default: true).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a downsampling block for the VAE encoder.
    ///
    /// Parameters explained:
    /// - inChannels/outChannels: Feature depth before/after this block
    /// - numLayers: More layers = more feature processing but slower
    /// - hasDownsample: Set to false for the last encoder block to keep resolution
    ///
    /// Typical usage in an encoder:
    /// - Block 1: 128 -> 128, downsample (64x64 -> 32x32)
    /// - Block 2: 128 -> 256, downsample (32x32 -> 16x16)
    /// - Block 3: 256 -> 512, downsample (16x16 -> 8x8)
    /// - Block 4: 512 -> 512, no downsample (8x8 -> 8x8)
    /// </para>
    /// </remarks>
    public DownBlock(
        int inChannels,
        int outChannels,
        int numLayers = 2,
        int numGroups = 32,
        int inputSpatialSize = 64,
        bool hasDownsample = true)
        : base(
            CalculateInputShape(inChannels, inputSpatialSize),
            CalculateOutputShape(outChannels, inputSpatialSize, hasDownsample))
    {
        if (inChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inChannels), "Input channels must be positive.");
        if (outChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outChannels), "Output channels must be positive.");
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be positive.");

        _inChannels = inChannels;
        _outChannels = outChannels;
        _numLayers = numLayers;
        _numGroups = CalculateValidGroups(numGroups, inChannels, outChannels);
        _inputSpatialSize = inputSpatialSize;
        _hasDownsample = hasDownsample;
        _resBlockOutputs = new Tensor<T>?[numLayers];

        // Create residual blocks
        _resBlocks = new VAEResBlock<T>[numLayers];

        // First block handles channel change (inChannels -> outChannels)
        _resBlocks[0] = new VAEResBlock<T>(inChannels, outChannels, _numGroups, inputSpatialSize);

        // Remaining blocks maintain outChannels
        for (int i = 1; i < numLayers; i++)
        {
            _resBlocks[i] = new VAEResBlock<T>(outChannels, outChannels, _numGroups, inputSpatialSize);
        }

        // Strided convolution for 2x downsampling
        // kernel=3, stride=2, padding=1 -> output_size = (input_size + 2*1 - 3) / 2 + 1 = input_size / 2
        _downsample = new ConvolutionalLayer<T>(
            outputDepth: outChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    private static int[] CalculateOutputShape(int channels, int inputSpatialSize, bool hasDownsample)
    {
        int outputSpatialSize = hasDownsample ? inputSpatialSize / 2 : inputSpatialSize;
        return new[] { channels, outputSpatialSize, outputSpatialSize };
    }

    private static int CalculateValidGroups(int preferredGroups, int inChannels, int outChannels)
    {
        int groups = Math.Min(preferredGroups, Math.Min(inChannels, outChannels));
        while (groups > 1 && (inChannels % groups != 0 || outChannels % groups != 0))
        {
            groups--;
        }
        return Math.Max(1, groups);
    }

    /// <summary>
    /// Performs the forward pass through the down block.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, inChannels, H, W].</param>
    /// <returns>Output tensor with shape [batch, outChannels, H/2, W/2] if hasDownsample, else [batch, outChannels, H, W].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _lastInput = input;
        var x = input;

        // Process through residual blocks
        for (int i = 0; i < _numLayers; i++)
        {
            x = _resBlocks[i].Forward(x);
            _resBlockOutputs[i] = x;
        }

        _preDownsampleOutput = x;

        // Apply downsampling if enabled
        if (_hasDownsample)
        {
            x = _downsample.Forward(x);
        }

        return x;
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    /// <param name="learningRate">The learning rate for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        foreach (var block in _resBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        if (_hasDownsample)
        {
            _downsample.UpdateParameters(learningRate);
        }
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
        _preDownsampleOutput = null;

        for (int i = 0; i < _resBlockOutputs.Length; i++)
        {
            _resBlockOutputs[i] = null;
        }

        foreach (var block in _resBlocks)
        {
            block.ResetState();
        }

        _downsample.ResetState();
    }


    /// <summary>
    /// Gets the residual blocks for external access (e.g., for skip connections in UNet).
    /// </summary>
    /// <returns>Array of residual blocks.</returns>
    public IReadOnlyList<VAEResBlock<T>> GetResBlocks() => _resBlocks;
}
