using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
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
public class DownBlock<T> : LayerBase<T>
{
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
            inputDepth: outChannels,
            outputDepth: outChannels,
            kernelSize: 3,
            inputHeight: inputSpatialSize,
            inputWidth: inputSpatialSize,
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
    public override Tensor<T> Forward(Tensor<T> input)
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
    /// Performs the backward pass through the down block.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _preDownsampleOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var gradient = outputGradient;

        // Backward through downsample if present
        if (_hasDownsample)
        {
            gradient = _downsample.Backward(gradient);
        }

        // Backward through residual blocks in reverse order
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            gradient = _resBlocks[i].Backward(gradient);
        }

        return gradient;
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

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        foreach (var block in _resBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
        }

        if (_hasDownsample)
        {
            AddParameters(paramsList, _downsample.GetParameters());
        }

        return new Vector<T>(paramsList.ToArray());
    }

    private static void AddParameters(List<T> list, Vector<T> parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
        {
            list.Add(parameters[i]);
        }
    }

    /// <summary>
    /// Sets all trainable parameters from a single vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int index = 0;

        foreach (var block in _resBlocks)
        {
            SetLayerParams(block, parameters, ref index);
        }

        if (_hasDownsample)
        {
            SetLayerParams(_downsample, parameters, ref index);
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "DownBlock JIT compilation is not yet implemented. " +
            "Use the layer in interpreted mode.");
    }

    /// <summary>
    /// Saves the block's state to a binary writer.
    /// </summary>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(_inChannels);
        writer.Write(_outChannels);
        writer.Write(_numLayers);
        writer.Write(_numGroups);
        writer.Write(_inputSpatialSize);
        writer.Write(_hasDownsample);

        foreach (var block in _resBlocks)
        {
            block.Serialize(writer);
        }

        _downsample.Serialize(writer);
    }

    /// <summary>
    /// Loads the block's state from a binary reader.
    /// </summary>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        var inChannels = reader.ReadInt32();
        var outChannels = reader.ReadInt32();
        var numLayers = reader.ReadInt32();
        var numGroups = reader.ReadInt32();
        var inputSpatialSize = reader.ReadInt32();
        var hasDownsample = reader.ReadBoolean();

        if (inChannels != _inChannels || outChannels != _outChannels ||
            numLayers != _numLayers || hasDownsample != _hasDownsample)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch in DownBlock deserialization.");
        }

        foreach (var block in _resBlocks)
        {
            block.Deserialize(reader);
        }

        _downsample.Deserialize(reader);
    }

    /// <summary>
    /// Gets the residual blocks for external access (e.g., for skip connections in UNet).
    /// </summary>
    /// <returns>Array of residual blocks.</returns>
    public IReadOnlyList<VAEResBlock<T>> GetResBlocks() => _resBlocks;
}
