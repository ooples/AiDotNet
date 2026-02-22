using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Upsampling block for VAE decoder with transposed convolution and multiple ResBlocks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements an upsampling block following the Stable Diffusion VAE architecture:
/// - Transposed convolution (deconvolution) to increase spatial dimensions by 2x
/// - Multiple VAEResBlocks to process features at the upsampled resolution
/// </para>
/// <para>
/// <b>For Beginners:</b> An UpBlock is like a decompression stage in a decoder.
///
/// What it does:
/// 1. Increases spatial size by 2x using transposed convolution (decompression)
/// 2. Processes the upsampled features through multiple residual blocks
///
/// Example: 8x8 input -> 16x16 output (spatial dimensions doubled)
///
/// Why use transposed convolution instead of simple interpolation?
/// - Transposed conv is learnable (the network decides how to upsample)
/// - Simple interpolation (bilinear, nearest) has fixed behavior
/// - Learnable upsampling can generate sharper details
///
/// Structure:
/// ```
///     input [B, C_in, H, W]
///           │
///           ├─→ ConvTranspose (stride=2) ─→ upsample
///           │
///           ↓
///     [B, C_out, 2*H, 2*W]
///           │
///           ├─→ ResBlock → ResBlock → ... (numLayers blocks)
///           │
///           ↓
///     output [B, C_out, 2*H, 2*W]
/// ```
/// </para>
/// </remarks>
public class UpBlock<T> : LayerBase<T>
{
    /// <summary>
    /// Transposed convolution for upsampling.
    /// </summary>
    private readonly DeconvolutionalLayer<T>? _upsample;

    /// <summary>
    /// Residual blocks in this up block.
    /// </summary>
    private readonly VAEResBlock<T>[] _resBlocks;

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
    /// Spatial size at input (before upsampling).
    /// </summary>
    private readonly int _inputSpatialSize;

    /// <summary>
    /// Spatial size at output (after upsampling).
    /// </summary>
    private readonly int _outputSpatialSize;

    /// <summary>
    /// Whether this block includes upsampling (false for the first decoder block).
    /// </summary>
    private readonly bool _hasUpsample;

    /// <summary>
    /// Cached inputs and intermediate values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _postUpsampleOutput;
    private readonly Tensor<T>?[] _resBlockOutputs;

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
    /// Gets whether this block performs upsampling.
    /// </summary>
    public bool HasUpsample => _hasUpsample;

    /// <summary>
    /// Initializes a new instance of the UpBlock class.
    /// </summary>
    /// <param name="inChannels">Number of input channels.</param>
    /// <param name="outChannels">Number of output channels.</param>
    /// <param name="numLayers">Number of residual blocks (default: 2).</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="inputSpatialSize">Spatial dimensions at input (default: 8).</param>
    /// <param name="hasUpsample">Whether to include upsampling (default: true).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create an upsampling block for the VAE decoder.
    ///
    /// Parameters explained:
    /// - inChannels/outChannels: Feature depth before/after this block
    /// - numLayers: More layers = more feature processing but slower
    /// - hasUpsample: Set to false for the first decoder block to keep resolution
    ///
    /// Typical usage in a decoder (mirror of encoder):
    /// - Block 1: 512 -> 512, no upsample (8x8 -> 8x8)
    /// - Block 2: 512 -> 256, upsample (8x8 -> 16x16)
    /// - Block 3: 256 -> 128, upsample (16x16 -> 32x32)
    /// - Block 4: 128 -> 128, upsample (32x32 -> 64x64)
    /// </para>
    /// </remarks>
    public UpBlock(
        int inChannels,
        int outChannels,
        int numLayers = 2,
        int numGroups = 32,
        int inputSpatialSize = 8,
        bool hasUpsample = true)
        : base(
            CalculateInputShape(inChannels, inputSpatialSize),
            CalculateOutputShape(outChannels, inputSpatialSize, hasUpsample))
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
        _outputSpatialSize = hasUpsample ? inputSpatialSize * 2 : inputSpatialSize;
        _hasUpsample = hasUpsample;
        _resBlockOutputs = new Tensor<T>?[numLayers];

        // Transposed convolution for 2x upsampling
        // kernel=4, stride=2, padding=1 -> output_size = (input_size - 1) * 2 - 2*1 + 4 = 2*input_size
        if (hasUpsample)
        {
            _upsample = new DeconvolutionalLayer<T>(
                inputShape: new[] { 1, inChannels, inputSpatialSize, inputSpatialSize },
                outputDepth: outChannels,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: new IdentityActivation<T>());
        }

        // Create residual blocks
        _resBlocks = new VAEResBlock<T>[numLayers];

        // Channel count after upsampling (or just inChannels if no upsample)
        int postUpsampleChannels = hasUpsample ? outChannels : inChannels;

        // First block may handle channel change if no upsample
        if (!hasUpsample && inChannels != outChannels)
        {
            _resBlocks[0] = new VAEResBlock<T>(inChannels, outChannels, _numGroups, _outputSpatialSize);
            postUpsampleChannels = outChannels;
        }
        else
        {
            _resBlocks[0] = new VAEResBlock<T>(postUpsampleChannels, outChannels, _numGroups, _outputSpatialSize);
        }

        // Remaining blocks maintain outChannels
        for (int i = 1; i < numLayers; i++)
        {
            _resBlocks[i] = new VAEResBlock<T>(outChannels, outChannels, _numGroups, _outputSpatialSize);
        }
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    private static int[] CalculateOutputShape(int channels, int inputSpatialSize, bool hasUpsample)
    {
        int outputSpatialSize = hasUpsample ? inputSpatialSize * 2 : inputSpatialSize;
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
    /// Performs the forward pass through the up block.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch, inChannels, H, W].</param>
    /// <returns>Output tensor with shape [batch, outChannels, 2*H, 2*W] if hasUpsample, else [batch, outChannels, H, W].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        var x = input;

        // Apply upsampling if enabled
        if (_hasUpsample && _upsample != null)
        {
            x = _upsample.Forward(x);
        }

        _postUpsampleOutput = x;

        // Process through residual blocks
        for (int i = 0; i < _numLayers; i++)
        {
            x = _resBlocks[i].Forward(x);
            _resBlockOutputs[i] = x;
        }

        return x;
    }

    /// <summary>
    /// Performs the backward pass through the up block.
    /// </summary>
    /// <param name="outputGradient">Gradient of loss with respect to output.</param>
    /// <returns>Gradient of loss with respect to input.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        var gradient = outputGradient;

        // Backward through residual blocks in reverse order
        for (int i = _numLayers - 1; i >= 0; i--)
        {
            gradient = _resBlocks[i].Backward(gradient);
        }

        // Backward through upsample if present
        if (_hasUpsample && _upsample != null)
        {
            gradient = _upsample.Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    /// <param name="learningRate">The learning rate for the update.</param>
    public override void UpdateParameters(T learningRate)
    {
        _upsample?.UpdateParameters(learningRate);

        foreach (var block in _resBlocks)
        {
            block.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        if (_hasUpsample && _upsample != null)
        {
            AddParameters(paramsList, _upsample.GetParameters());
        }

        foreach (var block in _resBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
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

        if (_hasUpsample && _upsample != null)
        {
            SetLayerParams(_upsample, parameters, ref index);
        }

        foreach (var block in _resBlocks)
        {
            SetLayerParams(block, parameters, ref index);
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
        _postUpsampleOutput = null;

        for (int i = 0; i < _resBlockOutputs.Length; i++)
        {
            _resBlockOutputs[i] = null;
        }

        _upsample?.ResetState();

        foreach (var block in _resBlocks)
        {
            block.ResetState();
        }
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException(
            "UpBlock JIT compilation is not yet implemented. " +
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
        writer.Write(_hasUpsample);

        if (_hasUpsample && _upsample != null)
        {
            _upsample.Serialize(writer);
        }

        foreach (var block in _resBlocks)
        {
            block.Serialize(writer);
        }
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
        var hasUpsample = reader.ReadBoolean();

        if (inChannels != _inChannels || outChannels != _outChannels ||
            numLayers != _numLayers || hasUpsample != _hasUpsample)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch in UpBlock deserialization.");
        }

        if (_hasUpsample && _upsample != null)
        {
            _upsample.Deserialize(reader);
        }

        foreach (var block in _resBlocks)
        {
            block.Deserialize(reader);
        }
    }

    /// <summary>
    /// Gets the residual blocks for external access.
    /// </summary>
    /// <returns>Array of residual blocks.</returns>
    public IReadOnlyList<VAEResBlock<T>> GetResBlocks() => _resBlocks;
}
