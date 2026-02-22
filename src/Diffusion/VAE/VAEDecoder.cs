using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Convolutional decoder for VAE that reconstructs images from latent space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the decoder portion of a VAE following the Stable Diffusion architecture:
/// - Post-quant convolution to expand latent channels
/// - Middle blocks at the bottleneck
/// - Multiple UpBlocks with transposed conv upsampling and ResBlocks
/// - Output convolution to produce final image channels
/// </para>
/// <para>
/// <b>For Beginners:</b> The VAE decoder is like an intelligent image decompressor.
///
/// What it does step by step:
/// 1. Takes a compressed latent (e.g., 64x64x4)
/// 2. Post-quant conv: Expands channels (4 -> 512)
/// 3. Middle blocks: Extra processing at the bottleneck
/// 4. UpBlocks: Progressively doubles resolution while decreasing channels
///    - Block 1: 512 channels, 64x64 -> 64x64 (no upsample at start)
///    - Block 2: 512 channels, 64x64 -> 128x128
///    - Block 3: 256 channels, 128x128 -> 256x256
///    - Block 4: 128 channels, 256x256 -> 512x512
/// 5. Output: Produces 3-channel RGB image with tanh activation
///
/// The result is a high-resolution image reconstructed from the compressed latent.
/// </para>
/// </remarks>
public class VAEDecoder<T> : LayerBase<T>
{
    /// <summary>
    /// Post-quant convolution to expand latent channels.
    /// </summary>
    private readonly ConvolutionalLayer<T> _postQuantConv;

    /// <summary>
    /// Convolution to expand latent to decoder channels.
    /// </summary>
    private readonly ConvolutionalLayer<T> _inputConv;

    /// <summary>
    /// Middle residual blocks at the bottleneck.
    /// </summary>
    private readonly VAEResBlock<T>[] _midBlocks;

    /// <summary>
    /// Upsampling blocks.
    /// </summary>
    private readonly UpBlock<T>[] _upBlocks;

    /// <summary>
    /// Group normalization before output.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _normOut;

    /// <summary>
    /// Output convolution to image channels.
    /// </summary>
    private readonly ConvolutionalLayer<T> _outputConv;

    /// <summary>
    /// SiLU activation function.
    /// </summary>
    private readonly IActivationFunction<T> _silu;

    /// <summary>
    /// Tanh activation for output.
    /// </summary>
    private readonly IActivationFunction<T> _tanh;

    /// <summary>
    /// Number of output image channels.
    /// </summary>
    private readonly int _outputChannels;

    /// <summary>
    /// Number of latent channels.
    /// </summary>
    private readonly int _latentChannels;

    /// <summary>
    /// Base channel count.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Channel multipliers for each level.
    /// </summary>
    private readonly int[] _channelMults;

    /// <summary>
    /// Number of groups for GroupNorm.
    /// </summary>
    private readonly int _numGroups;

    /// <summary>
    /// Spatial size at decoder input (bottleneck).
    /// </summary>
    private readonly int _bottleneckSize;

    /// <summary>
    /// Final output spatial size.
    /// </summary>
    private readonly int _outputSpatialSize;

    /// <summary>
    /// Cached intermediate values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _postQuantOutput;
    private Tensor<T>? _inputConvOutput;
    private Tensor<T>? _midBlock1Output;
    private Tensor<T>? _midBlock2Output;
    private readonly Tensor<T>?[] _upBlockOutputs;
    private Tensor<T>? _normOutOutput;
    private Tensor<T>? _siluOutput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of output channels.
    /// </summary>
    public int OutputChannels => _outputChannels;

    /// <summary>
    /// Gets the number of latent channels.
    /// </summary>
    public int LatentChannels => _latentChannels;

    /// <summary>
    /// Gets the upsampling factor (spatial expansion from input to output).
    /// </summary>
    public int UpsampleFactor => (int)Math.Pow(2, _channelMults.Length - 1);

    /// <summary>
    /// Initializes a new instance of the VAEDecoder class.
    /// </summary>
    /// <param name="outputChannels">Number of output image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMults">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per UpBlock (default: 2).</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="outputSpatialSize">Spatial size of output images (default: 512).</param>
    public VAEDecoder(
        int outputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMults = null,
        int numResBlocks = 2,
        int numGroups = 32,
        int outputSpatialSize = 512)
        : base(
            CalculateInputShape(latentChannels, outputSpatialSize, channelMults?.Length ?? 4),
            CalculateOutputShape(outputChannels, outputSpatialSize))
    {
        if (outputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (latentChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (baseChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseChannels));

        _outputChannels = outputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMults = channelMults ?? new[] { 1, 2, 4, 4 };
        _numGroups = numGroups;
        _outputSpatialSize = outputSpatialSize;
        _silu = new SiLUActivation<T>();
        _tanh = new TanhActivation<T>();

        // Calculate bottleneck spatial size
        _bottleneckSize = outputSpatialSize;
        for (int i = 0; i < _channelMults.Length - 1; i++)
        {
            _bottleneckSize /= 2;
        }

        _upBlockOutputs = new Tensor<T>?[_channelMults.Length];

        // Post-quant convolution
        _postQuantConv = new ConvolutionalLayer<T>(
            inputDepth: latentChannels,
            outputDepth: latentChannels,
            kernelSize: 1,
            inputHeight: _bottleneckSize,
            inputWidth: _bottleneckSize,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        // Input convolution to expand latent to decoder channels
        int lastChannels = baseChannels * _channelMults[^1];
        _inputConv = new ConvolutionalLayer<T>(
            inputDepth: latentChannels,
            outputDepth: lastChannels,
            kernelSize: 3,
            inputHeight: _bottleneckSize,
            inputWidth: _bottleneckSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Middle blocks at bottleneck
        int midGroups = CalculateValidGroups(numGroups, lastChannels, lastChannels);
        _midBlocks = new VAEResBlock<T>[2];
        _midBlocks[0] = new VAEResBlock<T>(lastChannels, lastChannels, midGroups, _bottleneckSize);
        _midBlocks[1] = new VAEResBlock<T>(lastChannels, lastChannels, midGroups, _bottleneckSize);

        // Build up blocks (mirror of encoder, in reverse order of channel multipliers)
        _upBlocks = new UpBlock<T>[_channelMults.Length];
        int currentSpatialSize = _bottleneckSize;
        int inCh = lastChannels;

        for (int level = _channelMults.Length - 1; level >= 0; level--)
        {
            int outCh = baseChannels * _channelMults[level];
            bool hasUpsample = level > 0; // No upsample on first block (which is last in reversed order)

            int blockIndex = _channelMults.Length - 1 - level;
            _upBlocks[blockIndex] = new UpBlock<T>(
                inChannels: inCh,
                outChannels: outCh,
                numLayers: numResBlocks,
                numGroups: CalculateValidGroups(numGroups, inCh, outCh),
                inputSpatialSize: currentSpatialSize,
                hasUpsample: hasUpsample);

            inCh = outCh;
            if (hasUpsample)
            {
                currentSpatialSize *= 2;
            }
        }

        // Output normalization
        int outNormGroups = CalculateValidGroups(numGroups, baseChannels, baseChannels);
        _normOut = new GroupNormalizationLayer<T>(outNormGroups, baseChannels);

        // Output convolution with tanh activation for [-1, 1] output
        _outputConv = new ConvolutionalLayer<T>(
            inputDepth: baseChannels,
            outputDepth: outputChannels,
            kernelSize: 3,
            inputHeight: outputSpatialSize,
            inputWidth: outputSpatialSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    private static new int[] CalculateInputShape(int latentChannels, int outputSpatialSize, int numLevels)
    {
        int bottleneckSize = outputSpatialSize;
        for (int i = 0; i < numLevels - 1; i++)
        {
            bottleneckSize /= 2;
        }
        return new[] { latentChannels, bottleneckSize, bottleneckSize };
    }

    private static int[] CalculateOutputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
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
    /// Decodes a latent representation to an image.
    /// </summary>
    /// <param name="input">Latent tensor [batch, latentChannels, H, W].</param>
    /// <returns>Decoded image [batch, outputChannels, H*f, W*f] where f is upsample factor.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Post-quant convolution
        var x = _postQuantConv.Forward(input);
        _postQuantOutput = x;

        // Input convolution
        x = _inputConv.Forward(x);
        _inputConvOutput = x;

        // Middle blocks
        x = _midBlocks[0].Forward(x);
        _midBlock1Output = x;
        x = _midBlocks[1].Forward(x);
        _midBlock2Output = x;

        // Up blocks
        for (int i = 0; i < _upBlocks.Length; i++)
        {
            x = _upBlocks[i].Forward(x);
            _upBlockOutputs[i] = x;
        }

        // Output normalization and activation
        x = _normOut.Forward(x);
        _normOutOutput = x;
        x = ApplySiLU(x);
        _siluOutput = x;

        // Output convolution
        x = _outputConv.Forward(x);

        // Apply tanh for [-1, 1] output range
        return ApplyTanh(x);
    }

    private Tensor<T> ApplySiLU(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = _silu.Activate(inputSpan[i]);
        }

        return output;
    }

    private Tensor<T> ApplyTanh(Tensor<T> input)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = _tanh.Activate(inputSpan[i]);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass through the decoder.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _siluOutput == null || _normOutOutput == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        // Backward through tanh
        var grad = ApplyTanhDerivative(_siluOutput, outputGradient);

        // Backward through output conv
        grad = _outputConv.Backward(grad);

        // Backward through SiLU
        grad = ApplySiLUDerivative(_normOutOutput, grad);

        // Backward through output normalization
        grad = _normOut.Backward(grad);

        // Backward through up blocks
        for (int i = _upBlocks.Length - 1; i >= 0; i--)
        {
            grad = _upBlocks[i].Backward(grad);
        }

        // Backward through middle blocks
        grad = _midBlocks[1].Backward(grad);
        grad = _midBlocks[0].Backward(grad);

        // Backward through input conv
        grad = _inputConv.Backward(grad);

        // Backward through post-quant conv
        return _postQuantConv.Backward(grad);
    }

    private Tensor<T> ApplySiLUDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var gradSpan = gradient.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = NumOps.Multiply(_silu.Derivative(inputSpan[i]), gradSpan[i]);
        }

        return output;
    }

    private Tensor<T> ApplyTanhDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var output = new Tensor<T>(input.Shape);
        var inputSpan = input.AsSpan();
        var gradSpan = gradient.AsSpan();
        var outputSpan = output.AsWritableSpan();

        for (int i = 0; i < inputSpan.Length; i++)
        {
            outputSpan[i] = NumOps.Multiply(_tanh.Derivative(inputSpan[i]), gradSpan[i]);
        }

        return output;
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _postQuantConv.UpdateParameters(learningRate);
        _inputConv.UpdateParameters(learningRate);

        foreach (var block in _midBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        foreach (var block in _upBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        _normOut.UpdateParameters(learningRate);
        _outputConv.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        AddParameters(paramsList, _postQuantConv.GetParameters());
        AddParameters(paramsList, _inputConv.GetParameters());

        foreach (var block in _midBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
        }

        foreach (var block in _upBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
        }

        AddParameters(paramsList, _normOut.GetParameters());
        AddParameters(paramsList, _outputConv.GetParameters());

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

        SetLayerParams(_postQuantConv, parameters, ref index);
        SetLayerParams(_inputConv, parameters, ref index);

        foreach (var block in _midBlocks)
        {
            SetLayerParams(block, parameters, ref index);
        }

        foreach (var block in _upBlocks)
        {
            SetLayerParams(block, parameters, ref index);
        }

        SetLayerParams(_normOut, parameters, ref index);
        SetLayerParams(_outputConv, parameters, ref index);
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
    /// Resets the internal state of the decoder.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _postQuantOutput = null;
        _inputConvOutput = null;
        _midBlock1Output = null;
        _midBlock2Output = null;
        _normOutOutput = null;
        _siluOutput = null;

        for (int i = 0; i < _upBlockOutputs.Length; i++)
        {
            _upBlockOutputs[i] = null;
        }

        _postQuantConv.ResetState();
        _inputConv.ResetState();
        foreach (var block in _midBlocks)
        {
            block.ResetState();
        }
        foreach (var block in _upBlocks)
        {
            block.ResetState();
        }
        _normOut.ResetState();
        _outputConv.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("VAEDecoder JIT compilation is not yet implemented.");
    }

    /// <summary>
    /// Saves the decoder's state to a binary writer.
    /// </summary>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(_outputChannels);
        writer.Write(_latentChannels);
        writer.Write(_baseChannels);
        writer.Write(_channelMults.Length);
        foreach (var mult in _channelMults)
        {
            writer.Write(mult);
        }
        writer.Write(_numGroups);
        writer.Write(_bottleneckSize);
        writer.Write(_outputSpatialSize);

        _postQuantConv.Serialize(writer);
        _inputConv.Serialize(writer);

        foreach (var block in _midBlocks)
        {
            block.Serialize(writer);
        }

        foreach (var block in _upBlocks)
        {
            block.Serialize(writer);
        }

        _normOut.Serialize(writer);
        _outputConv.Serialize(writer);
    }

    /// <summary>
    /// Loads the decoder's state from a binary reader.
    /// </summary>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        var outputChannels = reader.ReadInt32();
        var latentChannels = reader.ReadInt32();
        var baseChannels = reader.ReadInt32();
        var numMults = reader.ReadInt32();
        var channelMults = new int[numMults];
        for (int i = 0; i < numMults; i++)
        {
            channelMults[i] = reader.ReadInt32();
        }
        _ = reader.ReadInt32(); // numGroups
        _ = reader.ReadInt32(); // bottleneckSize
        _ = reader.ReadInt32(); // outputSpatialSize

        if (outputChannels != _outputChannels || latentChannels != _latentChannels ||
            baseChannels != _baseChannels || !channelMults.SequenceEqual(_channelMults))
        {
            throw new InvalidOperationException("Architecture mismatch in VAEDecoder deserialization.");
        }

        _postQuantConv.Deserialize(reader);
        _inputConv.Deserialize(reader);

        foreach (var block in _midBlocks)
        {
            block.Deserialize(reader);
        }

        foreach (var block in _upBlocks)
        {
            block.Deserialize(reader);
        }

        _normOut.Deserialize(reader);
        _outputConv.Deserialize(reader);
    }
}
