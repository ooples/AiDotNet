using AiDotNet.ActivationFunctions;
using AiDotNet.ModelLoading;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Convolutional encoder for VAE that compresses images to latent space.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the encoder portion of a VAE following the Stable Diffusion architecture:
/// - Input convolution to initial feature channels
/// - Multiple DownBlocks with ResBlocks and strided conv downsampling
/// - Middle blocks with attention at the bottleneck
/// - Final convolutions to produce mean and log variance for the latent distribution
/// </para>
/// <para>
/// <b>For Beginners:</b> The VAE encoder is like an intelligent image compressor.
///
/// What it does step by step:
/// 1. Takes a high-resolution image (e.g., 512x512x3 RGB)
/// 2. Initial conv: Expands channels (3 -> 128) at full resolution
/// 3. DownBlocks: Progressively halves resolution while increasing channels
///    - Block 1: 128 channels, 512x512 -> 256x256
///    - Block 2: 256 channels, 256x256 -> 128x128
///    - Block 3: 512 channels, 128x128 -> 64x64
///    - Block 4: 512 channels, 64x64 -> 64x64 (no downsample at end)
/// 4. Middle: Extra processing at the bottleneck
/// 5. Output: Produces mean and log-variance for 4-channel latent
///
/// The result is a 64x64x4 latent that captures the image's essence
/// in a compressed form suitable for diffusion.
/// </para>
/// </remarks>
public class VAEEncoder<T> : LayerBase<T>
{
    /// <summary>
    /// Input convolution from image channels to base channels.
    /// </summary>
    private readonly ConvolutionalLayer<T> _inputConv;

    /// <summary>
    /// Downsampling blocks.
    /// </summary>
    private readonly DownBlock<T>[] _downBlocks;

    /// <summary>
    /// Middle residual blocks at the bottleneck.
    /// </summary>
    private readonly VAEResBlock<T>[] _midBlocks;

    /// <summary>
    /// Convolution to project to mean.
    /// </summary>
    private readonly ConvolutionalLayer<T> _meanConv;

    /// <summary>
    /// Convolution to project to log variance.
    /// </summary>
    private readonly ConvolutionalLayer<T> _logVarConv;

    /// <summary>
    /// Quant convolution for latent processing.
    /// </summary>
    private readonly ConvolutionalLayer<T> _quantConv;

    /// <summary>
    /// Group normalization before output projections.
    /// </summary>
    private readonly GroupNormalizationLayer<T> _normOut;

    /// <summary>
    /// SiLU activation function.
    /// </summary>
    private readonly IActivationFunction<T> _silu;

    /// <summary>
    /// Parameter registry for named weight access.
    /// </summary>
    private ParameterRegistry<T>? _parameterRegistry;

    /// <summary>
    /// Number of input image channels.
    /// </summary>
    private readonly int _inputChannels;

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
    /// Spatial size at encoder output (bottleneck).
    /// </summary>
    private readonly int _bottleneckSize;

    /// <summary>
    /// Cached intermediate values for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;
    private Tensor<T>? _inputConvOutput;
    private readonly Tensor<T>?[] _downBlockOutputs;
    private Tensor<T>? _midBlock1Output;
    private Tensor<T>? _midBlock2Output;
    private Tensor<T>? _normOutOutput;
    private Tensor<T>? _siluOutput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    public int InputChannels => _inputChannels;

    /// <summary>
    /// Gets the number of latent channels.
    /// </summary>
    public int LatentChannels => _latentChannels;

    /// <summary>
    /// Gets the downsampling factor (spatial reduction from input to output).
    /// </summary>
    public int DownsampleFactor => (int)Math.Pow(2, _channelMults.Length - 1);

    /// <summary>
    /// Initializes a new instance of the VAEEncoder class.
    /// </summary>
    /// <param name="inputChannels">Number of input image channels (default: 3 for RGB).</param>
    /// <param name="latentChannels">Number of latent channels (default: 4).</param>
    /// <param name="baseChannels">Base channel count (default: 128).</param>
    /// <param name="channelMults">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per DownBlock (default: 2).</param>
    /// <param name="numGroups">Number of groups for GroupNorm (default: 32).</param>
    /// <param name="inputSpatialSize">Spatial size of input images (default: 512).</param>
    public VAEEncoder(
        int inputChannels = 3,
        int latentChannels = 4,
        int baseChannels = 128,
        int[]? channelMults = null,
        int numResBlocks = 2,
        int numGroups = 32,
        int inputSpatialSize = 512)
        : base(
            CalculateInputShape(inputChannels, inputSpatialSize),
            CalculateOutputShape(latentChannels, inputSpatialSize, channelMults?.Length ?? 4))
    {
        if (inputChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (latentChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(latentChannels));
        if (baseChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseChannels));

        _inputChannels = inputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMults = channelMults ?? new[] { 1, 2, 4, 4 };
        _numGroups = numGroups;
        _silu = new SiLUActivation<T>();

        // Calculate bottleneck spatial size
        _bottleneckSize = inputSpatialSize;
        for (int i = 0; i < _channelMults.Length - 1; i++)
        {
            _bottleneckSize /= 2;
        }

        _downBlockOutputs = new Tensor<T>?[_channelMults.Length];

        // Input convolution: [inputChannels] -> [baseChannels]
        _inputConv = new ConvolutionalLayer<T>(
            inputDepth: inputChannels,
            outputDepth: baseChannels,
            kernelSize: 3,
            inputHeight: inputSpatialSize,
            inputWidth: inputSpatialSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Build down blocks
        _downBlocks = new DownBlock<T>[_channelMults.Length];
        int currentSpatialSize = inputSpatialSize;
        int inCh = baseChannels;

        for (int level = 0; level < _channelMults.Length; level++)
        {
            int outCh = baseChannels * _channelMults[level];
            bool hasDownsample = level < _channelMults.Length - 1; // No downsample on last block

            _downBlocks[level] = new DownBlock<T>(
                inChannels: inCh,
                outChannels: outCh,
                numLayers: numResBlocks,
                numGroups: CalculateValidGroups(numGroups, inCh, outCh),
                inputSpatialSize: currentSpatialSize,
                hasDownsample: hasDownsample);

            inCh = outCh;
            if (hasDownsample)
            {
                currentSpatialSize /= 2;
            }
        }

        // Middle blocks at bottleneck
        int lastChannels = baseChannels * _channelMults[^1];
        int midGroups = CalculateValidGroups(numGroups, lastChannels, lastChannels);

        _midBlocks = new VAEResBlock<T>[2];
        _midBlocks[0] = new VAEResBlock<T>(lastChannels, lastChannels, midGroups, _bottleneckSize);
        _midBlocks[1] = new VAEResBlock<T>(lastChannels, lastChannels, midGroups, _bottleneckSize);

        // Output normalization and projection
        _normOut = new GroupNormalizationLayer<T>(midGroups, lastChannels);

        // Mean and log variance projections
        _meanConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels,
            outputDepth: latentChannels,
            kernelSize: 3,
            inputHeight: _bottleneckSize,
            inputWidth: _bottleneckSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _logVarConv = new ConvolutionalLayer<T>(
            inputDepth: lastChannels,
            outputDepth: latentChannels,
            kernelSize: 3,
            inputHeight: _bottleneckSize,
            inputWidth: _bottleneckSize,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Quant conv for latent processing
        _quantConv = new ConvolutionalLayer<T>(
            inputDepth: latentChannels,
            outputDepth: latentChannels,
            kernelSize: 1,
            inputHeight: _bottleneckSize,
            inputWidth: _bottleneckSize,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());
    }

    private static int[] CalculateInputShape(int channels, int spatialSize)
    {
        return new[] { channels, spatialSize, spatialSize };
    }

    private static new int[] CalculateOutputShape(int latentChannels, int inputSpatialSize, int numLevels)
    {
        int bottleneckSize = inputSpatialSize;
        for (int i = 0; i < numLevels - 1; i++)
        {
            bottleneckSize /= 2;
        }
        // Output is 2x latent channels (mean + logvar concatenated)
        return new[] { latentChannels * 2, bottleneckSize, bottleneckSize };
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
    /// Encodes an image to latent space, returning concatenated mean and log variance.
    /// </summary>
    /// <param name="input">Input image tensor [batch, inputChannels, H, W].</param>
    /// <returns>Concatenated mean and log variance [batch, 2*latentChannels, H/f, W/f].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Input convolution
        var x = _inputConv.Forward(input);
        _inputConvOutput = x;

        // Down blocks
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            x = _downBlocks[i].Forward(x);
            _downBlockOutputs[i] = x;
        }

        // Middle blocks
        x = _midBlocks[0].Forward(x);
        _midBlock1Output = x;
        x = _midBlocks[1].Forward(x);
        _midBlock2Output = x;

        // Output normalization and activation
        x = _normOut.Forward(x);
        _normOutOutput = x;
        x = ApplySiLU(x);
        _siluOutput = x;

        // Project to mean and log variance
        var mean = _meanConv.Forward(x);
        var logVar = _logVarConv.Forward(x);

        // Apply quant conv to mean
        mean = _quantConv.Forward(mean);

        // Concatenate mean and logVar along channel dimension
        return ConcatenateChannels(mean, logVar);
    }

    /// <summary>
    /// Encodes and returns mean and log variance separately.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <returns>Tuple of (mean, logVariance) tensors.</returns>
    public (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> input)
    {
        var combined = Forward(input);

        // Split combined output back into mean and logVar
        return SplitChannels(combined, _latentChannels);
    }

    /// <summary>
    /// Encodes an image and samples from the latent distribution.
    /// </summary>
    /// <param name="input">Input image tensor.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>Sampled latent tensor.</returns>
    public Tensor<T> EncodeAndSample(Tensor<T> input, int? seed = null)
    {
        var (mean, logVar) = EncodeWithDistribution(input);
        return Sample(mean, logVar, seed);
    }

    /// <summary>
    /// Samples from the latent distribution using the reparameterization trick.
    /// </summary>
    private Tensor<T> Sample(Tensor<T> mean, Tensor<T> logVar, int? seed)
    {
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var epsilon = SampleNoise(mean.Shape, rng);

        var result = new Tensor<T>(mean.Shape);
        var meanSpan = mean.AsSpan();
        var logVarSpan = logVar.AsSpan();
        var epsilonSpan = epsilon.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var halfOne = NumOps.FromDouble(0.5);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            // std = exp(0.5 * logVar), z = mean + std * epsilon
            var std = NumOps.Exp(NumOps.Multiply(halfOne, logVarSpan[i]));
            resultSpan[i] = NumOps.Add(meanSpan[i], NumOps.Multiply(std, epsilonSpan[i]));
        }

        return result;
    }

    /// <summary>
    /// Samples random noise from a standard normal distribution.
    /// </summary>
    private Tensor<T> SampleNoise(int[] shape, Random rng)
    {
        var noise = new Tensor<T>(shape);
        var noiseSpan = noise.AsWritableSpan();

        for (int i = 0; i < noiseSpan.Length; i++)
        {
            noiseSpan[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
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

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Assuming shape [batch, channels, H, W]
        var shape = a.Shape;
        int batch = shape.Length > 3 ? shape[0] : 1;
        int channelsA = shape.Length > 3 ? shape[1] : shape[0];
        int channelsB = b.Shape.Length > 3 ? b.Shape[1] : b.Shape[0];
        int height = shape.Length > 3 ? shape[2] : shape[1];
        int width = shape.Length > 3 ? shape[3] : shape[2];

        var result = new Tensor<T>(new[] { batch, channelsA + channelsB, height, width });
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var resultSpan = result.AsWritableSpan();

        int spatialSize = height * width;

        for (int n = 0; n < batch; n++)
        {
            // Copy channels from a
            for (int c = 0; c < channelsA; c++)
            {
                int srcOffset = n * channelsA * spatialSize + c * spatialSize;
                int dstOffset = n * (channelsA + channelsB) * spatialSize + c * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    resultSpan[dstOffset + s] = aSpan[srcOffset + s];
                }
            }

            // Copy channels from b
            for (int c = 0; c < channelsB; c++)
            {
                int srcOffset = n * channelsB * spatialSize + c * spatialSize;
                int dstOffset = n * (channelsA + channelsB) * spatialSize + (channelsA + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                {
                    resultSpan[dstOffset + s] = bSpan[srcOffset + s];
                }
            }
        }

        return result;
    }

    private (Tensor<T> First, Tensor<T> Second) SplitChannels(Tensor<T> combined, int splitChannels)
    {
        var shape = combined.Shape;
        int batch = shape.Length > 3 ? shape[0] : 1;
        int totalChannels = shape.Length > 3 ? shape[1] : shape[0];
        int height = shape.Length > 3 ? shape[2] : shape[1];
        int width = shape.Length > 3 ? shape[3] : shape[2];

        var first = new Tensor<T>(new[] { batch, splitChannels, height, width });
        var second = new Tensor<T>(new[] { batch, splitChannels, height, width });
        var combinedSpan = combined.AsSpan();
        var firstSpan = first.AsWritableSpan();
        var secondSpan = second.AsWritableSpan();

        int spatialSize = height * width;

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < splitChannels; c++)
            {
                int srcOffsetFirst = n * totalChannels * spatialSize + c * spatialSize;
                int srcOffsetSecond = n * totalChannels * spatialSize + (splitChannels + c) * spatialSize;
                int dstOffset = n * splitChannels * spatialSize + c * spatialSize;

                for (int s = 0; s < spatialSize; s++)
                {
                    firstSpan[dstOffset + s] = combinedSpan[srcOffsetFirst + s];
                    secondSpan[dstOffset + s] = combinedSpan[srcOffsetSecond + s];
                }
            }
        }

        return (first, second);
    }

    /// <summary>
    /// Performs the backward pass through the encoder.
    /// </summary>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Split gradient for mean and logVar paths
        var (meanGrad, logVarGrad) = SplitChannels(outputGradient, _latentChannels);

        // Backward through quant conv (mean path only)
        meanGrad = _quantConv.Backward(meanGrad);

        // Backward through mean and logVar convs
        var normGradMean = _meanConv.Backward(meanGrad);
        var normGradLogVar = _logVarConv.Backward(logVarGrad);

        // Combine gradients
        var normGrad = Engine.TensorAdd(normGradMean, normGradLogVar);

        // Backward through SiLU and normalization
        normGrad = ApplySiLUDerivative(_normOutOutput!, normGrad);
        normGrad = _normOut.Backward(normGrad);

        // Backward through middle blocks
        normGrad = _midBlocks[1].Backward(normGrad);
        normGrad = _midBlocks[0].Backward(normGrad);

        // Backward through down blocks
        for (int i = _downBlocks.Length - 1; i >= 0; i--)
        {
            normGrad = _downBlocks[i].Backward(normGrad);
        }

        // Backward through input conv
        return _inputConv.Backward(normGrad);
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

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        _inputConv.UpdateParameters(learningRate);

        foreach (var block in _downBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        foreach (var block in _midBlocks)
        {
            block.UpdateParameters(learningRate);
        }

        _normOut.UpdateParameters(learningRate);
        _meanConv.UpdateParameters(learningRate);
        _logVarConv.UpdateParameters(learningRate);
        _quantConv.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        AddParameters(paramsList, _inputConv.GetParameters());

        foreach (var block in _downBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
        }

        foreach (var block in _midBlocks)
        {
            AddParameters(paramsList, block.GetParameters());
        }

        AddParameters(paramsList, _normOut.GetParameters());
        AddParameters(paramsList, _meanConv.GetParameters());
        AddParameters(paramsList, _logVarConv.GetParameters());
        AddParameters(paramsList, _quantConv.GetParameters());

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

        SetLayerParams(_inputConv, parameters, ref index);

        foreach (var block in _downBlocks)
        {
            SetLayerParams(block, parameters, ref index);
        }

        foreach (var block in _midBlocks)
        {
            SetLayerParams(block, parameters, ref index);
        }

        SetLayerParams(_normOut, parameters, ref index);
        SetLayerParams(_meanConv, parameters, ref index);
        SetLayerParams(_logVarConv, parameters, ref index);
        SetLayerParams(_quantConv, parameters, ref index);
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
    /// Resets the internal state of the encoder.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _inputConvOutput = null;
        _midBlock1Output = null;
        _midBlock2Output = null;
        _normOutOutput = null;
        _siluOutput = null;

        for (int i = 0; i < _downBlockOutputs.Length; i++)
        {
            _downBlockOutputs[i] = null;
        }

        _inputConv.ResetState();
        foreach (var block in _downBlocks)
        {
            block.ResetState();
        }
        foreach (var block in _midBlocks)
        {
            block.ResetState();
        }
        _normOut.ResetState();
        _meanConv.ResetState();
        _logVarConv.ResetState();
        _quantConv.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("VAEEncoder JIT compilation is not yet implemented.");
    }

    /// <summary>
    /// Saves the encoder's state to a binary writer.
    /// </summary>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(_inputChannels);
        writer.Write(_latentChannels);
        writer.Write(_baseChannels);
        writer.Write(_channelMults.Length);
        foreach (var mult in _channelMults)
        {
            writer.Write(mult);
        }
        writer.Write(_numGroups);
        writer.Write(_bottleneckSize);

        _inputConv.Serialize(writer);

        foreach (var block in _downBlocks)
        {
            block.Serialize(writer);
        }

        foreach (var block in _midBlocks)
        {
            block.Serialize(writer);
        }

        _normOut.Serialize(writer);
        _meanConv.Serialize(writer);
        _logVarConv.Serialize(writer);
        _quantConv.Serialize(writer);
    }

    /// <summary>
    /// Loads the encoder's state from a binary reader.
    /// </summary>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        var inputChannels = reader.ReadInt32();
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

        if (inputChannels != _inputChannels || latentChannels != _latentChannels ||
            baseChannels != _baseChannels || !channelMults.SequenceEqual(_channelMults))
        {
            throw new InvalidOperationException("Architecture mismatch in VAEEncoder deserialization.");
        }

        _inputConv.Deserialize(reader);

        foreach (var block in _downBlocks)
        {
            block.Deserialize(reader);
        }

        foreach (var block in _midBlocks)
        {
            block.Deserialize(reader);
        }

        _normOut.Deserialize(reader);
        _meanConv.Deserialize(reader);
        _logVarConv.Deserialize(reader);
        _quantConv.Deserialize(reader);
    }

    #region IWeightLoadable Implementation

    /// <summary>
    /// Builds the parameter registry for named weight access.
    /// </summary>
    private ParameterRegistry<T> BuildParameterRegistry()
    {
        var registry = new ParameterRegistry<T>();

        // Register input convolution
        registry.RegisterLayer("inputConv", _inputConv);

        // Register down blocks (use RegisterLayer since they inherit IWeightLoadable from LayerBase)
        for (int i = 0; i < _downBlocks.Length; i++)
        {
            RegisterWeightLoadable(registry, $"down{i}", _downBlocks[i]);
        }

        // Register middle blocks
        for (int i = 0; i < _midBlocks.Length; i++)
        {
            RegisterWeightLoadable(registry, $"mid{i}", _midBlocks[i]);
        }

        // Register output layers
        registry.RegisterLayer("normOut", _normOut);
        registry.RegisterLayer("meanConv", _meanConv);
        registry.RegisterLayer("logVarConv", _logVarConv);
        registry.RegisterLayer("quantConv", _quantConv);

        return registry;
    }

    /// <summary>
    /// Registers all parameters from an IWeightLoadable into the registry with a prefix.
    /// </summary>
    private static void RegisterWeightLoadable(ParameterRegistry<T> registry, string prefix, IWeightLoadable<T> weightLoadable)
    {
        foreach (var paramName in weightLoadable.GetParameterNames())
        {
            var fullName = $"{prefix}.{paramName}";
            var shape = weightLoadable.GetParameterShape(paramName);
            if (shape != null)
            {
                registry.Register(
                    fullName,
                    shape,
                    () =>
                    {
                        weightLoadable.TryGetParameter(paramName, out var tensor);
                        return tensor;
                    },
                    tensor => weightLoadable.SetParameter(paramName, tensor));
            }
        }
    }

    /// <summary>
    /// Gets or creates the parameter registry.
    /// </summary>
    private ParameterRegistry<T> GetParameterRegistry()
    {
        _parameterRegistry ??= BuildParameterRegistry();
        return _parameterRegistry;
    }

    /// <inheritdoc />
    public override IEnumerable<string> GetParameterNames()
    {
        return GetParameterRegistry().GetNames();
    }

    /// <inheritdoc />
    public override bool TryGetParameter(string name, out Tensor<T>? tensor)
    {
        return GetParameterRegistry().TryGet(name, out tensor);
    }

    /// <inheritdoc />
    public override bool SetParameter(string name, Tensor<T> value)
    {
        return GetParameterRegistry().TrySet(name, value);
    }

    /// <inheritdoc />
    public override int[]? GetParameterShape(string name)
    {
        return GetParameterRegistry().GetShape(name);
    }

    /// <inheritdoc />
    public override int NamedParameterCount => GetParameterRegistry().Count;

    /// <inheritdoc />
    public override WeightLoadValidation ValidateWeights(IEnumerable<string> weightNames, Func<string, string?>? mapping = null)
    {
        return GetParameterRegistry().Validate(weightNames, mapping);
    }

    /// <inheritdoc />
    public override WeightLoadResult LoadWeights(Dictionary<string, Tensor<T>> weights, Func<string, string?>? mapping = null, bool strict = false)
    {
        return GetParameterRegistry().Load(weights, mapping, strict);
    }

    /// <summary>
    /// Builds and returns the parameter registry for external use.
    /// </summary>
    public ParameterRegistry<T> BuildParameterRegistryPublic()
    {
        return BuildParameterRegistry();
    }

    #endregion
}
