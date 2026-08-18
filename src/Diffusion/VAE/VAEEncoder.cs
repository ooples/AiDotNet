using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.ModelLoading;
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
// Roles from this encoder's own ForwardTraced doc - "Input image tensor [batch, inputChannels, H, W]"
// producing "Concatenated mean and log variance [batch, 2*latentChannels, H/f, W/f]". Batch is NOT
// marked optional: every pre-resolve in the constructor hands its convolutions a rank-4 shape
// (ResolveFromShape(new[] { 1, inputChannels, inputSpatialSize, inputSpatialSize })), and nothing here
// establishes that the stack accepts an unbatched [C,H,W].
// OutputAxesFor below is HAND-WRITTEN: both the latent width and the downsample depth are options.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output,
    Note = "Channels carry mean and log-variance concatenated, hence 2 * latentChannels.")]
[AutoParameters]
public partial class VAEEncoder<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// CHANNELS. <c>Fixed(_latentChannels * 2)</c>, matching the private <c>CalculateOutputShape</c>
    /// (<c>new[] { latentChannels * 2, ... }</c>) and what <c>ForwardEager</c> actually builds: a
    /// <c>_meanConv</c> and a <c>_logVarConv</c> both of <c>outputDepth: latentChannels</c>, joined by
    /// <c>ConcatenateChannels(mean, logVar)</c>. Nothing about the input channel count survives - the
    /// very first <c>_inputConv</c> replaces it with <c>baseChannels</c>.
    /// </para>
    /// <para>
    /// SPATIAL. Every convolution in this stack is extent-preserving (3x3 stride 1 padding 1, or 1x1
    /// stride 1 padding 0) EXCEPT the strided downsample inside each <c>DownBlock</c>, so the whole
    /// encoder's spatial relation is exactly that one downsample repeated. There are
    /// <c>_channelMults.Length - 1</c> of them - the constructor sets
    /// <c>hasDownsample = level &lt; _channelMults.Length - 1</c>, "No downsample on last block" - and
    /// each is <c>kernelSize: 3, stride: 2, padding: 1</c>.
    /// </para>
    /// <para>
    /// A STACK OF WINDOWS FOLDS INTO ONE WINDOW, left to right:
    /// <c>k = k1 + (k2-1)*s1</c>, <c>s = s1*s2</c>, <c>p = p1 + p2*s1</c>. Repeating (3, 2, 1) L times
    /// gives <c>s = 2^L</c>, <c>k = 2*s - 1</c>, <c>p = s - 1</c> - which is why the kernel and padding
    /// below are derived from <see cref="DownsampleFactor"/> (itself <c>2^(_channelMults.Length - 1)</c>)
    /// rather than written out. Evaluated, that window is <c>ceil(H / 2^L)</c>.
    /// </para>
    /// <para>
    /// <c>Window</c> and NOT <c>Scaled(Height, 1, DownsampleFactor)</c>, which is the tempting shorthand
    /// and is wrong twice over: <c>Scaled</c> refuses uneven division, so it would decline on any odd
    /// extent the layer in fact accepts, and where it did resolve it would report <c>floor</c> where a
    /// padded stride-2 convolution produces <c>ceil</c>. The two agree only on even sizes.
    /// </para>
    /// <para>
    /// The <c>inputSpatialSize</c> constructor argument is not a claim about the input. It sizes the
    /// pre-resolved parameter shapes so <c>ParameterRegistry</c> and pretrained-weight loading work
    /// before any forward pass; the relation above holds for whatever extent actually arrives.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 || _latentChannels <= 0) return null;

        int stride = DownsampleFactor;
        if (stride <= 0) return null;
        int kernel = 2 * stride - 1;
        int padding = stride - 1;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_latentChannels * 2)),
            new OutputAxisContract(
                TensorAxis.Height, AxisRelation.Window(TensorAxis.Height, kernel, stride, padding)),
            new OutputAxisContract(
                TensorAxis.Width, AxisRelation.Window(TensorAxis.Width, kernel, stride, padding)),
        };
    }

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
    [Scratch]
    private Tensor<T>? _lastInput;
    private Tensor<T>? _inputConvOutput;
    private readonly Tensor<T>?[] _downBlockOutputs;
    private Tensor<T>? _midBlock1Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _midBlock2Output;
    [AiDotNet.Attributes.Scratch]
    private Tensor<T>? _normOutOutput;
    [AiDotNet.Attributes.Scratch]
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

    /// <summary>Construction state: the 'inputSpatialSize' the layer was built with.</summary>
    private readonly int _inputSpatialSize;

    /// <summary>Construction state: the 'numResBlocks' the layer was built with.</summary>
    private readonly int _numResBlocks;

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
        _numResBlocks = numResBlocks;
        _inputSpatialSize = inputSpatialSize;
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
            outputDepth: baseChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
        // Pre-resolve so ParameterRegistry, GetParameters, SetParameters, and
        // pretrained weight loading all work on a freshly constructed encoder.
        _inputConv.ResolveFromShape(new[] { 1, inputChannels, inputSpatialSize, inputSpatialSize });

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
            outputDepth: latentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        _logVarConv = new ConvolutionalLayer<T>(
            outputDepth: latentChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: new IdentityActivation<T>());

        // Quant conv for latent processing
        _quantConv = new ConvolutionalLayer<T>(
            outputDepth: latentChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        // Pre-resolve mean/logvar/quant convs from the bottleneck topology so the
        // ParameterRegistry has consistent named entries for pretrained-weight loading
        // before any forward pass fires.
        _meanConv.ResolveFromShape(new[] { 1, lastChannels, _bottleneckSize, _bottleneckSize });
        _logVarConv.ResolveFromShape(new[] { 1, lastChannels, _bottleneckSize, _bottleneckSize });
        _quantConv.ResolveFromShape(new[] { 1, latentChannels, _bottleneckSize, _bottleneckSize });
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
    /// Per-instance compile host. Routing the encoder's full forward through
    /// this host means a second + Nth call at the same input shape replays a
    /// cached compiled plan instead of re-tracing the entire ResBlock + Conv
    /// stack. The host is materialized lazily on first Forward call so the
    /// per-instance setup cost is paid only by callers that actually invoke
    /// the encoder (a VAEEncoder constructed but never called pays nothing).
    /// (#1272 W1: VAEEncoder gets its own CompiledModelHost.)
    /// </summary>
    private AiDotNet.NeuralNetworks.CompiledModelHost<T>? _compileHost;
    private int _compileStructureVersion;

    private AiDotNet.NeuralNetworks.CompiledModelHost<T> EnsureCompileHost() =>
        _compileHost ??= new AiDotNet.NeuralNetworks.CompiledModelHost<T>(
            modelIdentity: nameof(VAEEncoder<T>));

    /// <summary>
    /// Encodes an image to latent space, returning concatenated mean and log variance.
    /// </summary>
    /// <param name="input">Input image tensor [batch, inputChannels, H, W].</param>
    /// <returns>Concatenated mean and log variance [batch, 2*latentChannels, H/f, W/f].</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _lastInput = input;
        return EnsureCompileHost().Predict(input, _compileStructureVersion, () => ForwardEager(input));
    }

    /// <summary>
    /// Eager forward pass (the body of the original Forward). Called by the
    /// compile host on cache miss / when compilation is disabled, and reused
    /// by the backward path when it needs intermediate-tensor caching.
    /// </summary>
    private Tensor<T> ForwardEager(Tensor<T> input)
    {
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
    /// Async overload of <see cref="Forward(Tensor{T})"/>. Routes through the
    /// compile host's <c>PredictAsync</c> so callers in async pipelines get
    /// the GPU-stream-aware <c>ExecuteAsync</c> path. (#1272 W1.)
    /// </summary>
    public System.Threading.Tasks.ValueTask<Tensor<T>> ForwardAsync(
        Tensor<T> input,
        System.Threading.CancellationToken cancellationToken = default)
    {
        _lastInput = input;
        return EnsureCompileHost().PredictAsync(
            input, _compileStructureVersion, () => ForwardEager(input), cancellationToken);
    }

    /// <summary>
    /// Bumps the structure-version counter so the next call to
    /// <see cref="Forward"/> drops any plan compiled against an older
    /// graph topology. Called by serialization / weight-loading paths
    /// that mutate the underlying layer parameters.
    /// </summary>
    public void InvalidateCompiledPlans()
    {
        _compileStructureVersion++;
        _compileHost?.Invalidate();
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
        var epsilon = SampleNoise(mean._shape, rng);

        var result = new Tensor<T>(mean._shape);
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
        return Engine.Swish(input);
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Channel axis is 0 for 3D [C, H, W] and 1 for 4D [N, C, H, W]
        int channelAxis = a.Shape.Length == 4 ? 1 : 0;
        return Engine.TensorConcatenate([a, b], axis: channelAxis);
    }

    private (Tensor<T> First, Tensor<T> Second) SplitChannels(Tensor<T> combined, int splitChannels)
    {
        var shape = combined._shape;
        bool batched = shape.Length > 3;
        int batch = batched ? shape[0] : 1;
        int totalChannels = batched ? shape[1] : shape[0];
        int height = batched ? shape[2] : shape[1];
        int width = batched ? shape[3] : shape[2];

        // Preserve the input rank so callers that round-trip rank-3 [C, H, W] tensors
        // through the encoder don't see their tensor silently promoted to rank-4.
        int[] outShape = batched
            ? new[] { batch, splitChannels, height, width }
            : new[] { splitChannels, height, width };

        var first = new Tensor<T>(outShape);
        var second = new Tensor<T>(outShape);
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
    // GetParameters / SetParameters are deliberately NOT overridden. LayerBase now implements both
    // concretely, folding over the same registry ParameterCount sums -- Parameters, this layer's
    // registered tensors, then each sub-layer -- in one order, so the count, the vector and the
    // restore cannot describe different tensors. The hand-written versions here walked the child
    // fields directly while ParameterCount walked the registry, which is why this layer reported 0
    // against a 56,092-value vector and every model containing it inherited the mismatch.

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
    /// Builds and returns the parameter registry for use inside the assembly. Kept
    /// <c>internal</c> so the raw <see cref="ParameterRegistry{T}"/> plumbing isn't
    /// frozen into the public API surface — external callers should use the
    /// <see cref="LoadWeights"/>/<see cref="ValidateWeights"/> entry points instead.
    /// </summary>
    internal ParameterRegistry<T> BuildParameterRegistryPublic()
    {
        return BuildParameterRegistry();
    }

    #endregion
}
