using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Engines;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.VAE;

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
// Roles from this decoder's own ForwardTraced doc - "Latent tensor [batch, latentChannels, H, W]"
// producing "Decoded image [batch, outputChannels, H*f, W*f] where f is upsample factor". Batch is NOT
// marked optional: nothing in this file establishes that the convolution/GroupNorm stack accepts an
// unbatched [C,H,W], so declaring rank 3 would be a claim made on no evidence.
// OutputAxesFor below is HAND-WRITTEN: both the image width and the upsample depth are options.
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class VAEDecoder<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// CHANNELS. <c>Fixed(_outputChannels)</c>, read off the constructor argument. It is what the final
    /// <c>_outputConv</c> is built with (<c>outputDepth: outputChannels</c>) and what the private
    /// <c>CalculateOutputShape</c> declares. The latent channel count does not survive - the second
    /// convolution in <c>ForwardEager</c> has already replaced it with <c>baseChannels *
    /// _channelMults[^1]</c>.
    /// </para>
    /// <para>
    /// SPATIAL. Every convolution here is extent-preserving (3x3 stride 1 padding 1, or the 1x1
    /// <c>_postQuantConv</c>) EXCEPT the transposed-conv upsampler inside each <c>UpBlock</c>, so the
    /// decoder's spatial relation is that one upsample repeated. There are
    /// <c>_channelMults.Length - 1</c> of them - the constructor sets <c>hasUpsample = level &gt; 0</c>,
    /// "No upsample on first block" - which is exactly <see cref="UpsampleFactor"/>,
    /// <c>2^(_channelMults.Length - 1)</c>.
    /// </para>
    /// <para>
    /// <c>Scaled</c> and not <c>Window</c>, and that is the OPPOSITE choice from
    /// <c>VAEEncoder</c> - deliberately, because the two are not mirror images arithmetically. The
    /// encoder's stride-2 convolution rounds (it produces <c>ceil(H/2)</c>), so it needs the window
    /// formula; the decoder's upsampler does not. <c>UpBlock</c> states the guarantee itself: "The
    /// spatial factor is EXACTLY two, not approximately: the upsampler is a DeconvolutionalLayer built
    /// with kernelSize: 4, stride: 2, padding: 1", giving
    /// <c>(in - 1) * 2 - 2 + 4 = 2 * in</c> with no floor anywhere. An exact doubling repeated L times
    /// is an exact multiplication by <c>2^L</c>.
    /// </para>
    /// <para>
    /// Note the consequence for chaining: encode-then-decode is NOT guaranteed to return the original
    /// extent. At an odd input the encoder rounds up and the decoder then doubles that, so the round
    /// trip grows. The contract reports this rather than hiding it, which is the point of stating each
    /// half from its own arithmetic instead of assuming they invert.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 || _outputChannels <= 0) return null;

        int factor = UpsampleFactor;
        if (factor <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outputChannels)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, factor)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, factor)),
        };
    }

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
    /// Number of residual blocks per up-stage. Persisted in the checkpoint so that
    /// <see cref="Deserialize(BinaryReader)"/> can reject files built from a decoder with
    /// a different up-block layout.
    /// </summary>
    private readonly int _numResBlocks;

    /// <summary>
    /// Final output spatial size.
    /// </summary>
    private readonly int _outputSpatialSize;

    /// <summary>
    /// Cached intermediate values for backward pass.
    /// </summary>
    [Scratch]
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

        // Defensive copy — _channelMults must not alias the caller-provided array
        // (later mutation would silently desync runtime properties from the layers
        // already built and from checkpoint metadata).
        var resolvedChannelMults = (channelMults ?? new[] { 1, 2, 4, 4 }).ToArray();
        if (resolvedChannelMults.Length == 0)
            throw new ArgumentException("At least one channel multiplier is required.", nameof(channelMults));
        for (int i = 0; i < resolvedChannelMults.Length; i++)
        {
            if (resolvedChannelMults[i] <= 0)
            {
                throw new ArgumentException(
                    $"channelMults[{i}] = {resolvedChannelMults[i]} must be positive.",
                    nameof(channelMults));
            }
        }
        if (numResBlocks <= 0)
            throw new ArgumentOutOfRangeException(nameof(numResBlocks), "Number of residual blocks must be positive.");
        if (outputSpatialSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(outputSpatialSize), "Output spatial size must be positive.");
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");

        // outputSpatialSize must be evenly divisible by 2^(levels-1) so the
        // bottleneck → output upsample chain returns exactly to outputSpatialSize.
        // Without this, integer division silently truncates and the actual decoded
        // shape diverges from the declared output shape.
        int upsampleFactor = 1 << (resolvedChannelMults.Length - 1);
        if (outputSpatialSize % upsampleFactor != 0)
        {
            throw new ArgumentException(
                $"outputSpatialSize ({outputSpatialSize}) must be divisible by " +
                $"{upsampleFactor} = 2^{resolvedChannelMults.Length - 1} for " +
                $"{resolvedChannelMults.Length}-level upsampling without shape drift.",
                nameof(outputSpatialSize));
        }

        _outputChannels = outputChannels;
        _latentChannels = latentChannels;
        _baseChannels = baseChannels;
        _channelMults = resolvedChannelMults;
        _numGroups = numGroups;
        _numResBlocks = numResBlocks;
        _outputSpatialSize = outputSpatialSize;
        _silu = new SiLUActivation<T>();
        _tanh = new TanhActivation<T>();

        // Bottleneck spatial size: divide by the validated upsample factor.
        _bottleneckSize = outputSpatialSize / upsampleFactor;

        _upBlockOutputs = new Tensor<T>?[_channelMults.Length];

        // Post-quant convolution
        _postQuantConv = new ConvolutionalLayer<T>(
            outputDepth: latentChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activationFunction: new IdentityActivation<T>());

        // Input convolution to expand latent to decoder channels
        int lastChannels = baseChannels * _channelMults[^1];
        _inputConv = new ConvolutionalLayer<T>(
            outputDepth: lastChannels,
            kernelSize: 3,
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
            outputDepth: outputChannels,
            kernelSize: 3,
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
    /// Per-instance compile host. Routing the decoder's full forward through
    /// this host means a second + Nth call at the same latent shape replays a
    /// cached compiled plan instead of re-tracing the entire ResBlock + Conv
    /// + Upsample stack. (#1272 W1: VAEDecoder gets its own CompiledModelHost.)
    /// </summary>
    private AiDotNet.NeuralNetworks.CompiledModelHost<T>? _compileHost;
    private int _compileStructureVersion;

    private AiDotNet.NeuralNetworks.CompiledModelHost<T> EnsureCompileHost() =>
        _compileHost ??= new AiDotNet.NeuralNetworks.CompiledModelHost<T>(
            modelIdentity: nameof(VAEDecoder<T>));

    /// <summary>
    /// Decodes a latent representation to an image.
    /// </summary>
    /// <param name="input">Latent tensor [batch, latentChannels, H, W].</param>
    /// <returns>Decoded image [batch, outputChannels, H*f, W*f] where f is upsample factor.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _lastInput = input;
        return EnsureCompileHost().Predict(input, _compileStructureVersion, () => ForwardEager(input));
    }

    /// <summary>
    /// Eager forward pass body — invoked by the compile host on cache miss /
    /// when compilation is disabled, also reused by the backward path that
    /// depends on the cached intermediate tensors.
    /// </summary>
    private Tensor<T> ForwardEager(Tensor<T> input)
    {
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

    /// <summary>
    /// Async overload of <see cref="Forward(Tensor{T})"/> — routes through
    /// the compile host's <c>PredictAsync</c>.
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
    /// Bumps the structure-version so the next Forward drops stale plans.
    /// </summary>
    public void InvalidateCompiledPlans()
    {
        _compileStructureVersion++;
        _compileHost?.Invalidate();
    }

    private Tensor<T> ApplySiLU(Tensor<T> input)
    {
        return Engine.Swish(input);
    }

    private Tensor<T> ApplyTanh(Tensor<T> input)
    {
        return Engine.Tanh(input);
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

    private Tensor<T> ApplyTanhDerivative(Tensor<T> input, Tensor<T> gradient)
    {
        var output = new Tensor<T>(input._shape);
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
    // GetParameters / SetParameters are deliberately NOT overridden. LayerBase now implements both
    // concretely, folding over the same registry ParameterCount sums -- Parameters, this layer's
    // registered tensors, then each sub-layer -- in one order, so the count, the vector and the
    // restore cannot describe different tensors. The hand-written versions here walked the child
    // fields directly while ParameterCount walked the registry, which is why this layer reported 0
    // against a 56,092-value vector and every model containing it inherited the mismatch.

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

        // Caller (SetParameters above) has already validated the total length, so
        // each layer's slice is guaranteed to fit.
        for (int i = 0; i < layerParams.Length; i++)
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
}
