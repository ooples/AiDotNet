using System.Diagnostics.CodeAnalysis;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Diffusion.Attention;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// U-Net architecture for noise prediction in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The U-Net is the most common architecture for diffusion model noise prediction.
/// It has an encoder-decoder structure with skip connections that help preserve
/// fine-grained details during the denoising process.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of U-Net like a funnel:
/// 1. Encoder (going down): Compresses the image, capturing patterns at different scales
/// 2. Middle: Processes the most compressed representation
/// 3. Decoder (going up): Expands back to original size, using skip connections
///
/// Skip connections are like "shortcuts" that connect encoder layers directly to
/// decoder layers, helping the network preserve fine details that might otherwise
/// be lost during compression.
/// </para>
/// <para>
/// This implementation follows the Stable Diffusion architecture with:
/// - Residual blocks with group normalization
/// - Self-attention at lower resolutions
/// - Cross-attention for text conditioning
/// - Time embedding injection via adaptive normalization
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new UNetNoisePredictor&lt;float&gt;(inputChannels: 4, baseChannels: 320, contextDim: 768);
/// var noisyLatent = Tensor&lt;float&gt;.Random(new[] { 1, 4, 64, 64 });
/// var predicted = predictor.PredictNoise(noisyLatent, timestep: 500);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Denoising Diffusion Probabilistic Models", "https://arxiv.org/abs/2006.11239")]
public class UNetNoisePredictor<T> : NoisePredictorBase<T>
{
    /// <summary>
    /// Channel multipliers for each resolution level.
    /// </summary>
    private readonly int[] _channelMultipliers;

    /// <summary>
    /// Number of residual blocks per resolution level.
    /// </summary>
    private readonly int _numResBlocks;

    /// <summary>
    /// Resolutions at which to apply attention.
    /// </summary>
    private readonly int[] _attentionResolutions;

    /// <summary>
    /// Encoder blocks (downsampling path).
    /// </summary>
    private readonly List<UNetBlock> _encoderBlocks;

    /// <summary>
    /// Middle blocks (bottleneck).
    /// </summary>
    private readonly List<UNetBlock> _middleBlocks;

    /// <summary>
    /// Decoder blocks (upsampling path).
    /// </summary>
    private readonly List<UNetBlock> _decoderBlocks;

    /// <summary>
    /// Input convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _inputConv;

    /// <summary>
    /// Output convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// Time embedding MLP.
    /// </summary>
    private DenseLayer<T>? _timeEmbedMlp1;
    private DenseLayer<T>? _timeEmbedMlp2;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached output for backward pass.
    /// </summary>
    private Tensor<T>? _lastOutput;

    /// <summary>
    /// Number of input channels.
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Number of output channels.
    /// </summary>
    private readonly int _outputChannels;

    /// <summary>
    /// Base channel count.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// Time embedding dimension.
    /// </summary>
    private readonly int _timeEmbeddingDim;

    /// <summary>
    /// Context dimension for cross-attention.
    /// </summary>
    private readonly int _contextDim;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Latent spatial height for the noise predictor (default: 64).
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// The neural network architecture configuration, if provided.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _baseChannels;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _timeEmbeddingDim;

    /// <inheritdoc />
    public override int ParameterCount => CalculateParameterCount();

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => _contextDim > 0;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Initializes a new instance of the UNetNoisePredictor class with full customization support.
    /// </summary>
    /// <param name="architecture">
    /// Optional neural network architecture with custom layers. If the architecture's Layers
    /// list contains layers, those will be used for the encoder blocks. If null or empty,
    /// industry-standard layers from the Stable Diffusion paper are created automatically.
    /// </param>
    /// <param name="inputChannels">Number of input channels (default: 4 for latent diffusion).</param>
    /// <param name="outputChannels">Number of output channels (default: same as input).</param>
    /// <param name="baseChannels">Base channel count (default: 320 for SD).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per level (default: 2).</param>
    /// <param name="attentionResolutions">Resolution indices for attention (default: [1, 2, 3]).</param>
    /// <param name="contextDim">Context dimension for cross-attention (default: 768 for CLIP).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="inputHeight">Latent spatial height (default: 64 for SD 512/8).</param>
    /// <param name="encoderBlocks">
    /// Optional custom encoder blocks. If provided, these blocks are used instead of creating
    /// default blocks. This allows full customization of the encoder path.
    /// </param>
    /// <param name="middleBlocks">
    /// Optional custom middle (bottleneck) blocks.
    /// </param>
    /// <param name="decoderBlocks">
    /// Optional custom decoder blocks.
    /// </param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> All parameters are optional with industry-standard defaults
    /// from the original Stable Diffusion paper. You can create a ready-to-use U-Net
    /// with no arguments, or customize any component:
    ///
    /// <code>
    /// // Default configuration (recommended for most users)
    /// var unet = new UNetNoisePredictor&lt;float&gt;();
    ///
    /// // Custom layers via NeuralNetworkArchitecture
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(..., layers: myCustomLayers);
    /// var unet = new UNetNoisePredictor&lt;float&gt;(architecture: arch);
    ///
    /// // SDXL configuration
    /// var unet = new UNetNoisePredictor&lt;float&gt;(
    ///     baseChannels: 320,
    ///     channelMultipliers: new[] { 1, 2, 4 },
    ///     contextDim: 2048);
    /// </code>
    /// </para>
    /// </remarks>
    public UNetNoisePredictor(
        NeuralNetworkArchitecture<T>? architecture = null,
        int inputChannels = 4,
        int? outputChannels = null,
        int baseChannels = 320,
        int[]? channelMultipliers = null,
        int numResBlocks = 2,
        int[]? attentionResolutions = null,
        int contextDim = 768,
        int numHeads = 8,
        int inputHeight = 64,
        List<UNetBlock>? encoderBlocks = null,
        List<UNetBlock>? middleBlocks = null,
        List<UNetBlock>? decoderBlocks = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null)
        : base(lossFunction, seed)
    {
        _architecture = architecture;
        _inputChannels = inputChannels;
        _outputChannels = outputChannels ?? inputChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? [1, 2, 4, 4];
        _numResBlocks = numResBlocks;
        _attentionResolutions = attentionResolutions ?? [1, 2, 3];
        _contextDim = contextDim;
        _numHeads = numHeads;
        _timeEmbeddingDim = baseChannels * 4;
        _inputHeight = inputHeight;

        _encoderBlocks = new List<UNetBlock>();
        _middleBlocks = new List<UNetBlock>();
        _decoderBlocks = new List<UNetBlock>();

        InitializeLayers(architecture, encoderBlocks, middleBlocks, decoderBlocks);
    }

    /// <summary>
    /// Initializes all layers of the U-Net, using custom layers from the user
    /// if provided or creating industry-standard layers from the Stable Diffusion paper.
    /// </summary>
    /// <param name="architecture">Optional architecture with custom layers.</param>
    /// <param name="customEncoderBlocks">Optional custom encoder blocks.</param>
    /// <param name="customMiddleBlocks">Optional custom middle blocks.</param>
    /// <param name="customDecoderBlocks">Optional custom decoder blocks.</param>
    /// <remarks>
    /// <para>
    /// Layer resolution order:
    /// 1. If custom encoder/middle/decoder blocks are provided directly, use those
    /// 2. If a NeuralNetworkArchitecture with layers is provided, wrap those as encoder blocks
    /// 3. Otherwise, create industry-standard layers from the Stable Diffusion paper
    /// </para>
    /// </remarks>
    [MemberNotNull(nameof(_inputConv), nameof(_outputConv), nameof(_timeEmbedMlp1), nameof(_timeEmbedMlp2))]
    private void InitializeLayers(
        NeuralNetworkArchitecture<T>? architecture,
        List<UNetBlock>? customEncoderBlocks,
        List<UNetBlock>? customMiddleBlocks,
        List<UNetBlock>? customDecoderBlocks)
    {
        // Create input/output convolutions and time embedding MLP via LayerHelper
        var baseLayers = LayerHelper<T>.CreateUNetNoisePredictorEncoderLayers(
            _inputChannels, _baseChannels, _channelMultipliers, _numResBlocks,
            _contextDim, _numHeads).ToList();

        // First layer is input conv, next two are time embedding MLP
        _inputConv = (ConvolutionalLayer<T>)baseLayers[0];
        _timeEmbedMlp1 = (DenseLayer<T>)baseLayers[1];
        _timeEmbedMlp2 = (DenseLayer<T>)baseLayers[2];

        // Output conv from decoder layers
        var decoderBaseLayers = LayerHelper<T>.CreateUNetNoisePredictorDecoderLayers(
            _outputChannels, _baseChannels, _channelMultipliers, _numResBlocks).ToList();
        _outputConv = (ConvolutionalLayer<T>)decoderBaseLayers[^1];

        // Priority 1: Use custom blocks passed directly
        if (customEncoderBlocks != null && customEncoderBlocks.Count > 0 &&
            customMiddleBlocks != null && customMiddleBlocks.Count > 0 &&
            customDecoderBlocks != null && customDecoderBlocks.Count > 0)
        {
            _encoderBlocks.AddRange(customEncoderBlocks);
            _middleBlocks.AddRange(customMiddleBlocks);
            _decoderBlocks.AddRange(customDecoderBlocks);
            return;
        }

        // Priority 2: Use layers from NeuralNetworkArchitecture as encoder ResBlocks
        if (architecture?.Layers != null && architecture.Layers.Count > 0)
        {
            foreach (var layer in architecture.Layers)
            {
                _encoderBlocks.Add(new UNetBlock { ResBlock = layer });
            }
            CreateDefaultMiddleBlocks(_baseChannels * _channelMultipliers[^1]);
            CreateDefaultDecoderBlocks();
            return;
        }

        // Priority 3: Create industry-standard layers from the Stable Diffusion paper
        CreateDefaultEncoderBlocks();
        CreateDefaultMiddleBlocks(_baseChannels * _channelMultipliers[^1]);
        CreateDefaultDecoderBlocks();
    }

    /// <summary>
    /// Creates industry-standard encoder blocks based on the Stable Diffusion U-Net.
    /// </summary>
    private void CreateDefaultEncoderBlocks()
    {
        var inChannels = _baseChannels;
        int spatialSize = _inputHeight;
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block < _numResBlocks; block++)
            {
                _encoderBlocks.Add(new UNetBlock
                {
                    ResBlock = CreateResBlock(inChannels, outChannels, spatialSize),
                    AttentionBlock = useAttention ? CreateAttentionBlock(outChannels, spatialSize) : null,
                    CrossAttentionBlock = useAttention && _contextDim > 0 ? CreateCrossAttentionBlock(outChannels, spatialSize) : null
                });
                inChannels = outChannels;
            }

            // Add downsampling except for last level
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderBlocks.Add(new UNetBlock
                {
                    Downsample = CreateDownsample(outChannels, spatialSize)
                });
                spatialSize = Math.Max(1, spatialSize / 2);
            }
        }
    }

    /// <summary>
    /// Creates industry-standard middle (bottleneck) blocks.
    /// </summary>
    private void CreateDefaultMiddleBlocks(int channels)
    {
        // Spatial size at the bottleneck = _inputHeight / (2^(numDownsamples))
        int numDownsamples = _channelMultipliers.Length - 1;
        int bottleneckSpatial = Math.Max(1, _inputHeight >> numDownsamples);

        _middleBlocks.Add(new UNetBlock
        {
            ResBlock = CreateResBlock(channels, channels, bottleneckSpatial),
            AttentionBlock = CreateAttentionBlock(channels, bottleneckSpatial),
            CrossAttentionBlock = _contextDim > 0 ? CreateCrossAttentionBlock(channels, bottleneckSpatial) : null
        });
        _middleBlocks.Add(new UNetBlock
        {
            ResBlock = CreateResBlock(channels, channels, bottleneckSpatial)
        });
    }

    /// <summary>
    /// Creates industry-standard decoder blocks (reverse of encoder).
    /// </summary>
    private void CreateDefaultDecoderBlocks()
    {
        var inChannels = _baseChannels * _channelMultipliers[^1];
        int numDownsamples = _channelMultipliers.Length - 1;
        int spatialSize = Math.Max(1, _inputHeight >> numDownsamples);

        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            // Each level has _numResBlocks decoder blocks that consume skip connections
            // from the matching encoder level (same spatial resolution)
            for (int block = 0; block < _numResBlocks; block++)
            {
                var skipChannels = outChannels;

                _decoderBlocks.Add(new UNetBlock
                {
                    ResBlock = CreateResBlock(inChannels + skipChannels, outChannels, spatialSize),
                    AttentionBlock = useAttention ? CreateAttentionBlock(outChannels, spatialSize) : null,
                    CrossAttentionBlock = useAttention && _contextDim > 0 ? CreateCrossAttentionBlock(outChannels, spatialSize) : null
                });
                inChannels = outChannels;
            }

            // Add upsampling except for first level
            if (level > 0)
            {
                _decoderBlocks.Add(new UNetBlock
                {
                    Upsample = CreateUpsample(outChannels, spatialSize)
                });
                spatialSize = Math.Min(_inputHeight, spatialSize * 2);
            }
        }
    }


    /// <summary>
    /// Per-instance compile cache for the UNet forward pass. Lazy-allocated
    /// on first <see cref="PredictNoise"/> call when
    /// <see cref="AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.EnableCompilation"/>
    /// is on. Keyed by noisy-sample input shape — different shapes compile
    /// different plans. Dropped when <see cref="ResetState"/> runs.
    /// </summary>
    private AiDotNet.Tensors.Engines.Compilation.CompiledModelCache<T>? _compiledInferenceCache;

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Compute timestep embedding (already cached per-timestep in the base class).
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        // Compiled fast path — only when TensorCodecOptions.EnableCompilation is on.
        // First call traces the eager ForwardUNet and compiles the plan; subsequent
        // calls at the same shape replay the compiled plan.
        var output = PredictCompiledForward(noisySample, timeEmbed, conditioning);

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Eagerly compiles the UNet forward pass for the given sample input shape,
    /// storing the plan in the per-instance cache. Addresses the
    /// "UNetNoisePredictor compiled forward" checklist item on
    /// github.com/ooples/AiDotNet#1015.
    /// </summary>
    /// <param name="sampleNoisy">Representative noisy-sample tensor whose shape keys the plan.</param>
    /// <param name="sampleTimestep">Timestep used for tracing. Any valid integer works — the plan's replay uses whichever timestep <see cref="PredictNoise"/> is called with.</param>
    /// <param name="conditioning">Optional conditioning tensor of the shape production will use.</param>
    /// <returns><c>true</c> when a compiled plan is cached; <c>false</c> when compilation is disabled or tracing throws.</returns>
    /// <remarks>
    /// Call this at startup with a representative input shape to avoid the
    /// one-time trace+compile cost hitting the first real inference. Multiple
    /// calls with different shapes pre-warm multiple plans in the same cache.
    /// </remarks>
    public bool CompileForward(Tensor<T> sampleNoisy, int sampleTimestep = 0, Tensor<T>? conditioning = null)
    {
        if (sampleNoisy is null)
            throw new ArgumentNullException(nameof(sampleNoisy));
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return false;

        try
        {
            var timeEmbed = GetTimestepEmbedding(sampleTimestep);
            timeEmbed = ProjectTimeEmbedding(timeEmbed);

            var cache = _compiledInferenceCache ??= new AiDotNet.Tensors.Engines.Compilation.CompiledModelCache<T>();
            // Key includes conditioning presence so null-vs-present paths
            // don't share plans (different traced graphs).
            var key = BuildCompileKey(sampleNoisy, conditioning);
            var plan = cache.GetOrCompileInference(
                key,
                () => ForwardUNet(sampleNoisy, timeEmbed, conditioning));
            // Execute once to validate the plan replays correctly and warm
            // workspace buffers. Dispose the output to avoid leaking the
            // pooled tensor allocation from the warm-up.
            var warmupOutput = plan.Execute();
            if (warmupOutput is IDisposable disposableOutput)
                disposableOutput.Dispose();
            return true;
        }
        catch (Exception ex) when (
            ex is not OutOfMemoryException &&
            ex is not StackOverflowException &&
            ex is not AccessViolationException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"UNetNoisePredictor.CompileForward failed for shape [{string.Join(",", sampleNoisy._shape)}]: " +
                $"{ex.GetType().Name}: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Internal: dispatches to the compiled plan when available + enabled,
    /// falling back to eager <see cref="ForwardUNet"/> otherwise. Mirrors
    /// <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.PredictCompiled"/>.
    /// </summary>
    private Tensor<T> PredictCompiledForward(Tensor<T> noisySample, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return ForwardUNet(noisySample, timeEmbed, conditioning);

        try
        {
            var cache = _compiledInferenceCache ??= new AiDotNet.Tensors.Engines.Compilation.CompiledModelCache<T>();
            // Key includes conditioning presence so null-vs-present paths
            // don't share plans (different traced graphs).
            var key = BuildCompileKey(noisySample, conditioning);
            var plan = cache.GetOrCompileInference(
                key,
                () => ForwardUNet(noisySample, timeEmbed, conditioning));
            return plan.Execute();
        }
        catch (Exception ex) when (
            ex is not OutOfMemoryException &&
            ex is not StackOverflowException &&
            ex is not AccessViolationException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"UNetNoisePredictor.PredictCompiledForward fallback: " +
                $"{ex.GetType().Name}: {ex.Message}");
            return ForwardUNet(noisySample, timeEmbed, conditioning);
        }
    }

    /// <summary>
    /// Builds a composite cache key that includes BOTH the noisy-sample
    /// shape AND the conditioning shape (or a sentinel for null). This
    /// prevents a plan traced with conditioning=null from being replayed
    /// with a conditioning tensor (different graph) or vice versa.
    /// </summary>
    private static int[] BuildCompileKey(Tensor<T> noisySample, Tensor<T>? conditioning)
    {
        var inputShape = noisySample._shape;
        if (conditioning is null)
        {
            // Append a single -1 sentinel so "no conditioning" has a
            // distinct key from any real conditioning shape.
            var key = new int[inputShape.Length + 1];
            Array.Copy(inputShape, key, inputShape.Length);
            key[^1] = -1;
            return key;
        }
        var condShape = conditioning._shape;
        var compositeKey = new int[inputShape.Length + 1 + condShape.Length];
        Array.Copy(inputShape, 0, compositeKey, 0, inputShape.Length);
        compositeKey[inputShape.Length] = -1; // separator
        Array.Copy(condShape, 0, compositeKey, inputShape.Length + 1, condShape.Length);
        return compositeKey;
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Project time embedding
        var timeEmbed = ProjectTimeEmbedding(timeEmbedding);

        // Forward pass — saves skip connections for backward
        var (output, skips) = ForwardUNetWithSkips(noisySample, timeEmbed, conditioning);
        SaveForwardState(skips, timeEmbed);

        _lastOutput = output;
        return output;
    }

    /// <summary>
    /// Projects the sinusoidal timestep embedding through the MLP.
    /// </summary>
    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        if (_timeEmbedMlp1 == null || _timeEmbedMlp2 == null)
        {
            throw new InvalidOperationException("Time embedding layers not initialized.");
        }

        var x = _timeEmbedMlp1.Forward(timeEmbed);
        x = _timeEmbedMlp2.Forward(x);
        return x;
    }

    /// <summary>
    /// Performs the forward pass through the U-Net architecture.
    /// </summary>
    /// <summary>
    /// Forward pass returning skip connections for backward pass gradient routing.
    /// </summary>
    private (Tensor<T> output, List<Tensor<T>> skips) ForwardUNetWithSkips(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        if (_inputConv == null || _outputConv == null)
            throw new InvalidOperationException("Convolutional layers not initialized.");

        x = _inputConv.Forward(x);
        var skips = new List<Tensor<T>>(_encoderBlocks.Count);

        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            var block = _encoderBlocks[i];
            if (block.Downsample != null)
            {
                x = block.Downsample.Forward(x);
            }
            else
            {
                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null) x = block.AttentionBlock.Forward(x);
                if (block.CrossAttentionBlock != null && conditioning != null)
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
                skips.Add(x);
            }
        }

        for (int i = 0; i < _middleBlocks.Count; i++)
        {
            var block = _middleBlocks[i];
            x = ApplyResBlock(block.ResBlock, x, timeEmbed);
            if (block.AttentionBlock != null) x = block.AttentionBlock.Forward(x);
            if (block.CrossAttentionBlock != null && conditioning != null)
                x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
        }

        var skipIdx = skips.Count - 1;
        for (int i = 0; i < _decoderBlocks.Count; i++)
        {
            var block = _decoderBlocks[i];
            if (block.Upsample != null)
            {
                x = block.Upsample.Forward(x);
            }
            else
            {
                if (skipIdx >= 0)
                {
                    x = ConcatenateChannels(x, skips[skipIdx]);
                    skipIdx--;
                }
                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null) x = block.AttentionBlock.Forward(x);
                if (block.CrossAttentionBlock != null && conditioning != null)
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
            }
        }

        x = _outputConv.Forward(x);
        return (x, skips);
    }

    /// <summary>
    /// Forward pass for inference (no skip storage needed).
    /// </summary>
    private Tensor<T> ForwardUNet(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        if (_inputConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Convolutional layers not initialized.");
        }

        // Input convolution
        x = _inputConv.Forward(x);

        // Store skip connections — pre-allocate capacity to avoid List resizing
        var skips = new List<Tensor<T>>(_encoderBlocks.Count);

        // Encoder
        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            var block = _encoderBlocks[i];
            if (block.Downsample != null)
            {
                x = block.Downsample.Forward(x);
            }
            else
            {
                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null)
                {
                    x = block.AttentionBlock.Forward(x);
                }
                if (block.CrossAttentionBlock != null && conditioning != null)
                {
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
                }
                skips.Add(x);
            }
        }

        // Middle
        for (int i = 0; i < _middleBlocks.Count; i++)
        {
            var block = _middleBlocks[i];
            x = ApplyResBlock(block.ResBlock, x, timeEmbed);
            if (block.AttentionBlock != null)
            {
                x = block.AttentionBlock.Forward(x);
            }
            if (block.CrossAttentionBlock != null && conditioning != null)
            {
                x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
            }
        }

        // Decoder
        var skipIdx = skips.Count - 1;
        for (int i = 0; i < _decoderBlocks.Count; i++)
        {
            var block = _decoderBlocks[i];
            if (block.Upsample != null)
            {
                x = block.Upsample.Forward(x);
            }
            else
            {
                // Concatenate skip connection
                if (skipIdx >= 0)
                {
                    x = ConcatenateChannels(x, skips[skipIdx]);
                    skipIdx--;
                }

                x = ApplyResBlock(block.ResBlock, x, timeEmbed);
                if (block.AttentionBlock != null)
                {
                    x = block.AttentionBlock.Forward(x);
                }
                if (block.CrossAttentionBlock != null && conditioning != null)
                {
                    x = ApplyCrossAttention(block.CrossAttentionBlock, x, conditioning);
                }
            }
        }

        // Output convolution
        x = _outputConv.Forward(x);

        return x;
    }

    /// <summary>
    /// Applies a residual block with time embedding conditioning.
    /// </summary>
    private Tensor<T> ApplyResBlock(ILayer<T>? resBlock, Tensor<T> x, Tensor<T> timeEmbed)
    {
        if (resBlock == null) return x;

        // Use time-conditioned forward if the block supports it (DiffusionResBlock)
        if (resBlock is DiffusionResBlock<T> diffResBlock)
        {
            return diffResBlock.Forward(x, timeEmbed);
        }

        return resBlock.Forward(x);
    }

    /// <summary>
    /// Applies cross-attention between the sample and conditioning.
    /// </summary>
    /// <param name="crossAttn">The cross-attention layer.</param>
    /// <param name="x">Spatial features [batch, channels, height, width].</param>
    /// <param name="conditioning">Text embeddings [batch, seq_len, context_dim].</param>
    /// <returns>Attended spatial features with same shape as x.</returns>
    private Tensor<T> ApplyCrossAttention(ILayer<T>? crossAttn, Tensor<T> x, Tensor<T> conditioning)
    {
        if (crossAttn == null) return x;

        // CrossAttentionLayer handles:
        // - Query: spatial features from x (reshaped internally)
        // - Key/Value: text embeddings from conditioning
        // - Output: attended features with same shape as x

        // DiffusionCrossAttention uses ForwardWithContext for conditioning
        if (crossAttn is DiffusionCrossAttention<T> diffusionCrossAttn)
        {
            return diffusionCrossAttn.ForwardWithContext(x, conditioning);
        }

        // CrossAttentionLayer has Forward(input, context) overload
        if (crossAttn is CrossAttentionLayer<T> crossAttnLayer)
        {
            return crossAttnLayer.Forward(x, conditioning);
        }

        // Fallback for other layers
        return crossAttn.Forward(x);
    }

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate([a, b], axis: 1);
    }

    #region Layer Factory Methods

    private ILayer<T> CreateResBlock(int inChannels, int outChannels)
    {
        return CreateResBlock(inChannels, outChannels, _inputHeight);
    }

    private ILayer<T> CreateResBlock(int inChannels, int outChannels, int spatialSize)
    {
        // Per DDPM (Ho et al. 2020) and Stable Diffusion (Rombach et al. 2022):
        // ResBlock = GroupNorm → SiLU → Conv3x3 → (+time) → GroupNorm → SiLU → Conv3x3 → (+skip)
        return new DiffusionResBlock<T>(
            inChannels: inChannels,
            outChannels: outChannels,
            spatialSize: spatialSize,
            timeEmbedDim: _timeEmbeddingDim);
    }

    private ILayer<T> CreateAttentionBlock(int channels, int spatialSize)
    {
        return new DiffusionAttention<T>(
            channels: channels,
            numHeads: _numHeads,
            spatialSize: spatialSize,
            flashAttentionThreshold: spatialSize * spatialSize / 16);
    }

    private ILayer<T> CreateCrossAttentionBlock(int channels, int spatialSize)
    {
        return new DiffusionCrossAttention<T>(
            queryDim: channels,
            contextDim: _contextDim,
            numHeads: _numHeads,
            spatialSize: spatialSize);
    }

    private ILayer<T> CreateDownsample(int channels, int spatialSize)
    {
        // LazyConv2D: kernel tensor stays unallocated until first Forward() call.
        return LazyConv2D(
            inputDepth: channels,
            inputHeight: spatialSize,
            inputWidth: spatialSize,
            outputDepth: channels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            activation: new IdentityActivation<T>());
    }

    private ILayer<T> CreateUpsample(int channels, int spatialSize)
    {
        // spatialSize here is the current (smaller) spatial size before upsampling
        return new DeconvolutionalLayer<T>(
            inputShape: new[] { 1, channels, spatialSize, spatialSize },
            outputDepth: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    #endregion

    #region Parameter Management

    private int CalculateParameterCount()
    {
        // Approximate parameter count based on architecture
        long count = 0;

        // Input/output convolutions
        count += _inputChannels * _baseChannels * 9 + _baseChannels;
        count += _baseChannels * _outputChannels * 9 + _outputChannels;

        // Time embedding MLP
        count += (_timeEmbeddingDim / 4) * _timeEmbeddingDim + _timeEmbeddingDim;
        count += _timeEmbeddingDim * _timeEmbeddingDim + _timeEmbeddingDim;

        // Estimate blocks
        foreach (var channels in _channelMultipliers.Select(mult => _baseChannels * mult))
        {
            count += _numResBlocks * (channels * channels * 2); // Rough estimate per res block
        }

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Collect all sublayer parameter vectors first (each layer allocates its own)
        var layerParams = new List<Vector<T>>();
        int totalCount = 0;

        CollectLayerParams(layerParams, ref totalCount, _inputConv);
        CollectLayerParams(layerParams, ref totalCount, _timeEmbedMlp1);
        CollectLayerParams(layerParams, ref totalCount, _timeEmbedMlp2);

        for (int i = 0; i < _encoderBlocks.Count; i++)
            CollectBlockParams(layerParams, ref totalCount, _encoderBlocks[i]);
        for (int i = 0; i < _middleBlocks.Count; i++)
            CollectBlockParams(layerParams, ref totalCount, _middleBlocks[i]);
        for (int i = 0; i < _decoderBlocks.Count; i++)
            CollectBlockParams(layerParams, ref totalCount, _decoderBlocks[i]);

        CollectLayerParams(layerParams, ref totalCount, _outputConv);

        // Single allocation at exact size, then copy from cached sublayer vectors
        var parameters = new Vector<T>(totalCount);
        int idx = 0;
        for (int v = 0; v < layerParams.Count; v++)
        {
            var p = layerParams[v];
            for (int i = 0; i < p.Length; i++)
                parameters[idx++] = p[i];
        }

        return parameters;
    }

    private static void CollectLayerParams(List<Vector<T>> dest, ref int totalCount, ILayer<T>? layer)
    {
        if (layer == null) return;
        var p = layer.GetParameters();
        dest.Add(p);
        totalCount += p.Length;
    }

    private static void CollectBlockParams(List<Vector<T>> dest, ref int totalCount, UNetBlock block)
    {
        CollectLayerParams(dest, ref totalCount, block.ResBlock);
        CollectLayerParams(dest, ref totalCount, block.AttentionBlock);
        CollectLayerParams(dest, ref totalCount, block.CrossAttentionBlock);
        CollectLayerParams(dest, ref totalCount, block.Downsample);
        CollectLayerParams(dest, ref totalCount, block.Upsample);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var index = 0;

        SetLayerParameters(_inputConv, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp1, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp2, parameters, ref index);

        foreach (var block in _encoderBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        foreach (var block in _middleBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        foreach (var block in _decoderBlocks)
        {
            SetBlockParameters(block, parameters, ref index);
        }

        SetLayerParameters(_outputConv, parameters, ref index);
    }

    private void SetLayerParameters(ILayer<T>? layer, Vector<T> parameters, ref int index)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        var newParams = new Vector<T>(layerParams.Length);
        for (int i = 0; i < layerParams.Length && index < parameters.Length; i++)
        {
            newParams[i] = parameters[index++];
        }
        layer.SetParameters(newParams);
    }

    private void SetBlockParameters(UNetBlock block, Vector<T> parameters, ref int index)
    {
        SetLayerParameters(block.ResBlock, parameters, ref index);
        SetLayerParameters(block.AttentionBlock, parameters, ref index);
        SetLayerParameters(block.CrossAttentionBlock, parameters, ref index);
        SetLayerParameters(block.Downsample, parameters, ref index);
        SetLayerParameters(block.Upsample, parameters, ref index);
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new UNetNoisePredictor<T>(
            inputChannels: _inputChannels,
            outputChannels: _outputChannels,
            baseChannels: _baseChannels,
            channelMultipliers: _channelMultipliers,
            numResBlocks: _numResBlocks,
            attentionResolutions: _attentionResolutions,
            contextDim: _contextDim,
            numHeads: _numHeads,
            inputHeight: _inputHeight,
            lossFunction: LossFunction);

        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion

    #region Layer-Level Backpropagation

    // Skip connections saved during forward for proper gradient splitting
    private List<Tensor<T>>? _lastSkips;
    private Tensor<T>? _lastTimeEmbed;

    /// <summary>
    /// Stores skip connections and time embedding during forward for backward pass.
    /// Call this from PredictNoiseWithEmbedding during training.
    /// </summary>
    internal void SaveForwardState(List<Tensor<T>> skips, Tensor<T> timeEmbed)
    {
        _lastSkips = skips;
        _lastTimeEmbed = timeEmbed;
    }

    private (Tensor<T> decoder, Tensor<T> skip) SplitChannels(Tensor<T> grad, int decoderChannels, int skipChannels)
    {
        // Split along axis 1 (channel dimension): grad [B, C_dec+C_skip, H, W]
        int batch = grad.Shape[0];
        int totalChannels = grad.Shape[1];
        int height = grad.Shape.Length > 2 ? grad.Shape[2] : 1;
        int width = grad.Shape.Length > 3 ? grad.Shape[3] : 1;
        int spatialSize = height * width;

        var decoderGrad = new Tensor<T>([batch, decoderChannels, height, width]);
        var skipGrad = new Tensor<T>([batch, skipChannels, height, width]);

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < decoderChannels; c++)
            {
                int srcOffset = ((b * totalChannels) + c) * spatialSize;
                int dstOffset = ((b * decoderChannels) + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                    decoderGrad[dstOffset + s] = grad[srcOffset + s];
            }
            for (int c = 0; c < skipChannels; c++)
            {
                int srcOffset = ((b * totalChannels) + decoderChannels + c) * spatialSize;
                int dstOffset = ((b * skipChannels) + c) * spatialSize;
                for (int s = 0; s < spatialSize; s++)
                    skipGrad[dstOffset + s] = grad[srcOffset + s];
            }
        }

        return (decoderGrad, skipGrad);
    }

    private int FindEncoderBlockForSkip(int skipIndex)
    {
        // Skip connections come from non-downsample encoder blocks in order
        int skipCount = 0;
        for (int i = 0; i < _encoderBlocks.Count; i++)
        {
            if (_encoderBlocks[i].Downsample == null)
            {
                if (skipCount == skipIndex)
                    return i;
                skipCount++;
            }
        }
        return -1;
    }

    private Tensor<T>? AccumulateTimeEmbedGradients()
    {
        // Collect time embedding gradients from all DiffusionResBlocks
        Tensor<T>? accumulated = null;
        var numOps = NumOps;

        void AccumulateFromBlock(ILayer<T>? layer)
        {
            if (layer is DiffusionResBlock<T> resBlock)
            {
                var timeGrad = resBlock.GetTimeEmbedGradient();
                if (timeGrad is not null)
                {
                    if (accumulated is null)
                    {
                        accumulated = timeGrad;
                    }
                    else
                    {
                        for (int i = 0; i < accumulated.Length && i < timeGrad.Length; i++)
                            accumulated[i] = numOps.Add(accumulated[i], timeGrad[i]);
                    }
                }
            }
        }

        foreach (var block in _encoderBlocks) AccumulateFromBlock(block.ResBlock);
        foreach (var block in _middleBlocks) AccumulateFromBlock(block.ResBlock);
        foreach (var block in _decoderBlocks) AccumulateFromBlock(block.ResBlock);

        return accumulated;
    }

    /// <inheritdoc />
    protected override Vector<T> GetParameterGradients()
    {
        // Collect gradients in the same order as GetParameters
        var layerGrads = new List<Vector<T>>();
        int totalCount = 0;

        CollectLayerGradients(layerGrads, ref totalCount, _inputConv);
        CollectLayerGradients(layerGrads, ref totalCount, _timeEmbedMlp1);
        CollectLayerGradients(layerGrads, ref totalCount, _timeEmbedMlp2);

        for (int i = 0; i < _encoderBlocks.Count; i++)
            CollectBlockGradients(layerGrads, ref totalCount, _encoderBlocks[i]);
        for (int i = 0; i < _middleBlocks.Count; i++)
            CollectBlockGradients(layerGrads, ref totalCount, _middleBlocks[i]);
        for (int i = 0; i < _decoderBlocks.Count; i++)
            CollectBlockGradients(layerGrads, ref totalCount, _decoderBlocks[i]);

        CollectLayerGradients(layerGrads, ref totalCount, _outputConv);

        var gradients = new Vector<T>(totalCount);
        int idx = 0;
        for (int v = 0; v < layerGrads.Count; v++)
        {
            var g = layerGrads[v];
            for (int i = 0; i < g.Length; i++)
                gradients[idx++] = g[i];
        }

        return gradients;
    }

    private static void CollectLayerGradients(List<Vector<T>> dest, ref int totalCount, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        dest.Add(g);
        totalCount += g.Length;
    }

    private static void CollectBlockGradients(List<Vector<T>> dest, ref int totalCount, UNetBlock block)
    {
        CollectLayerGradients(dest, ref totalCount, block.ResBlock);
        CollectLayerGradients(dest, ref totalCount, block.AttentionBlock);
        CollectLayerGradients(dest, ref totalCount, block.CrossAttentionBlock);
        CollectLayerGradients(dest, ref totalCount, block.Downsample);
        CollectLayerGradients(dest, ref totalCount, block.Upsample);
    }

    #endregion

    /// <summary>
    /// Structure for U-Net blocks containing residual, attention, and sampling layers.
    /// </summary>
    public class UNetBlock
    {
        public ILayer<T>? ResBlock { get; set; }
        public ILayer<T>? AttentionBlock { get; set; }
        public ILayer<T>? CrossAttentionBlock { get; set; }
        public ILayer<T>? Downsample { get; set; }
        public ILayer<T>? Upsample { get; set; }
    }
}
