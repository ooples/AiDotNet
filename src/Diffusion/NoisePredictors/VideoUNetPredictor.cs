using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// 3D U-Net architecture for video noise prediction in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The VideoUNetPredictor extends the standard U-Net architecture to handle
/// video data by incorporating 3D convolutions and temporal attention.
/// This is the core noise prediction network used in video diffusion models
/// like Stable Video Diffusion.
/// </para>
/// <para>
/// <b>For Beginners:</b> While a regular U-Net processes single images,
/// VideoUNet processes sequences of frames as a 3D volume:
///
/// Regular U-Net:
/// - Input: [batch, channels, height, width]
/// - 2D convolutions across spatial dimensions only
/// - Each image processed independently
///
/// Video U-Net:
/// - Input: [batch, channels, frames, height, width]
/// - 3D convolutions across space AND time
/// - Frames are processed together, understanding motion
///
/// Key features:
/// - Temporal convolutions capture motion patterns
/// - Temporal attention for long-range frame relationships
/// - Skip connections across both space and time
/// - Image conditioning for image-to-video generation
///
/// Used in: Stable Video Diffusion, ModelScope, VideoCrafter
/// </para>
/// <para>
/// Architecture details:
/// - Encoder: 3D ResBlocks with temporal + spatial attention
/// - Middle: Multiple 3D attention blocks
/// - Decoder: 3D ResBlocks with skip connections
/// - Temporal convolutions with kernel size 3 across frames
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var predictor = new VideoUNetPredictor&lt;float&gt;(inputChannels: 4, baseChannels: 320, numFrames: 14);
/// var noisyVideo = Tensor&lt;float&gt;.Random(new[] { 1, 4, 14, 64, 64 });
/// var predicted = predictor.PredictNoise(noisyVideo, timestep: 500);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Generative)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelTask(ModelTask.Denoising)]
[ModelTask(ModelTask.VideoGeneration)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Video Diffusion Models", "https://arxiv.org/abs/2204.03458")]
public class VideoUNetPredictor<T> : NoisePredictorBase<T>
{

    /// <inheritdoc />
    /// <remarks>Lazy weights, same reasoning as UNetNoisePredictor.</remarks>
    protected override void EnsureParametersReady()
    {
        // A concatenated image condition changes the entry convolution from C to C+conditionC.
        // Resolve through that real public path; an unconditioned dummy would resize the lazy
        // convolution back to C and make ParameterCount disagree with GetParameters by exactly the
        // missing condition-channel kernel slice.
        TriggerLazyShapeResolution(
            includeImageConditioning: _supportsImageConditioning && _concatenateImageCondition);
    }
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
    /// Number of temporal transformer layers.
    /// </summary>
    private readonly int _numTemporalLayers;

    /// <summary>Architecture-specific topology rules used by this shared predictor.</summary>
    private readonly VideoUNetArchitectureProfile _architectureProfile;

    /// <summary>
    /// Encoder blocks.
    /// </summary>
    private readonly List<VideoBlock> _encoderBlocks;

    /// <summary>
    /// Middle blocks.
    /// </summary>
    private readonly List<VideoBlock> _middleBlocks;

    /// <summary>
    /// Decoder blocks.
    /// </summary>
    private readonly List<VideoBlock> _decoderBlocks;

    private bool _lazyShapeResolved;
    private bool _lazyShapeResolvedWithVideo;
    private bool _lazyShapeResolvedWithTextConditioning;
    private bool _lazyShapeResolvedWithImageConditioning;
    private IReadOnlyList<ILayer<T>>? _temporalTrainingLayers;

    /// <summary>
    /// Input convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _inputConv;

    /// <summary>
    /// Output convolution.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>Released U-Net output normalization before SiLU and conv-out.</summary>
    private GroupNormalizationLayer<T>? _outputNorm;

    /// <summary>
    /// Time embedding MLP.
    /// </summary>
    private DenseLayer<T>? _timeEmbedMlp1;
    private DenseLayer<T>? _timeEmbedMlp2;

    /// <summary>
    /// Image conditioning projection (for image-to-video).
    /// </summary>
    private ConvolutionalLayer<T>? _imageCondProjection;

    /// <summary>Learned embedding for super-resolution degradation/noise levels.</summary>
    private EmbeddingLayer<T>? _classEmbedding;

    /// <summary>
    /// Cached input for backward pass.
    /// </summary>
    private Tensor<T>? _lastInput;

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
    /// Whether to support image conditioning.
    /// </summary>
    private readonly bool _supportsImageConditioning;

    /// <summary>Number of channels in the image/video condition.</summary>
    private readonly int _imageConditionChannels;

    /// <summary>
    /// Whether the condition is concatenated with the latent before the input convolution.
    /// Stable-Diffusion x4 / Upscale-A-Video use this seven-channel (4 + 3) contract.
    /// </summary>
    private readonly bool _concatenateImageCondition;

    /// <summary>Number of learned class/noise-level embeddings, or zero when disabled.</summary>
    private readonly int _numClassEmbeddings;

    /// <summary>
    /// Latent spatial height.
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// Latent spatial width.
    /// </summary>
    private readonly int _inputWidth;

    /// <summary>
    /// Typical number of video frames for temporal attention.
    /// </summary>
    private readonly int _numFrames;

    /// <summary>
    /// CLIP text token sequence length for cross-attention.
    /// </summary>
    private readonly int _clipTokenLength;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _outputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _baseChannels;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _timeEmbeddingDim;

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => _contextDim > 0;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Gets whether this predictor supports image conditioning for image-to-video.
    /// </summary>
    public bool SupportsImageConditioning => _supportsImageConditioning;

    /// <summary>Gets the expected image/video condition channel count.</summary>
    public int ImageConditionChannels => _imageConditionChannels;

    /// <summary>Gets whether conditioning is concatenated before the input convolution.</summary>
    public bool ConcatenatesImageCondition => _concatenateImageCondition;

    /// <summary>Gets the number of learned degradation/noise-level embeddings.</summary>
    public int NumClassEmbeddings => _numClassEmbeddings;

    /// <summary>Gets the paper-defined channel multiplier at each U-Net resolution.</summary>
    public IReadOnlyList<int> ChannelMultipliers => Array.AsReadOnly(_channelMultipliers);

    /// <summary>
    /// Gets the number of temporal transformer layers.
    /// </summary>
    public int NumTemporalLayers => _numTemporalLayers;

    /// <summary>Gets the number of spatial residual layers in each down block.</summary>
    public int NumResBlocks => _numResBlocks;

    /// <summary>Gets the selected topology profile.</summary>
    public VideoUNetArchitectureProfile ArchitectureProfile => _architectureProfile;

    /// <summary>Gets the number of explicit temporal modules in the materialized topology.</summary>
    public int TemporalModuleCount =>
        _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks)
            .Count(block => block.TemporalResBlock is TemporalModule3DLayer<T>);

    /// <summary>Gets the number of spatial residual blocks in the materialized topology.</summary>
    public int SpatialResBlockCount =>
        _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks)
            .Count(block => block.SpatialResBlock is DiffusionResBlock<T>);

    /// <summary>Gets the number of released Transformer3D blocks.</summary>
    public int VideoTransformerCount =>
        _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks)
            .Count(block => block.SpatialAttention is VideoTransformer3DLayer<T>);

    /// <summary>Gets the number of Transformer3D blocks whose first attention is text-only.</summary>
    public int OnlyCrossAttentionTransformerCount =>
        _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks)
            .Select(block => block.SpatialAttention)
            .OfType<VideoTransformer3DLayer<T>>()
            .Count(layer => layer.OnlyCrossAttention);

    /// <summary>Gets the number of spatial downsampling operations.</summary>
    public int DownsampleCount => _encoderBlocks.Count(block => block.Downsample is not null);

    /// <summary>Gets the number of spatial upsampling operations.</summary>
    public int UpsampleCount => _decoderBlocks.Count(block => block.Upsample is not null);

    /// <summary>Gets whether any temporal transformer-attention layer is present.</summary>
    public bool UsesTemporalTransformerAttention =>
        _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks)
            .Any(block => block.TemporalAttention is not null ||
                block.SpatialAttention is VideoTransformer3DLayer<T> transformer &&
                transformer.UsesTemporalAttention);

    /// <summary>
    /// Gets the inserted temporal layers while excluding pretrained spatial U-Net layers.
    /// </summary>
    public IReadOnlyList<ILayer<T>> TemporalTrainingLayers =>
        _temporalTrainingLayers ??= BuildTemporalTrainingLayers();

    /// <summary>
    /// Initializes a new instance of the VideoUNetPredictor class.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (default: 4 for latent diffusion).</param>
    /// <param name="outputChannels">Number of output channels (default: same as input).</param>
    /// <param name="baseChannels">Base channel count (default: 320).</param>
    /// <param name="channelMultipliers">Channel multipliers per level (default: [1, 2, 4, 4]).</param>
    /// <param name="numResBlocks">Number of residual blocks per level (default: 2).</param>
    /// <param name="attentionResolutions">Resolution indices for attention (default: [1, 2, 3]).</param>
    /// <param name="numTemporalLayers">Number of temporal transformer layers (default: 1).</param>
    /// <param name="contextDim">Context dimension for cross-attention (default: 1024).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="supportsImageConditioning">Whether to support image conditioning (default: true).</param>
    /// <param name="inputHeight">Latent spatial height (default: 64 for 512/8).</param>
    /// <param name="inputWidth">Latent spatial width (default: 64 for 512/8).</param>
    /// <param name="numFrames">Typical number of video frames for temporal attention (default: 25).</param>
    /// <param name="clipTokenLength">CLIP text token sequence length for cross-attention (default: 77).</param>
    /// <param name="lossFunction">Optional loss function (default: MSE).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="imageConditionChannels">Channels in the image/video condition. Defaults to <paramref name="inputChannels"/>.</param>
    /// <param name="concatenateImageCondition">Concatenate condition and latent before the input convolution.</param>
    /// <param name="numClassEmbeddings">Number of learned class/noise-level embeddings, or zero to disable.</param>
    public VideoUNetPredictor(
        int inputChannels = 4,
        int? outputChannels = null,
        int baseChannels = 320,
        int[]? channelMultipliers = null,
        int numResBlocks = 2,
        int[]? attentionResolutions = null,
        int numTemporalLayers = 1,
        int contextDim = 1024,
        int numHeads = 8,
        bool supportsImageConditioning = true,
        int inputHeight = 64,
        int inputWidth = 64,
        int numFrames = 25,
        int clipTokenLength = 77,
        ILossFunction<T>? lossFunction = null,
        int? seed = null,
        int? imageConditionChannels = null,
        bool concatenateImageCondition = false,
        int numClassEmbeddings = 0,
        VideoUNetArchitectureProfile architectureProfile = VideoUNetArchitectureProfile.Generic)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels ?? inputChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? new[] { 1, 2, 4, 4 };
        _numResBlocks = numResBlocks;
        _attentionResolutions = attentionResolutions ?? new[] { 1, 2, 3 };
        _numTemporalLayers = numTemporalLayers;
        _architectureProfile = architectureProfile;
        _contextDim = contextDim;
        _numHeads = numHeads;
        _timeEmbeddingDim = baseChannels * 4;
        _supportsImageConditioning = supportsImageConditioning;
        _imageConditionChannels = imageConditionChannels ?? inputChannels;
        _concatenateImageCondition = concatenateImageCondition;
        _numClassEmbeddings = numClassEmbeddings;
        if (_imageConditionChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageConditionChannels));
        if (_numClassEmbeddings < 0)
            throw new ArgumentOutOfRangeException(nameof(numClassEmbeddings));
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _numFrames = numFrames;
        _clipTokenLength = clipTokenLength;

        _encoderBlocks = new List<VideoBlock>();
        _middleBlocks = new List<VideoBlock>();
        _decoderBlocks = new List<VideoBlock>();

        // Establish a deterministic per-layer init-seed sequence for the layers built
        // below. NoisePredictorBase (unlike NeuralNetworkBase) does not call this, so
        // without it every layer's lazy weight init falls back to the process-shared,
        // order-dependent RandomHelper.ThreadSafeRandom — making the predictor's
        // initial weights depend on how much unrelated work ran first. That is the
        // root of the suite-position-dependent training-invariant flakiness (a model
        // whose Training_ShouldReducePredictionError passes in isolation fails once
        // sibling tests have advanced the shared RNG). With a seed the init becomes
        // reproducible; with seed == null the scope is inert and the existing
        // non-reproducible production default is preserved.
        LayerInitializationSeedScope.ResetForModelConstruction(seed);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes all layers of the Video U-Net.
    /// </summary>
    private void InitializeLayers()
    {
        // Input convolution: [inputChannels] -> [baseChannels]. LazyConv2D keeps
        // kernel tensors unallocated until first Forward() — the full video U-Net
        // is multi-GB at default sizes.
        int networkInputChannels = _inputChannels +
            (_supportsImageConditioning && _concatenateImageCondition ? _imageConditionChannels : 0);
        _inputConv = LazyConv2D(
            inputDepth: networkInputChannels,
            inputHeight: _inputHeight,
            inputWidth: _inputWidth,
            outputDepth: _baseChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: new IdentityActivation<T>());

        // Time embedding MLP — lazy so constructor-time memory stays flat.
        // The first projection MUST take the full sinusoidal embedding width that
        // NoisePredictorBase.GetTimestepEmbedding actually emits — [1, TimeEmbeddingDim]
        // — not TimeEmbeddingDim/4. The earlier /4 input made the layer's CONSTRUCTION
        // shape ([TimeEmbeddingDim/4]) disagree with its real forward input
        // ([TimeEmbeddingDim]); the lazy DenseLayer then silently re-resolved to the
        // larger forward shape, so a freshly-constructed clone (which has no forward to
        // re-resolve it) kept the stale narrow shape and SetParameters threw a length
        // mismatch on Clone. Per Ho et al. 2022 §3 / Dhariwal & Nichol 2021 the time
        // embedding is a 2-layer MLP (Linear → SiLU → Linear) over the sinusoidal
        // features; here the sinusoidal width already equals TimeEmbeddingDim
        // (= 4·baseChannels), so both projections are [TimeEmbeddingDim → TimeEmbeddingDim].
        _timeEmbedMlp1 = LazyDense(_timeEmbeddingDim, _timeEmbeddingDim, new SiLUActivation<T>());
        _timeEmbedMlp2 = LazyDense(_timeEmbeddingDim, _timeEmbeddingDim, new SiLUActivation<T>());

        // Image conditioning projection (for image-to-video)
        if (_supportsImageConditioning && !_concatenateImageCondition)
        {
            _imageCondProjection = LazyConv2D(
                inputDepth: _imageConditionChannels,
                inputHeight: _inputHeight,
                inputWidth: _inputWidth,
                outputDepth: _baseChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activation: new IdentityActivation<T>());
        }

        if (_numClassEmbeddings > 0)
        {
            _classEmbedding = new EmbeddingLayer<T>(_numClassEmbeddings, _timeEmbeddingDim);
        }

        if (_architectureProfile == VideoUNetArchitectureProfile.UpscaleAVideo)
        {
            BuildUpscaleAVideoTopology();
            return;
        }

        // Build encoder
        var inChannels = _baseChannels;
        for (int level = 0; level < _channelMultipliers.Length; level++)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block < _numResBlocks; block++)
            {
                _encoderBlocks.Add(new VideoBlock
                {
                    SpatialResBlock = CreateSpatialResBlock(inChannels, outChannels),
                    TemporalResBlock = CreateTemporalMixingBlock(),
                    SpatialAttention = useAttention ? CreateSpatialAttention(outChannels, level) : null,
                    TemporalAttention = useAttention ? CreateTemporalAttention(outChannels) : null,
                    CrossAttention = useAttention && _contextDim > 0 ? CreateCrossAttention(outChannels, level) : null,
                    TimeCondProjection = CreateTimeCondProjection(outChannels)
                });
                inChannels = outChannels;
            }

            // Add downsampling except for last level
            if (level < _channelMultipliers.Length - 1)
            {
                _encoderBlocks.Add(new VideoBlock
                {
                    Downsample = CreateDownsample(outChannels, level)
                });
            }
        }

        // Build middle — operates at the deepest (smallest) encoder resolution.
        int middleLevel = _channelMultipliers.Length - 1;
        _middleBlocks.Add(new VideoBlock
        {
            SpatialResBlock = CreateSpatialResBlock(inChannels, inChannels),
            TemporalResBlock = CreateTemporalMixingBlock(),
            SpatialAttention = CreateSpatialAttention(inChannels, middleLevel),
            TemporalAttention = CreateTemporalAttention(inChannels),
            CrossAttention = _contextDim > 0 ? CreateCrossAttention(inChannels, middleLevel) : null,
            TimeCondProjection = CreateTimeCondProjection(inChannels)
        });
        _middleBlocks.Add(new VideoBlock
        {
            SpatialResBlock = CreateSpatialResBlock(inChannels, inChannels),
            TemporalResBlock = CreateTemporalMixingBlock(),
            TimeCondProjection = CreateTimeCondProjection(inChannels)
        });

        // Build decoder (reverse of encoder).
        //
        // Per-level skip semantics: the encoder pushes ONE skip per level
        // boundary (saved before each downsample). The decoder consumes
        // exactly one skip per upsample boundary, applied to the FIRST
        // ResBlock at each level except the deepest. Construction must
        // reflect this contract — every other block receives only the prior
        // block's output without skip augmentation.
        //
        // Channel-count math:
        //   - Block 0 at level L (when L is not the deepest level):
        //       upsample output: prior level's outChannels = multipliers[L+1] of
        //         the level we just left (since decoder iterates max-1 → 0,
        //         the "prior level" is the deeper one we just finished).
        //       skip from encoder level L: multipliers[L].
        //       Total input to ResBlock = multipliers[L+1] + multipliers[L].
        //   - Block 0 at the deepest level (L == max-1, no upsample feeding
        //     it, no skip): receives middle-block output directly.
        //   - All non-zero blocks: receive prior block's output (outChannels
        //     of the same level), no skip concat.
        //
        // The previous construction used `multipliers[level + 1]` as
        // skipChannels (wrong — that's the deeper level, not the current
        // level's encoder skip). It also used `outChannels` as skipChannels
        // for non-zero blocks, doubling their input channel count. Both bugs
        // produce SpatialResBlock weights at the wrong shape and the runtime
        // forward pass would either throw or silently mis-compute features.
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block <= _numResBlocks; block++)
            {
                int actualInChannels;
                if (block == 0 && level < _channelMultipliers.Length - 1)
                {
                    // First block of a non-deepest level: receives upsample
                    // output (inChannels at this point = previous level's
                    // outChannels) concatenated with encoder skip from THIS
                    // level (multipliers[level] × baseChannels).
                    int skipChannels = _baseChannels * _channelMultipliers[level];
                    actualInChannels = inChannels + skipChannels;
                }
                else
                {
                    // Either (a) first block of the deepest decoder level
                    //   — no skip, no upsample, just middle-block output, or
                    // (b) any non-first block — no skip, just prior block's
                    //   output.
                    actualInChannels = inChannels;
                }

                _decoderBlocks.Add(new VideoBlock
                {
                    SpatialResBlock = CreateSpatialResBlock(actualInChannels, outChannels),
                    TemporalResBlock = CreateTemporalMixingBlock(),
                    SpatialAttention = useAttention ? CreateSpatialAttention(outChannels, level) : null,
                    TemporalAttention = useAttention ? CreateTemporalAttention(outChannels) : null,
                    CrossAttention = useAttention && _contextDim > 0 ? CreateCrossAttention(outChannels, level) : null,
                    TimeCondProjection = CreateTimeCondProjection(outChannels)
                });
                inChannels = outChannels;
            }

            // Add upsampling except for the shallowest level (level 0 has no
            // further level to ascend to).
            if (level > 0)
            {
                _decoderBlocks.Add(new VideoBlock
                {
                    Upsample = CreateUpsample(outChannels, level)
                });
            }
        }

        // Output convolution (lazy).
        _outputConv = LazyConv2D(
            inputDepth: _baseChannels,
            inputHeight: _inputHeight,
            inputWidth: _inputWidth,
            outputDepth: _outputChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: new IdentityActivation<T>());
    }

    /// <summary>
    /// Builds the released Upscale-A-Video U-Net rather than the historical
    /// fixed-frame approximation. The topology follows unet_video_config.json:
    /// [256,512,512,1024], two down ResNets, three up ResNets, cross attention
    /// at levels 1-3, self-attention only at the deepest cross-attention stage,
    /// and one convolutional temporal module after every down/up stage plus mid.
    /// Transformer3D blocks retain their distinct zero-initialized temporal-attention path.
    /// </summary>
    private void BuildUpscaleAVideoTopology()
    {
        if (_inputChannels != 4 || _outputChannels != 4 || _baseChannels != 256)
            throw new ArgumentException(
                "Upscale-A-Video requires four latent input/output channels and baseChannels 256.");
        if (!_supportsImageConditioning || !_concatenateImageCondition || _imageConditionChannels != 3)
            throw new ArgumentException(
                "Upscale-A-Video requires concatenated three-channel low-resolution conditioning.");
        if (_numClassEmbeddings != 1000)
            throw new ArgumentException(
                "Upscale-A-Video requires the released 1,000-entry noise-level embedding.");
        if (_inputHeight != 128 || _inputWidth != 128)
            throw new ArgumentException(
                "Upscale-A-Video requires the released 128x128 latent sample size.");
        if (_channelMultipliers.Length != 4 ||
            !_channelMultipliers.SequenceEqual(new[] { 1, 2, 2, 4 }))
            throw new ArgumentException(
                "Upscale-A-Video requires channelMultipliers [1,2,2,4].");
        if (_numResBlocks != 2)
            throw new ArgumentException("Upscale-A-Video requires two spatial ResNets per down block.");
        if (_contextDim != 1024)
            throw new ArgumentException("Upscale-A-Video requires cross-attention width 1024.");
        if (_numHeads != 8)
            throw new ArgumentException("Upscale-A-Video requires eight attention heads.");

        int inChannels = _baseChannels;
        int lastLevel = _channelMultipliers.Length - 1;

        // Down blocks. Each ResNet output is a U-Net skip; the first three
        // downsample outputs are skips too, exactly as diffusers DownBlock3D.
        for (int level = 0; level <= lastLevel; level++)
        {
            int outChannels = _baseChannels * _channelMultipliers[level];
            bool useCrossAttention = level > 0;
            bool useSelfAttention = level == lastLevel;
            for (int block = 0; block < _numResBlocks; block++)
            {
                _encoderBlocks.Add(new VideoBlock
                {
                    SpatialResBlock = CreateSpatialResBlock(inChannels, outChannels, level),
                    SpatialAttention = useCrossAttention
                        ? CreateVideoTransformer(outChannels, level, onlyCrossAttention: !useSelfAttention)
                        : null,
                    CaptureSkipAfter = true
                });
                inChannels = outChannels;
            }

            if (level < lastLevel)
            {
                _encoderBlocks.Add(new VideoBlock
                {
                    Downsample = CreateDownsample(outChannels, level),
                    CaptureSkipAfter = true
                });
            }

            _encoderBlocks.Add(new VideoBlock
            {
                TemporalResBlock = new TemporalModule3DLayer<T>(
                    outChannels, _timeEmbeddingDim,
                    ResolutionAtLevel(System.Math.Min(level + 1, lastLevel)))
            });
        }

        // Stable-Diffusion mid block: ResNet → self/cross attention → ResNet,
        // followed by the released temporal module.
        _middleBlocks.Add(new VideoBlock
        {
            SpatialResBlock = CreateSpatialResBlock(inChannels, inChannels, lastLevel),
            SpatialAttention = CreateVideoTransformer(
                inChannels, lastLevel, onlyCrossAttention: false)
        });
        _middleBlocks.Add(new VideoBlock
        {
            SpatialResBlock = CreateSpatialResBlock(inChannels, inChannels, lastLevel)
        });
        _middleBlocks.Add(new VideoBlock
        {
            TemporalResBlock = new TemporalModule3DLayer<T>(
                inChannels, _timeEmbeddingDim, ResolutionAtLevel(lastLevel))
        });

        // Up blocks consume one skip before EACH of their three ResNets.
        // The third skip in a stage comes from the next shallower channel width.
        for (int level = lastLevel; level >= 0; level--)
        {
            int outChannels = _baseChannels * _channelMultipliers[level];
            int shallowSkipChannels = level > 0
                ? _baseChannels * _channelMultipliers[level - 1]
                : _baseChannels;
            bool useCrossAttention = level > 0;
            bool useSelfAttention = level == lastLevel;

            for (int block = 0; block < _numResBlocks + 1; block++)
            {
                int skipChannels = block == _numResBlocks
                    ? shallowSkipChannels
                    : outChannels;
                _decoderBlocks.Add(new VideoBlock
                {
                    SpatialResBlock = CreateSpatialResBlock(
                        inChannels + skipChannels, outChannels, level),
                    SpatialAttention = useCrossAttention
                        ? CreateVideoTransformer(outChannels, level, onlyCrossAttention: !useSelfAttention)
                        : null,
                    ConsumesSkip = true
                });
                inChannels = outChannels;
            }

            if (level > 0)
            {
                _decoderBlocks.Add(new VideoBlock
                {
                    Upsample = CreateUpsample(outChannels, level - 1)
                });
            }

            _decoderBlocks.Add(new VideoBlock
            {
                TemporalResBlock = new TemporalModule3DLayer<T>(
                    outChannels, _timeEmbeddingDim,
                    ResolutionAtLevel(System.Math.Max(0, level - 1)))
            });
        }

        _outputNorm = new GroupNormalizationLayer<T>(
            ComputeGroups(_baseChannels, 32), _baseChannels, 1e-5);
        _outputConv = LazyConv2D(
            inputDepth: _baseChannels,
            inputHeight: _inputHeight,
            inputWidth: _inputWidth,
            outputDepth: _outputChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: new IdentityActivation<T>());
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        using var streaming = BeginWeightStreamingForward();
        _lastInput = noisySample;

        // Compute timestep embedding
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        // Forward pass
        return streaming.Complete(ForwardVideoUNet(noisySample, timeEmbed, conditioning, imageCondition: null));
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        using var streaming = BeginWeightStreamingForward();
        _lastInput = noisySample;

        var timeEmbed = ProjectTimeEmbedding(timeEmbedding);
        return streaming.Complete(ForwardVideoUNet(noisySample, timeEmbed, conditioning, imageCondition: null));
    }

    /// <summary>
    /// Predicts noise for image-to-video generation with image conditioning.
    /// </summary>
    /// <param name="noisySample">The noisy video latent.</param>
    /// <param name="timestep">The current timestep.</param>
    /// <param name="imageCondition">The conditioning image (first frame).</param>
    /// <param name="textConditioning">Optional text conditioning.</param>
    /// <returns>The predicted noise.</returns>
    public Tensor<T> PredictNoiseWithImageCondition(
        Tensor<T> noisySample,
        int timestep,
        Tensor<T> imageCondition,
        Tensor<T>? textConditioning = null)
    {
        if (!_supportsImageConditioning)
        {
            throw new InvalidOperationException("This predictor does not support image conditioning.");
        }

        using var streaming = BeginWeightStreamingForward();
        _lastInput = noisySample;

        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        return streaming.Complete(ForwardVideoUNet(noisySample, timeEmbed, textConditioning, imageCondition));
    }

    /// <summary>
    /// Predicts noise while conditioning every frame on its corresponding low-resolution
    /// video latent. This is the conditioning contract used by video super-resolution;
    /// unlike image-to-video conditioning, it must not reuse the first frame for the clip.
    /// </summary>
    /// <param name="noisySample">Noisy target latents in [B,C,F,H,W] layout.</param>
    /// <param name="timestep">Current diffusion timestep.</param>
    /// <param name="videoCondition">Per-frame conditioning latents in [B,C,F,H,W] layout.</param>
    /// <param name="textConditioning">Optional text encoder states.</param>
    public Tensor<T> PredictNoiseWithVideoCondition(
        Tensor<T> noisySample,
        int timestep,
        Tensor<T> videoCondition,
        Tensor<T>? textConditioning = null,
        int? noiseLevel = null)
    {
        if (!_supportsImageConditioning)
            throw new InvalidOperationException("This predictor does not support video conditioning.");
        if (noisySample.Rank != 5 || videoCondition.Rank != 5)
            throw new ArgumentException("Video conditioning requires [B,C,F,H,W] tensors.");
        if (noisySample.Shape[0] != videoCondition.Shape[0] ||
            noisySample.Shape[2] != videoCondition.Shape[2] ||
            noisySample.Shape[3] != videoCondition.Shape[3] ||
            noisySample.Shape[4] != videoCondition.Shape[4])
            throw new ArgumentException(
                "Video condition must match latent batch, frame, height, and width dimensions.",
                nameof(videoCondition));
        if (videoCondition.Shape[1] != _imageConditionChannels)
            throw new ArgumentException(
                $"Expected {_imageConditionChannels} condition channels, got {videoCondition.Shape[1]}.",
                nameof(videoCondition));

        using var streaming = BeginWeightStreamingForward();
        _lastInput = noisySample;
        var timeEmbed = AddClassEmbedding(
            ProjectTimeEmbedding(GetTimestepEmbedding(timestep)), noiseLevel);
        return streaming.Complete(
            ForwardVideoUNet(noisySample, timeEmbed, textConditioning, videoCondition));
    }

    private Tensor<T> AddClassEmbedding(Tensor<T> timeEmbedding, int? classLabel)
    {
        if (_classEmbedding is null)
            return timeEmbedding;
        if (!classLabel.HasValue)
            throw new ArgumentNullException(nameof(classLabel),
                "This predictor requires a degradation/noise-level label.");
        if ((uint)classLabel.Value >= (uint)_numClassEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(classLabel), classLabel.Value,
                $"Class/noise level must be in [0, {_numClassEmbeddings - 1}].");

        var label = new Tensor<T>([1],
            new Vector<T>(new[] { NumOps.FromDouble(classLabel.Value) }));
        return Engine.TensorAdd(timeEmbedding, _classEmbedding.Forward(label));
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
    /// Performs the forward pass through the Video U-Net.
    /// </summary>
    private Tensor<T> ForwardVideoUNet(
        Tensor<T> x,
        Tensor<T> timeEmbed,
        Tensor<T>? textConditioning,
        Tensor<T>? imageCondition)
    {
        if (_inputConv == null || _outputConv == null)
        {
            throw new InvalidOperationException("Layers not initialized.");
        }

        // Input shape: [batch, channels, frames, height, width]
        bool isVideo = x.Shape.Length == 5;
        int numFrames = isVideo ? x.Shape[2] : 1;

        // Upscale-A-Video / SD-x4 concatenate the three-channel low-resolution
        // condition with the four-channel noisy latent before the seven-channel
        // input convolution. Other video models retain additive projection mode.
        if (imageCondition != null && _concatenateImageCondition)
        {
            if (imageCondition.Rank != x.Rank)
                throw new ArgumentException("Concatenated conditioning rank must match the sample rank.",
                    nameof(imageCondition));
            x = ConcatenateChannels(x, imageCondition, isVideo);
        }

        // Process each frame through input conv (or use 3D conv in production)
        x = isVideo
            ? ProcessVideoFrames(x, frame => _inputConv.Forward(frame))
            : _inputConv.Forward(x);

        // Add image condition (for image-to-video)
        if (imageCondition != null && !_concatenateImageCondition && _imageCondProjection != null)
        {
            if (imageCondition.Rank == 5)
            {
                var videoCond = ProcessVideoFrames(
                    imageCondition, frame => _imageCondProjection.Forward(frame));
                x = Engine.TensorAdd(x, videoCond);
            }
            else
            {
                var imageCond = _imageCondProjection.Forward(imageCondition);
                // Image-to-video retains first-frame conditioning semantics.
                x = AddImageCondition(x, imageCond, numFrames);
            }
        }

        // Store skip connections — one per spatial level, captured at the
        // level boundary right before each Downsample. Per DDPM (Ho et al.
        // 2020) §C "Architecture details" and Ronneberger et al. 2015 §2,
        // U-Net skips are level-tagged: each spatial level transfers a single
        // tensor across the network at the level boundary, and the decoder
        // consumes it at the matching level right after Upsample. Saving a
        // skip after every ResBlock (multiple per level) and popping LIFO
        // makes the decoder consume the wrong-level skip first — at one
        // spatial size — and then concat fails because non-channel dims
        // don't match. Saving exactly one skip per Downsample / consuming
        // exactly one skip per Upsample keeps spatial dims synchronised by
        // construction across the symmetric encoder/decoder pyramid.
        var skips = new List<Tensor<T>>();

        // Encoder. The released Upscale-A-Video/diffusers U-Net stores the
        // input-conv output, every ResNet output, and each non-final downsample
        // output. Generic callers retain the legacy level-boundary behavior.
        if (_architectureProfile == VideoUNetArchitectureProfile.UpscaleAVideo)
        {
            skips.Add(x);
            foreach (var block in _encoderBlocks)
            {
                x = block.Downsample is not null
                    ? ApplyDownsample(block.Downsample, x, isVideo)
                    : ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
                if (block.CaptureSkipAfter) skips.Add(x);
            }
        }
        else
        {
            for (int i = 0; i < _encoderBlocks.Count; i++)
            {
                var block = _encoderBlocks[i];
                if (block.Downsample != null)
                {
                    skips.Add(x);
                    x = ApplyDownsample(block.Downsample, x, isVideo);
                }
                else
                {
                    x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
                }
            }
        }

        // Middle
        foreach (var block in _middleBlocks)
        {
            x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
        }

        // Decoder: Upsample to a level then concat with that level's skip
        // exactly once before running the level's ResBlock(s). Skips pop in
        // reverse order — last-saved (deepest level) is the first consumed
        // because the decoder ascends from the bottleneck toward the input.
        var skipIdx = skips.Count - 1;
        if (_architectureProfile == VideoUNetArchitectureProfile.UpscaleAVideo)
        {
            foreach (var block in _decoderBlocks)
            {
                if (block.ConsumesSkip)
                {
                    if (skipIdx < 0)
                        throw new InvalidOperationException(
                            "Upscale-A-Video decoder requested more skips than the encoder produced.");
                    x = ConcatenateChannels(x, skips[skipIdx--], isVideo);
                }

                x = block.Upsample is not null
                    ? ApplyUpsample(block.Upsample, x, isVideo)
                    : ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
            }
            if (skipIdx != -1)
                throw new InvalidOperationException(
                    $"Upscale-A-Video decoder left {skipIdx + 1} encoder skip(s) unused.");
        }
        else
        {
            bool consumeSkipNext = false;
            foreach (var block in _decoderBlocks)
            {
                if (block.Upsample != null)
                {
                    x = ApplyUpsample(block.Upsample, x, isVideo);
                    consumeSkipNext = true;
                }
                else
                {
                    if (consumeSkipNext && skipIdx >= 0)
                    {
                        x = ConcatenateChannels(x, skips[skipIdx--], isVideo);
                        consumeSkipNext = false;
                    }
                    x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
                }
            }
        }

        if (_outputNorm is not null)
        {
            x = isVideo
                ? ProcessVideoFrames(x, frame => Engine.TensorSiLU(_outputNorm.Forward(frame)))
                : Engine.TensorSiLU(_outputNorm.Forward(x));
        }

        // Output convolution
        x = isVideo
            ? ProcessVideoFrames(x, frame => _outputConv.Forward(frame))
            : _outputConv.Forward(x);

        _lazyShapeResolved = true;
        _lazyShapeResolvedWithVideo |= isVideo;
        _lazyShapeResolvedWithTextConditioning |= textConditioning is not null;
        _lazyShapeResolvedWithImageConditioning |= imageCondition is not null;
        return x;
    }

    /// <summary>
    /// Applies a single video block: spatial ResBlock → FiLM timestep conditioning →
    /// temporal ResBlock → spatial attention → temporal attention → cross-attention.
    /// Per Ho et al. 2022 "Video Diffusion Models" §3.1, timestep conditioning is
    /// injected via Adaptive Group Normalization (AdaGN): the time embedding is projected
    /// to per-channel scale and shift parameters, then the feature map is modulated as
    /// <c>x = x * (1 + scale) + shift</c>. Temporal processing applies a learned
    /// temporal mixing layer across the frame axis with a residual connection.
    /// </summary>
    private Tensor<T> ApplyVideoBlock(
        VideoBlock block,
        Tensor<T> x,
        Tensor<T> timeEmbed,
        Tensor<T>? conditioning,
        bool isVideo)
    {
        // Upscale-A-Video freezes the Stable Diffusion spatial backbone and optimizes only the
        // inserted temporal modules. Retaining every internal spatial activation until backward
        // made the paper-scale 733M model require over 45 GiB for one step. Checkpoint each block:
        // retain its boundary, recompute its interior during backward, and explicitly request only
        // the temporal parameter gradients. The input VJP still crosses every frozen spatial op,
        // so this is mathematically the same temporal-only fine-tuning objective, not a smaller model.
        if (_architectureProfile == VideoUNetArchitectureProfile.UpscaleAVideo
            && GradientTape<T>.Current is not null)
        {
            return GradientCheckpointing<T>.Checkpoint(
                [input => ApplyVideoBlockCore(block, input, timeEmbed, conditioning, isVideo)],
                x,
                parameterSourceFactory: () => GetBlockTemporalTrainingParameters(block),
                segmentSize: 1);
        }

        return ApplyVideoBlockCore(block, x, timeEmbed, conditioning, isVideo);
    }

    private Tensor<T> ApplyVideoBlockCore(
        VideoBlock block,
        Tensor<T> x,
        Tensor<T> timeEmbed,
        Tensor<T>? conditioning,
        bool isVideo)
    {
        // Spatial ResBlock
        if (block.SpatialResBlock is DiffusionResBlock<T> diffusionResBlock)
        {
            x = isVideo
                ? ApplySpatialResBlockToVideo(diffusionResBlock, x, timeEmbed)
                : diffusionResBlock.Forward(x, timeEmbed);
        }
        else if (block.SpatialResBlock != null)
        {
            x = isVideo
                ? ProcessVideoFrames(x, frame => block.SpatialResBlock.Forward(frame))
                : block.SpatialResBlock.Forward(x);
        }

        // FiLM timestep conditioning (Dhariwal & Nichol 2021 / Ho et al. 2022):
        // project timeEmbed → [scale, shift], then x = x * (1 + scale) + shift.
        // This makes the model's feature maps timestep-dependent — without it the
        // noise predictor output is invariant to the diffusion step, which breaks
        // the denoising objective fundamentally.
        if (block.TimeCondProjection != null)
        {
            x = ApplyFiLMConditioning(block.TimeCondProjection, x, timeEmbed, isVideo);
        }

        // Temporal ResBlock (only for video) — learned temporal mixing with residual
        if (block.TemporalResBlock is TemporalModule3DLayer<T> temporalModule && isVideo)
        {
            x = temporalModule.Forward(x, timeEmbed);
        }
        else if (block.TemporalResBlock != null && isVideo)
        {
            x = ApplyTemporalProcessing(block.TemporalResBlock, x);
        }

        // Released Upscale-A-Video Transformer3D jointly owns spatial/text and
        // zero-initialized temporal attention plus GEGLU residual processing.
        bool handledTransformer3D = false;
        if (block.SpatialAttention is VideoTransformer3DLayer<T> transformer3D)
        {
            if (conditioning is null)
                throw new InvalidOperationException(
                    "Upscale-A-Video Transformer3D requires Stable Diffusion x4 Upscaler " +
                    "CLIP encoder states; text conditioning cannot be omitted.");
            Tensor<T> videoInput = isVideo
                ? x
                : Engine.Reshape(x, [x.Shape[0], x.Shape[1], 1, x.Shape[2], x.Shape[3]]);
            var transformed = transformer3D.Forward(videoInput, conditioning);
            x = isVideo
                ? transformed
                : Engine.Reshape(
                    transformed,
                    [transformed.Shape[0], transformed.Shape[1], transformed.Shape[3], transformed.Shape[4]]);
            handledTransformer3D = true;
        }

        // Spatial attention. MHA expects [batch, seq, embed_dim]; spatial features
        // arrive as NCHW [B, C, H, W]. Reshape to [B, H*W, C] before the layer and
        // back after — without this MHA reads H or W as the embedding dim and
        // throws a weight-mismatch ArgumentException.
        if (block.SpatialAttention != null && !handledTransformer3D)
        {
            x = isVideo
                ? ProcessVideoFrames(x, frame => SpatialAttentionForward(block.SpatialAttention, frame))
                : SpatialAttentionForward(block.SpatialAttention, x);
        }

        // Temporal attention (only for video)
        if (block.TemporalAttention != null && isVideo)
        {
            x = ApplyTemporalAttention(block.TemporalAttention, x);
        }

        // Cross-attention with text conditioning: query=spatial features,
        // key=value=conditioning (text embedding). Cast to LayerBase to
        // access the params Forward(query, kv) overload — ILayer only
        // exposes Forward(single input).
        if (block.CrossAttention != null && conditioning != null)
        {
            // Cross-attention requires the LayerBase multi-input Forward(query, kv)
            // overload to receive the conditioning as the key/value tensor. The
            // ILayer<T> interface only exposes single-input Forward. If a caller
            // substitutes a non-LayerBase implementation, fail loudly rather than
            // silently degrading to self-attention — losing text conditioning in
            // production would be a catastrophic correctness regression that's
            // hard to debug from outputs alone.
            if (block.CrossAttention is not LayerBase<T> crossAttnBase)
            {
                throw new InvalidOperationException(
                    $"CrossAttention layer must derive from LayerBase<{typeof(T).Name}> " +
                    $"to support multi-input Forward(query, keyValue). " +
                    $"Got {block.CrossAttention.GetType().FullName}. " +
                    $"Substituting a single-input layer would silently drop the text " +
                    $"conditioning tensor and produce wrong outputs.");
            }

            x = isVideo
                ? ProcessVideoFrames(x, frame => SpatialCrossAttentionForward(crossAttnBase, frame, conditioning))
                : SpatialCrossAttentionForward(crossAttnBase, x, conditioning);
        }

        return x;
    }

    private IReadOnlyList<Tensor<T>> GetBlockTemporalTrainingParameters(VideoBlock block)
    {
        var parameters = new List<Tensor<T>>();
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);

        Add(block.TemporalResBlock);
        Add(block.TemporalAttention);
        if (block.SpatialAttention is VideoTransformer3DLayer<T> transformer)
        {
            foreach (var layer in transformer.TemporalTrainingLayers) Add(layer);
        }

        return parameters;

        void Add(ILayer<T>? layer)
        {
            if (layer is ITrainableLayer<T> trainable)
            {
                // The no-grad checkpoint forward runs before this factory, so lazy child weights
                // are materialized. TemporalTrainingLayers contains the explicit leaf/composite
                // adapters from the paper objective; read their tensor list directly without ever
                // flattening the foundation-scale predictor parameter vector.
                foreach (var parameter in trainable.GetTrainableParameters())
                    if (parameter.Length > 0 && seen.Add(parameter)) parameters.Add(parameter);
            }
        }
    }

    // Reshape NCHW [B, C, H, W] → [B, H*W, C] for self-attention, then back.
    // MultiHeadAttentionLayer treats last-dim as embedding; passing NCHW directly
    // makes it read W (8) as embed and the [C, C] weights mismatch.
    private Tensor<T> SpatialAttentionForward(ILayer<T> attn, Tensor<T> nchw)
    {
        int b = nchw.Shape[0];
        int c = nchw.Shape[1];
        int h = nchw.Shape[2];
        int w = nchw.Shape[3];
        // [B, C, H, W] → [B, H, W, C] → [B, H*W, C]
        var bhwc = Engine.TensorPermute(nchw, new[] { 0, 2, 3, 1 }).Contiguous();
        var seq = Engine.Reshape(bhwc, new[] { b, h * w, c });
        var attended = attn.Forward(seq);
        // [B, H*W, C] → [B, H, W, C] → [B, C, H, W]
        var attendedHwc = Engine.Reshape(attended, new[] { b, h, w, c });
        return Engine.TensorPermute(attendedHwc, new[] { 0, 3, 1, 2 }).Contiguous();
    }

    // Same NCHW↔BSC reshape for cross-attention (query is spatial, KV is conditioning).
    private Tensor<T> SpatialCrossAttentionForward(LayerBase<T> attn, Tensor<T> nchw, Tensor<T> kv)
    {
        int b = nchw.Shape[0];
        int c = nchw.Shape[1];
        int h = nchw.Shape[2];
        int w = nchw.Shape[3];
        var bhwc = Engine.TensorPermute(nchw, new[] { 0, 2, 3, 1 }).Contiguous();
        var seq = Engine.Reshape(bhwc, new[] { b, h * w, c });
        var attended = attn.Forward(seq, kv);
        var attendedHwc = Engine.Reshape(attended, new[] { b, h, w, c });
        return Engine.TensorPermute(attendedHwc, new[] { 0, 3, 1, 2 }).Contiguous();
    }

    /// <summary>
    /// Applies Feature-wise Linear Modulation (FiLM) from the timestep embedding to
    /// the feature map <paramref name="x"/>. The projection layer maps
    /// <c>[timeEmbedDim] → [channels * 2]</c>; the first half is scale, the second
    /// half is shift. The modulation is <c>x = x * (1 + scale) + shift</c>,
    /// broadcast across spatial (and temporal, for video) dimensions.
    /// </summary>
    private Tensor<T> ApplyFiLMConditioning(
        DenseLayer<T> projection, Tensor<T> x, Tensor<T> timeEmbed, bool isVideo)
    {
        // Ground truth for dimensions is the feature map `x`, not the projection
        // output — we want to fail loudly if the projection was constructed with
        // the wrong channel count rather than silently slicing the projection
        // vector and producing an out-of-shape broadcast.
        int batchSize = x.Shape[0];
        int channels = x.Shape[1];

        // timeEmbed shape depends on the caller:
        //   (a) 1D [timeEmbedDim] when GetTimestepEmbedding returns an unbatched
        //       sinusoidal embedding (the shared-across-batch case — standard
        //       path when batch originates from PredictNoise on a single int).
        //   (b) 2D [B, timeEmbedDim] when the caller pre-batched timeEmbed
        //       (e.g., PredictNoiseWithEmbedding on a [B, D] tensor).
        // After projection:
        //   (a) → 1D [channels*2]     — broadcast scale/shift across all batches.
        //   (b) → 2D [B, channels*2]  — per-batch scale/shift.
        var condVec = projection.Forward(timeEmbed);
        bool condIsBatched = condVec.Shape.Length >= 2;
        int expectedCondWidth = channels * 2;
        if (condVec.Shape[^1] != expectedCondWidth)
        {
            throw new InvalidOperationException(
                $"FiLM conditioning projection width mismatch: expected {expectedCondWidth} " +
                $"(2 * channels for [scale, shift]), got {condVec.Shape[^1]}. " +
                "This indicates the VideoBlock's TimeCondProjection was sized for a different channel count.");
        }
        if (condIsBatched && condVec.Shape[0] != batchSize)
        {
            throw new InvalidOperationException(
                $"FiLM conditioning batch-size mismatch: feature map has batch {batchSize} " +
                $"but projection output has batch {condVec.Shape[0]}. " +
                "Pass a 1D timeEmbed to broadcast across all batches, or a 2D [B, timeEmbedDim] " +
                "timeEmbed where B matches the feature map's batch size.");
        }

        // Split projection into scale and shift. When condVec is 1D, we
        // broadcast the single [channels*2] vector across all batches. When
        // 2D, we split per-batch.
        var scaleData = new T[batchSize * channels];
        var shiftData = new T[batchSize * channels];
        var condSpan = condVec.AsSpan();
        for (int b = 0; b < batchSize; b++)
        {
            // When condVec is 1D, all batches read from the same [channels*2]
            // block at offset 0. When 2D, batch b reads from offset b*(channels*2).
            int srcBase = condIsBatched ? b * channels * 2 : 0;
            int dstBase = b * channels;
            for (int c = 0; c < channels; c++)
            {
                scaleData[dstBase + c] = condSpan[srcBase + c];
                shiftData[dstBase + c] = condSpan[srcBase + channels + c];
            }
        }

        // Reshape scale/shift to broadcast over spatial (+ temporal) dims.
        // Image: x is [B, C, H, W] → scale/shift [B, C, 1, 1]
        // Video: x is [B, C, F, H, W] → scale/shift [B, C, 1, 1, 1]
        int[] broadcastShape = isVideo
            ? new[] { batchSize, channels, 1, 1, 1 }
            : new[] { batchSize, channels, 1, 1 };

        var scaleTensor = new Tensor<T>(scaleData, broadcastShape);
        var shiftTensor = new Tensor<T>(shiftData, broadcastShape);

        // x = x * (1 + scale) + shift
        var onePlusScale = Engine.TensorAdd(
            scaleTensor,
            Tensor<T>.CreateDefault(broadcastShape, NumOps.One));
        var modulated = Engine.TensorMultiply(x, onePlusScale);
        return Engine.TensorAdd(modulated, shiftTensor);
    }

    /// <summary>
    /// Processes each frame of a video through a layer.
    /// </summary>
    private Tensor<T> ProcessVideoFrames(Tensor<T> video, Func<Tensor<T>, Tensor<T>> processFrame)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];
        var bfchw = Engine.TensorPermute(video, [0, 2, 1, 3, 4]).Contiguous();
        var frameBatch = Engine.Reshape(
            bfchw, [batch * frames, video.Shape[1], height, width]);
        var processed = processFrame(frameBatch);
        var restored = Engine.Reshape(
            processed,
            [batch, frames, processed.Shape[1], processed.Shape[2], processed.Shape[3]]);
        return Engine.TensorPermute(restored, [0, 2, 1, 3, 4]).Contiguous();
    }

    private Tensor<T> ApplySpatialResBlockToVideo(
        DiffusionResBlock<T> block, Tensor<T> video, Tensor<T> timeEmbedding)
    {
        int batch = video.Shape[0];
        int frames = video.Shape[2];
        int height = video.Shape[3];
        int width = video.Shape[4];
        var bfchw = Engine.TensorPermute(video, [0, 2, 1, 3, 4]).Contiguous();
        var frameBatch = Engine.Reshape(
            bfchw, [batch * frames, video.Shape[1], height, width]);

        Tensor<T> timeBatch = timeEmbedding.Rank switch
        {
            1 => Engine.Reshape(timeEmbedding, [1, timeEmbedding.Shape[0]]),
            2 => timeEmbedding,
            _ => throw new ArgumentException("Time embedding must be [D] or [B,D].", nameof(timeEmbedding))
        };
        if (timeBatch.Shape[0] == 1 && batch > 1)
            timeBatch = Engine.TensorBroadcastTo(timeBatch, [batch, timeBatch.Shape[1]]);
        if (timeBatch.Shape[0] != batch)
            throw new ArgumentException(
                $"Time embedding batch {timeBatch.Shape[0]} does not match video batch {batch}.",
                nameof(timeEmbedding));
        var expanded = Engine.Reshape(timeBatch, [batch, 1, timeBatch.Shape[1]]);
        expanded = Engine.TensorBroadcastTo(expanded, [batch, frames, timeBatch.Shape[1]]);
        expanded = Engine.Reshape(expanded, [batch * frames, timeBatch.Shape[1]]);

        var processed = block.Forward(frameBatch, expanded);
        var restored = Engine.Reshape(
            processed,
            [batch, frames, processed.Shape[1], processed.Shape[2], processed.Shape[3]]);
        return Engine.TensorPermute(restored, [0, 2, 1, 3, 4]).Contiguous();
    }

    /// <summary>
    /// Applies temporal processing with a residual connection, per Ho et al. 2022 §3.1.
    /// The temporal layer is a learned mixing operation along the frame axis: for each
    /// (batch, channel, height, width) position, the layer receives a vector of length
    /// <c>numFrames</c> and outputs a mixed vector of the same length. A residual
    /// connection preserves the original signal so the layer only needs to learn the
    /// temporal delta.
    /// </summary>
    /// <remarks>
    /// The paper uses 3D convolution with kernel (3,1,1) for temporal processing. Since
    /// this codebase does not yet have a 1D/3D temporal conv primitive, we approximate
    /// with a DenseLayer(<c>numFrames</c>, <c>numFrames</c>) applied per spatial-channel
    /// position. This captures global temporal mixing (each output frame is a learned
    /// linear combination of ALL input frames, vs. the paper's local kernel-3 receptive
    /// field). Both are viable — the dense version is more expressive but has O(F²)
    /// parameters vs. O(F) for the kernel-3 conv.
    /// </remarks>
    private Tensor<T> ApplyTemporalProcessing(ILayer<T> temporalLayer, Tensor<T> video)
    {
        // Video shape: [B, C, F, H, W]. Use the public Shape API and materialize
        // an independent int[] so we don't couple to Tensor<T>'s internal backing
        // field (which could be refactored) or share mutable shape storage with
        // the source tensor.
        var shape = video._shape;
        int batch = shape[0];
        int channels = shape[1];
        int frames = shape[2];
        int height = shape[3];
        int width = shape[4];

        // A plain reshape from [B,C,F,H,W] to [B*C*H*W, F] does NOT produce
        // rows that hold one spatial-channel position's frame vector — F is
        // not the innermost dimension in the source, so neighboring elements
        // along the flattened last axis span multiple H/W/F indices. Rows of
        // the naive reshape would therefore mix values across H and W, and
        // the DenseLayer would learn a meaningless cross-spatial mixing
        // instead of temporal mixing.
        //
        // Correct layout: permute [B,C,F,H,W] → [B,C,H,W,F] so F is the
        // innermost axis. Then reshape to [B*C*H*W, F] yields rows that ARE
        // the per-(b,c,h,w) frame vectors. Apply the temporal mixing and
        // reverse the permute so the caller sees [B,C,F,H,W] again.
        var permuted = Engine.TensorPermute(video, new[] { 0, 1, 3, 4, 2 });
        int spatialChannelPositions = batch * channels * height * width;
        var flat = Engine.Reshape(permuted, new[] { spatialChannelPositions, frames });

        // Apply temporal mixing layer: [B*C*H*W, F] → [B*C*H*W, F]
        var mixed = temporalLayer.Forward(flat);

        // Reshape back to [B, C, H, W, F] then un-permute to [B, C, F, H, W].
        // The final .Contiguous() materializes the permuted view — downstream
        // ops (ExtractFrame, AsSpan callers) require contiguous backing buffers,
        // and permutation views don't materialize automatically.
        var mixedPermuted = Engine.Reshape(mixed, new[] { batch, channels, height, width, frames });
        var mixedVideo = Engine.TensorPermute(mixedPermuted, new[] { 0, 1, 4, 2, 3 }).Contiguous();

        // Residual connection: output = input + temporalDelta
        // Per the paper, the residual ensures the temporal block only needs to
        // learn the temporal refinement, not reconstruct the full signal from
        // scratch. .Contiguous() on the result — TensorAdd can return a view
        // in some engine paths.
        return Engine.TensorAdd(video, mixedVideo).Contiguous();
    }

    /// <summary>
    /// Applies temporal attention across frames using GPU/CPU accelerated tensor operations.
    /// </summary>
    /// <remarks>
    /// For each spatial position (h, w), attention is computed across all frames.
    /// This allows the model to capture long-range temporal dependencies.
    /// Uses TensorPermute for efficient layout transformations on GPU/CPU.
    /// </remarks>
    private Tensor<T> ApplyTemporalAttention(ILayer<T> temporalAttention, Tensor<T> video)
    {
        // Video shape: [batch, channels, frames, height, width] (NCFHW)
        var shape = video._shape;
        int batch = shape[0];
        int channels = shape[1];
        int frames = shape[2];
        int height = shape[3];
        int width = shape[4];
        int spatialSize = height * width;

        // Step 1: Permute from NCFHW to NHWFC using GPU-accelerated permute
        // [batch, channels, frames, height, width] -> [batch, height, width, frames, channels]
        var permuted = Engine.TensorPermute(video, new[] { 0, 3, 4, 2, 1 });

        // Step 2: Reshape to [batch * height * width, frames, channels] for attention.
        // Must go through Engine so the gradient tape records the op — direct
        // Tensor<T>.Reshape bypasses the tape and breaks gradient flow through
        // the temporal attention path.
        var reshaped = Engine.Reshape(permuted, new[] { batch * spatialSize, frames, channels });

        // Step 3: Apply temporal attention layer
        Tensor<T> attended;
        if (temporalAttention is LayerBase<T> layerBase)
        {
            attended = layerBase.Forward(reshaped);
        }
        else
        {
            attended = temporalAttention.Forward(reshaped);
        }

        // Step 4: Reshape back to [batch, height, width, frames, channels]
        var reshapedBack = Engine.Reshape(attended, new[] { batch, height, width, frames, channels });

        // Step 5: Permute back from NHWFC to NCFHW
        // [batch, height, width, frames, channels] -> [batch, channels, frames, height, width]
        // .Contiguous() materializes the permuted view — downstream ops
        // (ExtractFrame, AsSpan callers) require a contiguous backing buffer.
        var result = Engine.TensorPermute(reshapedBack, new[] { 0, 4, 3, 1, 2 }).Contiguous();

        return result;
    }

    /// <summary>
    /// Adds image condition to video features.
    /// </summary>
    private Tensor<T> AddImageCondition(Tensor<T> videoFeatures, Tensor<T> imageCond, int numFrames)
    {
        var result = new Tensor<T>(videoFeatures._shape);
        var resultSpan = result.AsWritableSpan();
        var videoSpan = videoFeatures.AsSpan();
        var imageSpan = imageCond.AsSpan();

        int batch = videoFeatures.Shape[0];
        int channels = videoFeatures.Shape[1];
        int spatialSize = videoFeatures.Shape[3] * videoFeatures.Shape[4];
        int frameSize = channels * spatialSize;

        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int s = 0; s < spatialSize; s++)
                    {
                        int videoIdx = b * numFrames * frameSize + f * frameSize + c * spatialSize + s;
                        int imageIdx = b * frameSize + c * spatialSize + s;

                        // Add image condition to first frame, scaled for others
                        var scale = f == 0 ? NumOps.One : NumOps.FromDouble(0.1);
                        resultSpan[videoIdx] = NumOps.Add(videoSpan[videoIdx],
                            NumOps.Multiply(imageSpan[imageIdx], scale));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies downsampling to video.
    /// </summary>
    private Tensor<T> ApplyDownsample(ILayer<T> downsample, Tensor<T> x, bool isVideo)
    {
        if (isVideo)
        {
            return ProcessVideoFrames(x, frame => downsample.Forward(frame));
        }
        return downsample.Forward(x);
    }

    /// <summary>
    /// Applies upsampling to video.
    /// </summary>
    private Tensor<T> ApplyUpsample(ILayer<T> upsample, Tensor<T> x, bool isVideo)
    {
        if (isVideo)
        {
            return ProcessVideoFrames(x, frame => upsample.Forward(frame));
        }
        return upsample.Forward(x);
    }

    /// <summary>
    /// Concatenates channels for skip connections.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b, bool isVideo)
    {
        // Concatenate along axis 1 (channel dimension) for both NCFHW (5D) and NCHW (4D)
        // The engine handles proper interleaving of data along the specified axis
        return Engine.TensorConcatenate(new[] { a, b }, axis: 1);
    }

    /// <summary>
    /// Extracts a single frame from video.
    /// </summary>
    private Tensor<T> ExtractFrame(Tensor<T> video, int frameIndex)
    {
        int batch = video.Shape[0];
        int channels = video.Shape[1];
        int height = video.Shape[3];
        int width = video.Shape[4];

        var frame = new Tensor<T>([batch, channels, height, width]);
        var frameSpan = frame.AsWritableSpan();
        var videoSpan = video.AsSpan();

        int spatialSize = height * width;
        int numFrames = video.Shape[2];

        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int videoIdx = b * channels * numFrames * spatialSize +
                                       c * numFrames * spatialSize +
                                       frameIndex * spatialSize +
                                       h * width + w;
                        int frameIdx = b * channels * spatialSize +
                                       c * spatialSize +
                                       h * width + w;
                        frameSpan[frameIdx] = videoSpan[videoIdx];
                    }
                }
            }
        }

        return frame;
    }

    /// <summary>
    /// Stacks frames into video tensor.
    /// </summary>
    private Tensor<T> StackFrames(List<Tensor<T>> frames)
    {
        int batch = frames[0].Shape[0];
        int channels = frames[0].Shape[1];
        int numFrames = frames.Count;
        int height = frames[0].Shape[2];
        int width = frames[0].Shape[3];

        var video = new Tensor<T>([batch, channels, numFrames, height, width]);
        var videoSpan = video.AsWritableSpan();
        int spatialSize = height * width;

        for (int f = 0; f < numFrames; f++)
        {
            var frameSpan = frames[f].AsSpan();

            for (int b = 0; b < batch; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            int frameIdx = b * channels * spatialSize +
                                           c * spatialSize +
                                           h * width + w;
                            int videoIdx = b * channels * numFrames * spatialSize +
                                           c * numFrames * spatialSize +
                                           f * spatialSize +
                                           h * width + w;
                            videoSpan[videoIdx] = frameSpan[frameIdx];
                        }
                    }
                }
            }
        }

        return video;
    }

    #region Layer Factory Methods

    private ILayer<T> CreateSpatialResBlock(int inChannels, int outChannels, int level = 0)
    {
        // Shared paper-faithful diffusion ResNet: GN → SiLU → 3×3 conv,
        // additive Linear(SiLU(timestep)), GN → SiLU → 3×3 conv + shortcut.
        // This replaces the historical 1×1 projection that had neither a
        // residual path nor the released receptive field.
        return new DiffusionResBlock<T>(
            inChannels,
            outChannels,
            ResolutionAtLevel(level),
            _timeEmbeddingDim,
            numGroups: 32,
            epsilon: 1e-5);
    }

    /// <summary>
    /// Creates a temporal mixing block that learns a frame-axis transform.
    /// Per Ho et al. 2022 §3.1 the paper uses a 1D convolution along T; since
    /// this codebase has no 1D temporal conv primitive we approximate with a
    /// DenseLayer mapping <c>[numFrames] → [numFrames]</c>, applied per
    /// (batch, channel, height, width) position (analogous to a depthwise
    /// temporal 1×1 conv). The reshape+layer+reshape pipeline lives in
    /// <see cref="ApplyTemporalProcessing"/>; a residual add connects input
    /// and output so this layer only needs to learn the temporal refinement.
    /// </summary>
    /// <remarks>
    /// Takes no parameters: the block's shape depends solely on
    /// <see cref="_numFrames"/>, not on the channel count. The earlier
    /// signature <c>CreateTemporalResBlock(int channels)</c> mislead callers
    /// into sizing the block by channel count — now removed per review.
    /// </remarks>
    private ILayer<T> CreateTemporalMixingBlock()
    {
        return LazyDense(_numFrames, _numFrames, new SiLUActivation<T>());
    }

    /// <summary>
    /// Creates a FiLM conditioning projection for a VideoBlock: timeEmbedDim → channels * 2.
    /// The first half of the output is the scale, the second half is the shift.
    /// </summary>
    private DenseLayer<T> CreateTimeCondProjection(int channels)
    {
        return LazyDense(_timeEmbeddingDim, channels * 2, activation: null);
    }

    /// <summary>
    /// Returns the spatial resolution (height = width) at encoder/decoder
    /// level <paramref name="level"/>. Level 0 is the top of the UNet (input
    /// resolution); each subsequent downsample halves spatial size, so level
    /// N has resolution <c>_inputHeight &gt;&gt; N</c>. Clamped at 1 so
    /// deeper-than-expected level counts don't underflow.
    /// </summary>
    private int ResolutionAtLevel(int level)
        => Math.Max(1, _inputHeight >> level);

    private static int ComputeGroups(int channels, int targetGroups)
    {
        for (int groups = System.Math.Min(channels, targetGroups); groups >= 1; groups--)
            if (channels % groups == 0) return groups;
        return 1;
    }

    private ILayer<T> CreateSpatialAttention(int channels, int level)
    {
        int res = ResolutionAtLevel(level);
        return LazyMHA(res * res, channels, _numHeads, new IdentityActivation<T>());
    }

    private ILayer<T> CreateTemporalAttention(int channels)
    {
        return LazyMHA(_numFrames, channels, _numHeads, new IdentityActivation<T>());
    }

    private ILayer<T> CreateCrossAttention(int channels, int level)
    {
        int resolution = ResolutionAtLevel(level);
        return new CrossAttentionLayer<T>(
            queryDim: channels,
            contextDim: _contextDim,
            headCount: _numHeads,
            sequenceLength: resolution * resolution);
    }

    private ILayer<T> CreateVideoTransformer(
        int channels, int level, bool onlyCrossAttention)
    {
        return new VideoTransformer3DLayer<T>(
            channels,
            _contextDim,
            _numHeads,
            ResolutionAtLevel(level),
            onlyCrossAttention);
    }

    private ILayer<T> CreateDownsample(int channels, int level)
    {
        int res = ResolutionAtLevel(level);
        return LazyConv2D(
            inputDepth: channels,
            inputHeight: res,
            inputWidth: res,
            outputDepth: channels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            activation: new IdentityActivation<T>());
    }

    private ILayer<T> CreateUpsample(int channels, int level)
    {
        // (output of the corresponding encoder downsample) and upsamples to
        // _inputHeight >> level (the paired encoder-level resolution).
        // Transposed convolution: stride=2, kernel=4, padding=1 ⇒ output = 2 * input.
        // Note: ResolutionAtLevel uses _inputHeight; non-square inputs
        // (_inputHeight != _inputWidth) would produce incorrect attention
        // sequence lengths — documented on the constructor's inputHeight param.
        int inputRes = ResolutionAtLevel(level + 1);
        return DeconvolutionalLayer<T>.WithInputDepth(
            inputDepth: channels,
            outputDepth: channels,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new IdentityActivation<T>());
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Enumerates every layer in the EXACT order used by <see cref="GetParameters"/> /
    /// <see cref="SetParameters"/> (input conv, time-embed MLPs, image-cond projection,
    /// then each encoder/middle/decoder block's components, then the output conv). Used
    /// by <see cref="Clone"/> for a paired per-layer weight copy that needs no shape-
    /// resolution forward. Null slots (attention disabled at a level, or a pure
    /// Downsample/Upsample block) are yielded as null so source and clone enumerations
    /// stay index-aligned by construction.
    /// </summary>
    private IEnumerable<ILayer<T>?> EnumerateLayersInParameterOrder()
    {
        var layers = new List<ILayer<T>?>
        {
            _inputConv,
            _timeEmbedMlp1,
            _timeEmbedMlp2,
            _imageCondProjection,
            _classEmbedding
        };

        foreach (var block in _encoderBlocks)
            layers.AddRange(BlockLayersInParameterOrder(block));
        foreach (var block in _middleBlocks)
            layers.AddRange(BlockLayersInParameterOrder(block));
        foreach (var block in _decoderBlocks)
            layers.AddRange(BlockLayersInParameterOrder(block));

        layers.Add(_outputNorm);
        layers.Add(_outputConv);
        return layers;
    }

    // This is the single canonical component order for values and gradients.
    private static IEnumerable<ILayer<T>?> BlockLayersInParameterOrder(VideoBlock block)
    {
        yield return block.SpatialResBlock;
        yield return block.TimeCondProjection;
        yield return block.TemporalResBlock;
        yield return block.SpatialAttention;
        yield return block.TemporalAttention;
        yield return block.CrossAttention;
        yield return block.Downsample;
        yield return block.Upsample;
    }

    private IReadOnlyList<ILayer<T>> BuildTemporalTrainingLayers()
    {
        var layers = new List<ILayer<T>>();
        var seen = new HashSet<ILayer<T>>(AiDotNet.Helpers.TensorReferenceComparer<ILayer<T>>.Instance);

        foreach (var block in _encoderBlocks.Concat(_middleBlocks).Concat(_decoderBlocks))
        {
            Add(block.TemporalResBlock);
            Add(block.TemporalAttention);
            if (block.SpatialAttention is VideoTransformer3DLayer<T> transformer)
                foreach (var temporalLayer in transformer.TemporalTrainingLayers)
                    Add(temporalLayer);
        }

        return layers.AsReadOnly();

        void Add(ILayer<T>? layer)
        {
            if (layer is not null && seen.Add(layer)) layers.Add(layer);
        }
    }

    private void AddLayerParameters(List<T> parameters, ILayer<T>? layer)
    {
        if (layer == null) return;
        var layerParams = layer.GetParameters();
        for (int i = 0; i < layerParams.Length; i++)
        {
            parameters.Add(layerParams[i]);
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();
        foreach (var layer in EnumerateLayersInParameterOrder())
            AddLayerParameters(parameters, layer);
        return new Vector<T>(parameters.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var layers = EnumerateLayersInParameterOrder()
            .Where(layer => layer is not null)
            .Cast<ILayer<T>>()
            .ToArray();
        var lengths = new int[layers.Length];
        long expected = 0;
        for (int i = 0; i < layers.Length; i++)
        {
            lengths[i] = layers[i].GetParameters().Length;
            expected = checked(expected + lengths[i]);
        }

        if (expected != parameters.Length)
            throw new ArgumentException(
                $"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        int offset = 0;
        for (int layerIndex = 0; layerIndex < layers.Length; layerIndex++)
        {
            int length = lengths[layerIndex];
            var layerParameters = new Vector<T>(length);
            for (int i = 0; i < length; i++)
                layerParameters[i] = parameters[offset++];
            layers[layerIndex].SetParameters(layerParameters);
        }
    }

    #endregion

    #region ICloneable Implementation

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new VideoUNetPredictor<T>(
            _inputChannels,
            _outputChannels,
            _baseChannels,
            _channelMultipliers,
            _numResBlocks,
            _attentionResolutions,
            _numTemporalLayers,
            _contextDim,
            _numHeads,
            _supportsImageConditioning,
            _inputHeight,
            _inputWidth,
            _numFrames,
            _clipTokenLength,
            LossFunction,
            seed: null,
            imageConditionChannels: _imageConditionChannels,
            concatenateImageCondition: _concatenateImageCondition,
            numClassEmbeddings: _numClassEmbeddings,
            architectureProfile: _architectureProfile);

        // Resolve the SOURCE's lazy layers so each source layer reports its real
        // parameter shape below. The source's resolving forward packs its OWN
        // (correct) weights, so the source stays self-consistent.
        TriggerLazyShapeResolution(
            includeImageConditioning: _supportsImageConditioning && _concatenateImageCondition);

        bool sourceUsedVideo = _lazyShapeResolvedWithVideo;
        bool sourceUsedTextConditioning = _lazyShapeResolvedWithTextConditioning;
        bool sourceUsedImageConditioning = _lazyShapeResolvedWithImageConditioning;

        // Materialize the clone with the same execution path, then copy values into its existing
        // tensors. This preserves layer-owned caches and avoids relying on SetParameters to infer
        // a lazy tensor's shape from a flat length (which is ambiguous for grouped/deconvolutional
        // kernels and caused output-divergent clones).
        clone.TriggerLazyShapeResolution(
            sourceUsedVideo,
            sourceUsedTextConditioning,
            sourceUsedImageConditioning);
        using (var srcEnum = EnumerateLayersInParameterOrder().GetEnumerator())
        using (var cloneEnum = clone.EnumerateLayersInParameterOrder().GetEnumerator())
        {
            while (srcEnum.MoveNext() && cloneEnum.MoveNext())
            {
                var srcLayer = srcEnum.Current;
                var cloneLayer = cloneEnum.Current;
                if (srcLayer is null || cloneLayer is null)
                    continue;
                cloneLayer.SetParameters(srcLayer.GetParameters());
            }
        }
        return clone;
    }

    /// <summary>
    /// Runs a single dummy forward through the network at the configured
    /// spatial / frame size so every lazy layer (time-embedding MLPs,
    /// temporal + cross attention, and the image-condition projection)
    /// resolves its weight shapes. Used by <see cref="Clone"/> to make the
    /// clone's layer parameter counts match the original's before
    /// <see cref="SetParameters"/> copies weights across. Mirrors
    /// UNetNoisePredictor.TriggerLazyShapeResolution.
    /// </summary>
    internal void TriggerLazyShapeResolution(
        bool includeVideo = false,
        bool includeTextConditioning = false,
        bool includeImageConditioning = false)
    {
        if (_lazyShapeResolved
            && (!includeVideo || _lazyShapeResolvedWithVideo)
            && (!includeTextConditioning || _lazyShapeResolvedWithTextConditioning)
            && (!includeImageConditioning || _lazyShapeResolvedWithImageConditioning))
        {
            return;
        }

        // Mirror the model's REAL Predict path exactly: a 4D image-mode forward
        // with no conditioning. LatentDiffusionModelBase.Predict drives the
        // denoising loop with a 4D latent [B, LatentChannels, H, W] and calls
        // NoisePredictor.PredictNoise(sample, t, null) — so isVideo=false and
        // the temporal-mixing, image-condition, and cross-attention layers are
        // never touched. Resolving those here (via a 5D video pass) would leave
        // them resolved on the clone but with state the real forward never
        // exercises, and the temporal-mixing DenseLayer is sized strictly
        // [_numFrames -> _numFrames] so any partial video pass risks shape
        // mismatches. The lazy layers the real forward DOES use (input/output
        // convs, spatial ResBlocks, spatial attention, time-embedding and FiLM
        // MLPs) all resolve to spatial-independent shapes, so a tiny dummy
        // resolves identical shapes on both source and clone at negligible cost.
        // Layers left unresolved stay at 0 params on BOTH sides, so the
        // GetParameters/SetParameters copy stays index-aligned. Mirrors
        // UNetNoisePredictor.TriggerLazyShapeResolution.
        int levels = _channelMultipliers.Length;
        int side = 1 << System.Math.Max(0, levels - 1);
        int sideH = System.Math.Min(_inputHeight, side);
        int sideW = System.Math.Min(_inputWidth, side);
        int frames = includeVideo ? System.Math.Max(1, _numFrames) : 1;
        int[] sampleShape = includeVideo
            ? [1, _inputChannels, frames, sideH, sideW]
            : [1, _inputChannels, sideH, sideW];
        var dummy = new Tensor<T>(sampleShape);
        Tensor<T>? textConditioning = includeTextConditioning
            ? new Tensor<T>([1, System.Math.Max(1, _clipTokenLength), _contextDim])
            : null;

        if (includeImageConditioning)
        {
            if (!_supportsImageConditioning)
                throw new InvalidOperationException(
                    "Cannot resolve an image-conditioning path on a predictor that does not support it.");

            int[] conditionShape = includeVideo
                ? [1, _imageConditionChannels, frames, sideH, sideW]
                : [1, _imageConditionChannels, sideH, sideW];
            var condition = new Tensor<T>(conditionShape);
            if (includeVideo)
            {
                _ = PredictNoiseWithVideoCondition(
                    dummy,
                    timestep: 0,
                    condition,
                    textConditioning,
                    noiseLevel: _classEmbedding is null ? null : 0);
            }
            else
            {
                _ = PredictNoiseWithImageCondition(
                    dummy, timestep: 0, condition, textConditioning);
            }
            return;
        }

        _ = PredictNoise(dummy, timestep: 0, conditioning: textConditioning);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    #endregion

    #region Layer-Level Backpropagation

    private static void BackwardBlock(VideoBlock block, ref Tensor<T> grad)
    {
    }

    protected override Vector<T> GetParameterGradients()
    {
        var gradients = new List<T>();
        foreach (var layer in EnumerateLayersInParameterOrder())
            AddLayerGradients(gradients, layer);

        return new Vector<T>(gradients.ToArray());
    }

    private void AddLayerGradients(List<T> gradients, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) gradients.Add(g[i]);
    }

    #endregion

    /// <summary>
    /// Internal structure for video U-Net blocks.
    /// </summary>
    private class VideoBlock
    {
        public ILayer<T>? SpatialResBlock { get; set; }
        public ILayer<T>? TemporalResBlock { get; set; }
        public ILayer<T>? SpatialAttention { get; set; }
        public ILayer<T>? TemporalAttention { get; set; }
        public ILayer<T>? CrossAttention { get; set; }
        public ILayer<T>? Downsample { get; set; }
        public ILayer<T>? Upsample { get; set; }
        public bool CaptureSkipAfter { get; set; }
        public bool ConsumesSkip { get; set; }

        /// <summary>
        /// FiLM conditioning projection: timeEmbedDim → channels * 2 (scale + shift).
        /// Per Ho et al. 2022 "Video Diffusion Models" §3.1, each residual block receives
        /// the timestep embedding and modulates its feature maps via
        /// <c>x = x * (1 + scale) + shift</c>, where <c>[scale, shift]</c> are linearly
        /// projected from the time embedding. This is the standard Adaptive Group
        /// Normalization (AdaGN) pattern from Dhariwal &amp; Nichol 2021 (ADM).
        /// </summary>
        public DenseLayer<T>? TimeCondProjection { get; set; }
    }
}
