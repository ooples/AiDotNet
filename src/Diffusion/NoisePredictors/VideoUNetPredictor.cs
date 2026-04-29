using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

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
    /// Image conditioning projection (for image-to-video).
    /// </summary>
    private ConvolutionalLayer<T>? _imageCondProjection;

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
    public override int ParameterCount => CalculateParameterCount();

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

    /// <summary>
    /// Gets the number of temporal transformer layers.
    /// </summary>
    public int NumTemporalLayers => _numTemporalLayers;

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
        int? seed = null)
        : base(lossFunction, seed)
    {
        _inputChannels = inputChannels;
        _outputChannels = outputChannels ?? inputChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers ?? new[] { 1, 2, 4, 4 };
        _numResBlocks = numResBlocks;
        _attentionResolutions = attentionResolutions ?? new[] { 1, 2, 3 };
        _numTemporalLayers = numTemporalLayers;
        _contextDim = contextDim;
        _numHeads = numHeads;
        _timeEmbeddingDim = baseChannels * 4;
        _supportsImageConditioning = supportsImageConditioning;
        _inputHeight = inputHeight;
        _inputWidth = inputWidth;
        _numFrames = numFrames;
        _clipTokenLength = clipTokenLength;

        _encoderBlocks = new List<VideoBlock>();
        _middleBlocks = new List<VideoBlock>();
        _decoderBlocks = new List<VideoBlock>();

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
        _inputConv = LazyConv2D(
            inputDepth: _inputChannels,
            inputHeight: _inputHeight,
            inputWidth: _inputWidth,
            outputDepth: _baseChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: new IdentityActivation<T>());

        // Time embedding MLP — lazy so constructor-time memory stays flat.
        _timeEmbedMlp1 = LazyDense(_timeEmbeddingDim / 4, _timeEmbeddingDim, new SiLUActivation<T>());
        _timeEmbedMlp2 = LazyDense(_timeEmbeddingDim, _timeEmbeddingDim, new SiLUActivation<T>());

        // Image conditioning projection (for image-to-video)
        if (_supportsImageConditioning)
        {
            _imageCondProjection = LazyConv2D(
                inputDepth: _inputChannels,
                inputHeight: _inputHeight,
                inputWidth: _inputWidth,
                outputDepth: _baseChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
                activation: new IdentityActivation<T>());
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
                    CrossAttention = useAttention && _contextDim > 0 ? CreateCrossAttention(outChannels) : null,
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
            CrossAttention = _contextDim > 0 ? CreateCrossAttention(inChannels) : null,
            TimeCondProjection = CreateTimeCondProjection(inChannels)
        });
        _middleBlocks.Add(new VideoBlock
        {
            SpatialResBlock = CreateSpatialResBlock(inChannels, inChannels),
            TemporalResBlock = CreateTemporalMixingBlock(),
            TimeCondProjection = CreateTimeCondProjection(inChannels)
        });

        // Build decoder (reverse of encoder)
        for (int level = _channelMultipliers.Length - 1; level >= 0; level--)
        {
            var outChannels = _baseChannels * _channelMultipliers[level];
            var useAttention = Array.IndexOf(_attentionResolutions, level) >= 0;

            for (int block = 0; block <= _numResBlocks; block++)
            {
                var skipChannels = block == 0 && level < _channelMultipliers.Length - 1
                    ? _baseChannels * _channelMultipliers[level + 1]
                    : outChannels;

                _decoderBlocks.Add(new VideoBlock
                {
                    SpatialResBlock = CreateSpatialResBlock(inChannels + skipChannels, outChannels),
                    TemporalResBlock = CreateTemporalMixingBlock(),
                    SpatialAttention = useAttention ? CreateSpatialAttention(outChannels, level) : null,
                    TemporalAttention = useAttention ? CreateTemporalAttention(outChannels) : null,
                    CrossAttention = useAttention && _contextDim > 0 ? CreateCrossAttention(outChannels) : null,
                    TimeCondProjection = CreateTimeCondProjection(outChannels)
                });
                inChannels = outChannels;
            }

            // Add upsampling except for first level
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

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        // Compute timestep embedding
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        // Forward pass
        return ForwardVideoUNet(noisySample, timeEmbed, conditioning, imageCondition: null);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;

        var timeEmbed = ProjectTimeEmbedding(timeEmbedding);
        return ForwardVideoUNet(noisySample, timeEmbed, conditioning, imageCondition: null);
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

        _lastInput = noisySample;

        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);

        return ForwardVideoUNet(noisySample, timeEmbed, textConditioning, imageCondition);
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

        // Process each frame through input conv (or use 3D conv in production)
        x = isVideo
            ? ProcessVideoFrames(x, frame => _inputConv.Forward(frame))
            : _inputConv.Forward(x);

        // Add image condition (for image-to-video)
        if (imageCondition != null && _imageCondProjection != null)
        {
            var imageCond = _imageCondProjection.Forward(imageCondition);
            // Broadcast to all frames and add
            x = AddImageCondition(x, imageCond, numFrames);
        }

        // Store skip connections
        var skips = new List<Tensor<T>>();

        // Encoder
        foreach (var block in _encoderBlocks)
        {
            if (block.Downsample != null)
            {
                x = ApplyDownsample(block.Downsample, x, isVideo);
            }
            else
            {
                x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
                skips.Add(x);
            }
        }

        // Middle
        foreach (var block in _middleBlocks)
        {
            x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
        }

        // Decoder
        var skipIdx = skips.Count - 1;
        foreach (var block in _decoderBlocks)
        {
            if (block.Upsample != null)
            {
                x = ApplyUpsample(block.Upsample, x, isVideo);
            }
            else
            {
                if (skipIdx >= 0)
                {
                    x = ConcatenateChannels(x, skips[skipIdx], isVideo);
                    skipIdx--;
                }

                x = ApplyVideoBlock(block, x, timeEmbed, textConditioning, isVideo);
            }
        }

        // Output convolution
        x = isVideo
            ? ProcessVideoFrames(x, frame => _outputConv.Forward(frame))
            : _outputConv.Forward(x);

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
        // Spatial ResBlock
        if (block.SpatialResBlock != null)
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
        if (block.TemporalResBlock != null && isVideo)
        {
            x = ApplyTemporalProcessing(block.TemporalResBlock, x);
        }

        // Spatial attention
        if (block.SpatialAttention != null)
        {
            x = isVideo
                ? ProcessVideoFrames(x, frame => block.SpatialAttention.Forward(frame))
                : block.SpatialAttention.Forward(x);
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
                ? ProcessVideoFrames(x, frame => crossAttnBase.Forward(frame, conditioning))
                : crossAttnBase.Forward(x, conditioning);
        }

        return x;
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
        var onePlusScale = Engine.TensorBroadcastAdd(
            scaleTensor,
            Tensor<T>.CreateDefault(broadcastShape, NumOps.One));
        var modulated = Engine.TensorBroadcastMultiply(x, onePlusScale);
        return Engine.TensorBroadcastAdd(modulated, shiftTensor);
    }

    /// <summary>
    /// Processes each frame of a video through a layer.
    /// </summary>
    private Tensor<T> ProcessVideoFrames(Tensor<T> video, Func<Tensor<T>, Tensor<T>> processFrame)
    {
        int frames = video.Shape[2];
        var processedFrames = new List<Tensor<T>>();

        for (int f = 0; f < frames; f++)
        {
            var frame = ExtractFrame(video, f);
            var processed = processFrame(frame);
            processedFrames.Add(processed);
        }

        return StackFrames(processedFrames);
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
        var shape = video.Shape.ToArray();
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

    private ILayer<T> CreateSpatialResBlock(int inChannels, int outChannels)
    {
        // A diffusion spatial ResBlock transforms the channel dimension of a
        // 4D [B, C, H, W] feature map. The previous implementation used
        // LazyDense(inChannels, outChannels), but DenseLayer projects the
        // *last* dimension of its input — for a 4D tensor that is the width
        // axis, not channels. The block therefore left C unchanged while
        // scrambling W to outChannels, and every subsequent TimeCondProjection
        // (sized for the planned outChannels) saw a feature map still at the
        // incoming channel count — hence "FiLM conditioning projection width
        // mismatch: expected 640, got 1280" on upscaleavideomodel and
        // streamingt2vmodel tests.
        // A 1x1 Conv2D is the standard channel-mixing primitive: it consumes
        // [B, inChannels, H, W] and produces [B, outChannels, H, W] while
        // leaving spatial dims alone.
        return LazyConv2D(
            inputDepth: inChannels,
            inputHeight: 1,
            inputWidth: 1,
            outputDepth: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            activation: new SiLUActivation<T>());
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

    private ILayer<T> CreateSpatialAttention(int channels, int level)
    {
        int res = ResolutionAtLevel(level);
        return LazyMHA(res * res, channels, _numHeads, new IdentityActivation<T>());
    }

    private ILayer<T> CreateTemporalAttention(int channels)
    {
        return LazyMHA(_numFrames, channels, _numHeads, new IdentityActivation<T>());
    }

    private ILayer<T> CreateCrossAttention(int channels)
    {
        return LazyMHA(_clipTokenLength, channels, _numHeads, new IdentityActivation<T>());
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
        return new DeconvolutionalLayer<T>(
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
        long count = 0;

        // Input/output convolutions
        count += _inputChannels * _baseChannels * 9 + _baseChannels;
        count += _baseChannels * _outputChannels * 9 + _outputChannels;

        // Time embedding MLP
        count += (_timeEmbeddingDim / 4) * _timeEmbeddingDim + _timeEmbeddingDim;
        count += _timeEmbeddingDim * _timeEmbeddingDim + _timeEmbeddingDim;

        // Image conditioning
        if (_supportsImageConditioning)
        {
            count += _inputChannels * _baseChannels + _baseChannels;
        }

        // Estimate blocks
        foreach (var channels in _channelMultipliers.Select(mult => _baseChannels * mult))
        {
            count += _numResBlocks * (channels * channels * 4); // Spatial + temporal
        }

        return (int)Math.Min(count, int.MaxValue);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new List<T>();

        AddLayerParameters(parameters, _inputConv);
        AddLayerParameters(parameters, _timeEmbedMlp1);
        AddLayerParameters(parameters, _timeEmbedMlp2);
        AddLayerParameters(parameters, _imageCondProjection);

        foreach (var block in _encoderBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        foreach (var block in _middleBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        foreach (var block in _decoderBlocks)
        {
            AddBlockParameters(parameters, block);
        }

        AddLayerParameters(parameters, _outputConv);

        return new Vector<T>(parameters.ToArray());
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

    private void AddBlockParameters(List<T> parameters, VideoBlock block)
    {
        AddLayerParameters(parameters, block.SpatialResBlock);
        AddLayerParameters(parameters, block.TemporalResBlock);
        AddLayerParameters(parameters, block.SpatialAttention);
        AddLayerParameters(parameters, block.TemporalAttention);
        AddLayerParameters(parameters, block.CrossAttention);
        AddLayerParameters(parameters, block.Downsample);
        AddLayerParameters(parameters, block.Upsample);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var index = 0;

        SetLayerParameters(_inputConv, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp1, parameters, ref index);
        SetLayerParameters(_timeEmbedMlp2, parameters, ref index);
        SetLayerParameters(_imageCondProjection, parameters, ref index);

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

    private void SetBlockParameters(VideoBlock block, Vector<T> parameters, ref int index)
    {
        SetLayerParameters(block.SpatialResBlock, parameters, ref index);
        SetLayerParameters(block.TemporalResBlock, parameters, ref index);
        SetLayerParameters(block.SpatialAttention, parameters, ref index);
        SetLayerParameters(block.TemporalAttention, parameters, ref index);
        SetLayerParameters(block.CrossAttention, parameters, ref index);
        SetLayerParameters(block.Downsample, parameters, ref index);
        SetLayerParameters(block.Upsample, parameters, ref index);
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
            LossFunction);

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

    private static void BackwardBlock(VideoBlock block, ref Tensor<T> grad)
    {
    }

    protected override Vector<T> GetParameterGradients()
    {
        var gradients = new List<T>();

        AddLayerGradients(gradients, _inputConv);
        AddLayerGradients(gradients, _timeEmbedMlp1);
        AddLayerGradients(gradients, _timeEmbedMlp2);
        AddLayerGradients(gradients, _imageCondProjection);

        foreach (var block in _encoderBlocks) AddBlockGradients(gradients, block);
        foreach (var block in _middleBlocks) AddBlockGradients(gradients, block);
        foreach (var block in _decoderBlocks) AddBlockGradients(gradients, block);

        AddLayerGradients(gradients, _outputConv);

        return new Vector<T>(gradients.ToArray());
    }

    private void AddLayerGradients(List<T> gradients, ILayer<T>? layer)
    {
        if (layer == null) return;
        var g = layer.GetParameterGradients();
        for (int i = 0; i < g.Length; i++) gradients.Add(g[i]);
    }

    private void AddBlockGradients(List<T> gradients, VideoBlock block)
    {
        AddLayerGradients(gradients, block.SpatialResBlock);
        AddLayerGradients(gradients, block.TemporalResBlock);
        AddLayerGradients(gradients, block.SpatialAttention);
        AddLayerGradients(gradients, block.TemporalAttention);
        AddLayerGradients(gradients, block.CrossAttention);
        AddLayerGradients(gradients, block.Downsample);
        AddLayerGradients(gradients, block.Upsample);
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
