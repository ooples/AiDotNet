using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// ControlNet model for adding spatial conditioning to diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// ControlNet enables fine-grained spatial control over image generation by adding
/// additional conditioning signals such as edge maps, depth maps, pose keypoints,
/// segmentation masks, and more. It works by creating a trainable copy of the
/// encoder blocks that process the control signal.
/// </para>
/// <para>
/// <b>For Beginners:</b> ControlNet is like giving the AI artist a reference sketch
/// or blueprint to follow while creating an image.
///
/// Supported control types:
/// - Canny edges: Outline/edge detection of shapes
/// - Depth maps: 3D depth information
/// - Pose keypoints: Human body positions (OpenPose)
/// - Segmentation: Region/object boundaries
/// - Normal maps: Surface orientation
/// - Scribbles: Simple user drawings
/// - Line art: Clean line drawings
///
/// How it works:
/// 1. You provide a control image (e.g., edge map of a house)
/// 2. ControlNet encodes this control signal
/// 3. The encoded control guides the diffusion process
/// 4. Result: Generated image follows the control structure
///
/// Example use cases:
/// - "Draw a Victorian house" + edge map = house in exact shape
/// - "Dancing woman" + pose skeleton = person in exact pose
/// - "Forest scene" + depth map = correct 3D perspective
/// </para>
/// <para>
/// Technical details:
/// - ControlNet is a "zero convolution" architecture
/// - Copies encoder weights from base model
/// - Adds control signal via residual connections
/// - Can be combined: multi-ControlNet stacking
/// - Supports conditioning strength adjustment (0-1)
///
/// Reference: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a ControlNet model
/// var controlNet = new ControlNetModel&lt;float&gt;(
///     controlType: ControlType.Canny);
///
/// // Generate with edge control
/// var edgeMap = LoadCannyEdges("house_edges.png");
/// var image = controlNet.GenerateWithControl(
///     prompt: "A beautiful Victorian house",
///     controlImage: edgeMap,
///     conditioningStrength: 1.0);
///
/// // Multi-control generation
/// var depthMap = LoadDepthMap("scene_depth.png");
/// var imageMulti = controlNet.GenerateWithMultiControl(
///     prompt: "Forest landscape",
///     controlImages: new[] { edgeMap, depthMap },
///     controlTypes: new[] { ControlType.Canny, ControlType.Depth },
///     conditioningStrengths: new[] { 0.8, 0.6 });
/// </code>
/// </example>
public class ControlNetModel<T> : LatentDiffusionModelBase<T>
{
    /// <summary>
    /// Standard ControlNet latent channels.
    /// </summary>
    private const int CN_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    private const int CN_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// The base noise predictor (U-Net).
    /// </summary>
    private readonly UNetNoisePredictor<T> _baseUNet;

    /// <summary>
    /// The ControlNet encoder blocks for the primary control type.
    /// </summary>
    private readonly ControlNetEncoder<T> _controlNetEncoder;

    /// <summary>
    /// Cache of encoders by control type for multi-control generation.
    /// </summary>
    private readonly Dictionary<ControlType, ControlNetEncoder<T>> _encoderCache;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The conditioning module for text encoding.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// The type of control signal this model handles.
    /// </summary>
    private readonly ControlType _controlType;

    /// <summary>
    /// Number of input channels for control signal.
    /// </summary>
    private readonly int _controlChannels;

    /// <summary>
    /// Default conditioning strength.
    /// </summary>
    private double _conditioningStrength = 1.0;

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => CN_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            var count = _baseUNet.ParameterCount;
            foreach (var encoder in _encoderCache.Values)
            {
                count += encoder.ParameterCount;
            }
            return count;
        }
    }

    /// <summary>
    /// Gets the type of control signal this model uses.
    /// </summary>
    public ControlType ControlType => _controlType;

    /// <summary>
    /// Gets or sets the default conditioning strength (0-1).
    /// </summary>
    public double ConditioningStrength
    {
        get => _conditioningStrength;
        set => _conditioningStrength = MathPolyfill.Clamp(value, 0.0, 1.0);
    }

    /// <summary>
    /// Initializes a new instance of ControlNetModel with default parameters.
    /// </summary>
    /// <param name="controlType">The type of control signal (default: Canny edges).</param>
    public ControlNetModel(ControlType controlType = ControlType.Canny)
        : this(
            options: null,
            scheduler: null,
            conditioner: null,
            controlType: controlType,
            seed: null)
    {
    }

    /// <summary>
    /// Initializes a new instance of ControlNetModel with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="baseUNet">Optional base U-Net noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text encoding.</param>
    /// <param name="controlType">The type of control signal.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ControlNetModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        ControlType controlType = ControlType.Canny,
        int? seed = null)
        : base(options ?? CreateDefaultOptions(), scheduler ?? CreateDefaultScheduler())
    {
        _controlType = controlType;
        _conditioner = conditioner;
        _controlChannels = GetControlChannels(controlType);

        // Create base U-Net
        _baseUNet = baseUNet ?? CreateDefaultUNet(seed);

        // Create VAE
        _vae = vae ?? CreateDefaultVAE(seed);

        // Create ControlNet encoder
        _controlNetEncoder = new ControlNetEncoder<T>(
            inputChannels: _controlChannels,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            seed: seed);

        // Initialize encoder cache with the primary encoder
        _encoderCache = new Dictionary<ControlType, ControlNetEncoder<T>>
        {
            { _controlType, _controlNetEncoder }
        };
    }

    /// <summary>
    /// Creates the default options.
    /// </summary>
    private static DiffusionModelOptions<T> CreateDefaultOptions()
    {
        return new DiffusionModelOptions<T>
        {
            TrainTimesteps = 1000,
            BetaStart = 0.00085,
            BetaEnd = 0.012,
            BetaSchedule = BetaSchedule.ScaledLinear
        };
    }

    /// <summary>
    /// Creates the default scheduler.
    /// </summary>
    private static INoiseScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default U-Net.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet(int? seed)
    {
        return new UNetNoisePredictor<T>(
            inputChannels: CN_LATENT_CHANNELS,
            outputChannels: CN_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);
    }

    /// <summary>
    /// Creates the default VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE(int? seed)
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: CN_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);
    }

    /// <summary>
    /// Gets the number of input channels for a control type.
    /// </summary>
    private static int GetControlChannels(ControlType type)
    {
        return type switch
        {
            ControlType.Canny => 1,      // Single channel edge map
            ControlType.Depth => 1,      // Single channel depth
            ControlType.Normal => 3,     // RGB normal map
            ControlType.Pose => 3,       // RGB pose visualization
            ControlType.Segmentation => 3, // RGB segmentation
            ControlType.Scribble => 1,   // Single channel scribble
            ControlType.LineArt => 1,    // Single channel line art
            ControlType.Hed => 1,        // Single channel HED edges
            ControlType.Mlsd => 1,       // Single channel MLSD lines
            _ => 3                       // Default to RGB
        };
    }

    /// <summary>
    /// Gets or creates a cached encoder for the specified control type.
    /// This ensures encoders are reused and their trained weights are preserved.
    /// </summary>
    /// <param name="controlType">The type of control signal.</param>
    /// <returns>A cached or newly created encoder for the control type.</returns>
    private ControlNetEncoder<T> GetOrCreateEncoder(ControlType controlType)
    {
        if (_encoderCache.TryGetValue(controlType, out var encoder))
        {
            return encoder;
        }

        // Create a new encoder for this control type and cache it
        var newEncoder = new ControlNetEncoder<T>(
            inputChannels: GetControlChannels(controlType),
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 });

        _encoderCache[controlType] = newEncoder;
        return newEncoder;
    }

    /// <summary>
    /// Generates an image with control signal.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="controlImage">The control image (e.g., edge map, depth map).</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output image width.</param>
    /// <param name="height">Output image height.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="conditioningStrength">How strongly to apply the control (0-1).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The generated image tensor.</returns>
    public virtual Tensor<T> GenerateWithControl(
        string prompt,
        Tensor<T> controlImage,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        double? conditioningStrength = null,
        int? seed = null)
    {
        var effectiveStrength = conditioningStrength ?? _conditioningStrength;

        // Encode control image
        var controlFeatures = _controlNetEncoder.Encode(controlImage);

        // Scale control features by conditioning strength
        if (Math.Abs(effectiveStrength - 1.0) > 1e-6)
        {
            for (int i = 0; i < controlFeatures.Count; i++)
            {
                controlFeatures[i] = ScaleTensor(controlFeatures[i], effectiveStrength);
            }
        }

        // Get text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            if (effectiveGuidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                    negativeEmbedding = _conditioner.EncodeText(negTokens);
                }
                else
                {
                    negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
                }
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / CN_VAE_SCALE_FACTOR;
        var latentWidth = width / CN_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, CN_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop with control
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (effectiveGuidanceScale > 1.0 && negativeEmbedding != null && promptEmbedding != null)
            {
                // Classifier-free guidance with control
                var condPred = PredictWithControl(latents, timestep, promptEmbedding, controlFeatures);
                var uncondPred = PredictWithControl(latents, timestep, negativeEmbedding, controlFeatures);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = PredictWithControl(latents, timestep, promptEmbedding, controlFeatures);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to image
        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Generates an image with multiple control signals.
    /// </summary>
    /// <param name="prompt">The text prompt.</param>
    /// <param name="controlImages">Array of control images.</param>
    /// <param name="controlTypes">Array of control types.</param>
    /// <param name="conditioningStrengths">Array of conditioning strengths.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width.</param>
    /// <param name="height">Output height.</param>
    /// <param name="numInferenceSteps">Number of steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>The generated image.</returns>
    public virtual Tensor<T> GenerateWithMultiControl(
        string prompt,
        Tensor<T>[] controlImages,
        ControlType[] controlTypes,
        double[]? conditioningStrengths = null,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        if (controlImages.Length != controlTypes.Length)
        {
            throw new ArgumentException("Number of control images must match number of control types.");
        }

        // Default strengths to 1.0
        var strengths = conditioningStrengths ?? Enumerable.Repeat(1.0, controlImages.Length).ToArray();

        // Encode all control images and combine features
        var combinedFeatures = new List<Tensor<T>>();
        var isFirst = true;

        for (int i = 0; i < controlImages.Length; i++)
        {
            // Use cached encoder to preserve trained weights
            var encoder = GetOrCreateEncoder(controlTypes[i]);
            var features = encoder.Encode(controlImages[i]);

            // Scale by strength
            for (int j = 0; j < features.Count; j++)
            {
                features[j] = ScaleTensor(features[j], strengths[i]);
            }

            if (isFirst)
            {
                combinedFeatures = features;
                isFirst = false;
            }
            else
            {
                // Add features from additional controls
                for (int j = 0; j < combinedFeatures.Count && j < features.Count; j++)
                {
                    combinedFeatures[j] = AddTensors(combinedFeatures[j], features[j]);
                }
            }
        }

        // Continue with standard generation using combined features
        return GenerateWithControlFeatures(
            prompt,
            combinedFeatures,
            negativePrompt,
            width,
            height,
            numInferenceSteps,
            guidanceScale,
            seed);
    }

    /// <summary>
    /// Generates an image using pre-computed control features.
    /// </summary>
    private Tensor<T> GenerateWithControlFeatures(
        string prompt,
        List<Tensor<T>> controlFeatures,
        string? negativePrompt,
        int width,
        int height,
        int numInferenceSteps,
        double? guidanceScale,
        int? seed)
    {
        // Get text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(prompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            if (effectiveGuidanceScale > 1.0)
            {
                if (!string.IsNullOrEmpty(negativePrompt))
                {
                    var negTokens = _conditioner.Tokenize(negativePrompt ?? string.Empty);
                    negativeEmbedding = _conditioner.EncodeText(negTokens);
                }
                else
                {
                    negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
                }
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / CN_VAE_SCALE_FACTOR;
        var latentWidth = width / CN_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, CN_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (effectiveGuidanceScale > 1.0 && negativeEmbedding != null && promptEmbedding != null)
            {
                var condPred = PredictWithControl(latents, timestep, promptEmbedding, controlFeatures);
                var uncondPred = PredictWithControl(latents, timestep, negativeEmbedding, controlFeatures);
                noisePrediction = ApplyGuidance(uncondPred, condPred, effectiveGuidanceScale);
            }
            else
            {
                noisePrediction = PredictWithControl(latents, timestep, promptEmbedding, controlFeatures);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Predicts noise with control signal integration.
    /// </summary>
    private Tensor<T> PredictWithControl(
        Tensor<T> latents,
        int timestep,
        Tensor<T>? conditioning,
        List<Tensor<T>> controlFeatures)
    {
        // Get base prediction
        var basePrediction = _baseUNet.PredictNoise(latents, timestep, conditioning);

        // Add control features (residual connection)
        // In practice, this would inject at various U-Net levels
        // Simplified version: add scaled control to output
        if (controlFeatures.Count > 0)
        {
            var controlSum = controlFeatures[controlFeatures.Count - 1];
            basePrediction = AddTensors(basePrediction, controlSum);
        }

        return basePrediction;
    }

    /// <summary>
    /// Scales a tensor by a scalar value.
    /// </summary>
    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale)
    {
        var result = new Tensor<T>(tensor.Shape);
        var inputSpan = tensor.AsSpan();
        var resultSpan = result.AsWritableSpan();
        var scaleT = NumOps.FromDouble(scale);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Multiply(scaleT, inputSpan[i]);
        }

        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise with proper shape handling.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        // Use the larger tensor's shape for the result
        var resultShape = aSpan.Length >= bSpan.Length ? a.Shape : b.Shape;
        var result = new Tensor<T>(resultShape);
        var resultSpan = result.AsWritableSpan();

        var minLen = Math.Min(aSpan.Length, bSpan.Length);
        var maxLen = Math.Max(aSpan.Length, bSpan.Length);

        // Add overlapping elements
        for (int i = 0; i < minLen; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        // Copy remaining elements from the larger tensor
        if (aSpan.Length > minLen)
        {
            for (int i = minLen; i < maxLen; i++)
            {
                resultSpan[i] = aSpan[i];
            }
        }
        else if (bSpan.Length > minLen)
        {
            for (int i = minLen; i < maxLen; i++)
            {
                resultSpan[i] = bSpan[i];
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Add base UNet parameters
        var baseParams = _baseUNet.GetParameters();
        for (int i = 0; i < baseParams.Length; i++)
        {
            allParams.Add(baseParams[i]);
        }

        // Add all cached encoder parameters (in deterministic order)
        foreach (var kvp in _encoderCache.OrderBy(kv => kv.Key))
        {
            var encoderParams = kvp.Value.GetParameters();
            for (int i = 0; i < encoderParams.Length; i++)
            {
                allParams.Add(encoderParams[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Set base UNet parameters
        var baseCount = _baseUNet.ParameterCount;
        var baseParams = new T[baseCount];
        for (int i = 0; i < baseCount; i++)
        {
            baseParams[i] = parameters[offset + i];
        }
        _baseUNet.SetParameters(new Vector<T>(baseParams));
        offset += baseCount;

        // Set all cached encoder parameters (in same order as GetParameters)
        foreach (var kvp in _encoderCache.OrderBy(kv => kv.Key))
        {
            var encoderCount = kvp.Value.ParameterCount;
            var encoderParams = new T[encoderCount];
            for (int i = 0; i < encoderCount; i++)
            {
                encoderParams[i] = parameters[offset + i];
            }
            kvp.Value.SetParameters(new Vector<T>(encoderParams));
            offset += encoderCount;
        }
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new ControlNetModel<T>(
            controlType: _controlType,
            conditioner: _conditioner,
            seed: RandomGenerator.Next());

        // Create matching encoder cache in clone before setting parameters
        foreach (var controlType in _encoderCache.Keys.Where(ct => ct != _controlType))
        {
            // GetOrCreateEncoder adds to the cache
            clone.GetOrCreateEncoder(controlType);
        }

        clone.SetParameters(GetParameters());
        clone.ConditioningStrength = _conditioningStrength;

        return clone;
    }
}

/// <summary>
/// Types of control signals supported by ControlNet.
/// </summary>
public enum ControlType
{
    /// <summary>Canny edge detection map.</summary>
    Canny,
    /// <summary>Depth map from MiDaS or similar.</summary>
    Depth,
    /// <summary>Surface normal map.</summary>
    Normal,
    /// <summary>OpenPose body keypoints.</summary>
    Pose,
    /// <summary>Semantic segmentation map.</summary>
    Segmentation,
    /// <summary>User-drawn scribbles.</summary>
    Scribble,
    /// <summary>Line art/sketch.</summary>
    LineArt,
    /// <summary>HED (Holistically-Nested Edge Detection).</summary>
    Hed,
    /// <summary>MLSD (Mobile Line Segment Detection).</summary>
    Mlsd,
    /// <summary>SoftEdge detection.</summary>
    SoftEdge,
    /// <summary>Shuffle/random structure.</summary>
    Shuffle,
    /// <summary>Inpaint mask.</summary>
    Inpaint
}

/// <summary>
/// ControlNet encoder that processes control signals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ControlNetEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _inputChannels;
    private readonly int _baseChannels;
    private readonly int[] _channelMultipliers;
    private readonly List<DenseLayer<T>> _downBlocks;
    private readonly List<DenseLayer<T>> _zeroConvs;
    private readonly int _imageSize;

    /// <summary>
    /// Gets the number of parameters in this encoder.
    /// </summary>
    public int ParameterCount { get; private set; }

    /// <summary>
    /// Initializes a new ControlNetEncoder.
    /// </summary>
    public ControlNetEncoder(
        int inputChannels,
        int baseChannels,
        int[] channelMultipliers,
        int imageSize = 64,
        int? seed = null)
    {
        _inputChannels = inputChannels;
        _baseChannels = baseChannels;
        _channelMultipliers = channelMultipliers;
        _imageSize = imageSize;

        _downBlocks = new List<DenseLayer<T>>();
        _zeroConvs = new List<DenseLayer<T>>();

        InitializeLayers(seed);
    }

    private void InitializeLayers(int? seed)
    {
        // Calculate sizes at each level (simulating stride-2 downsampling)
        var spatialSize = _imageSize;
        var inputDim = _inputChannels * spatialSize * spatialSize;
        var outputDim = _baseChannels * spatialSize * spatialSize;

        // Input projection
        var inProj = new DenseLayer<T>(inputDim, outputDim, (IActivationFunction<T>?)null);
        _downBlocks.Add(inProj);

        // Zero projection for input
        var zeroProj = new DenseLayer<T>(outputDim, outputDim, (IActivationFunction<T>?)null);
        _zeroConvs.Add(zeroProj);

        // Create down blocks with zero projections
        var prevDim = outputDim;
        foreach (var mult in _channelMultipliers)
        {
            // Downsample spatial dimension by 2
            spatialSize = Math.Max(1, spatialSize / 2);
            var channels = _baseChannels * mult;
            var newDim = channels * spatialSize * spatialSize;

            var downBlock = new DenseLayer<T>(prevDim, newDim, (IActivationFunction<T>?)null);
            _downBlocks.Add(downBlock);

            var zc = new DenseLayer<T>(newDim, newDim, (IActivationFunction<T>?)null);
            _zeroConvs.Add(zc);

            prevDim = newDim;
        }

        // Count parameters
        ParameterCount = 0;
        foreach (var block in _downBlocks)
        {
            ParameterCount += block.ParameterCount;
        }
        foreach (var zc in _zeroConvs)
        {
            ParameterCount += zc.ParameterCount;
        }
    }

    /// <summary>
    /// Encodes a control image into multi-scale features.
    /// </summary>
    public List<Tensor<T>> Encode(Tensor<T> controlImage)
    {
        var features = new List<Tensor<T>>();
        var x = controlImage;

        for (int i = 0; i < _downBlocks.Count; i++)
        {
            x = _downBlocks[i].Forward(x);
            var feat = _zeroConvs[i].Forward(x);
            features.Add(feat);
        }

        return features;
    }

    /// <summary>
    /// Gets all parameters as a vector.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var block in _downBlocks)
        {
            var p = block.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        foreach (var zc in _zeroConvs)
        {
            var p = zc.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Sets all parameters from a vector.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        foreach (var block in _downBlocks)
        {
            var count = block.ParameterCount;
            var p = new T[count];
            for (int i = 0; i < count; i++)
            {
                p[i] = parameters[offset + i];
            }
            block.SetParameters(new Vector<T>(p));
            offset += count;
        }

        foreach (var zc in _zeroConvs)
        {
            var count = zc.ParameterCount;
            var p = new T[count];
            for (int i = 0; i < count; i++)
            {
                p[i] = parameters[offset + i];
            }
            zc.SetParameters(new Vector<T>(p));
            offset += count;
        }
    }
}
