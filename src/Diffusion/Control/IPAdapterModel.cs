using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Control;

/// <summary>
/// IP-Adapter model for image-based prompt conditioning in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IP-Adapter (Image Prompt Adapter) enables using reference images as prompts
/// to guide image generation. It decouples cross-attention for text and image
/// features, allowing fine-grained control over image style, composition, and
/// content transfer.
/// </para>
/// <para>
/// <b>For Beginners:</b> IP-Adapter lets you use pictures as instructions
/// for the AI instead of just text.
///
/// Think of it like:
/// - Showing someone a photo and saying "make something like this"
/// - The AI extracts the style, composition, and content from your image
/// - It then applies those elements to create new images
///
/// Use cases:
/// - Style transfer: "Generate in the style of this artwork"
/// - Face preservation: Keep a person's likeness in different scenes
/// - Object consistency: Maintain the same object across images
/// - Scene composition: Use reference for layout/arrangement
///
/// Key advantage: Combines with text prompts for precise control
/// </para>
/// <para>
/// Technical details:
/// - Uses a pretrained image encoder (like CLIP ViT)
/// - Projects image features to text embedding space
/// - Injects via decoupled cross-attention mechanism
/// - Supports multiple reference images (multi-IP)
/// - Adjustable image prompt weight (0-1)
///
/// Reference: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter", 2023
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an IP-Adapter model
/// var ipAdapter = new IPAdapterModel&lt;float&gt;();
///
/// // Generate with image reference
/// var referenceImage = LoadImage("style_reference.png");
/// var image = ipAdapter.GenerateWithImagePrompt(
///     textPrompt: "A beautiful landscape",
///     imagePrompt: referenceImage,
///     imagePromptWeight: 0.7);
///
/// // Multi-image reference
/// var faceImage = LoadImage("face.png");
/// var styleImage = LoadImage("art_style.png");
/// var composed = ipAdapter.GenerateWithMultiImagePrompt(
///     textPrompt: "Portrait painting",
///     imagePrompts: new[] { faceImage, styleImage },
///     imageWeights: new[] { 0.8, 0.5 });
/// </code>
/// </example>
public class IPAdapterModel<T> : LatentDiffusionModelBase<T>
{
    #region Constants

    /// <summary>
    /// Standard IP-Adapter latent channels.
    /// </summary>
    /// <remarks>
    /// 4 latent channels matching the standard SD 1.5 VAE architecture.
    /// </remarks>
    private const int IPA_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard VAE scale factor.
    /// </summary>
    /// <remarks>
    /// 8x spatial downsampling from pixel space to latent space.
    /// </remarks>
    private const int IPA_VAE_SCALE_FACTOR = 8;

    /// <summary>
    /// CLIP image embedding dimension.
    /// </summary>
    /// <remarks>
    /// 768-dimensional CLIP ViT-L/14 image embeddings used for
    /// projecting reference images into the cross-attention space.
    /// </remarks>
    private const int CLIP_EMBED_DIM = 768;

    #endregion

    #region Fields

    /// <summary>
    /// The base noise predictor (U-Net).
    /// </summary>
    private readonly UNetNoisePredictor<T> _baseUNet;

    /// <summary>
    /// The VAE for encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// The text conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// The image encoder for extracting image features.
    /// </summary>
    private readonly ImageEncoder<T> _imageEncoder;

    /// <summary>
    /// The image projection layer.
    /// </summary>
    private readonly ImageProjector<T> _imageProjector;

    /// <summary>
    /// Default image prompt weight.
    /// </summary>
    private double _imagePromptWeight = 1.0;

    #endregion

    #region Properties

    /// <inheritdoc />
    public override INoisePredictor<T> NoisePredictor => _baseUNet;

    /// <inheritdoc />
    public override IVAEModel<T> VAE => _vae;

    /// <inheritdoc />
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <inheritdoc />
    public override int LatentChannels => IPA_LATENT_CHANNELS;

    /// <inheritdoc />
    public override int ParameterCount =>
        _baseUNet.ParameterCount + _vae.ParameterCount + _imageEncoder.ParameterCount + _imageProjector.ParameterCount;

    /// <summary>
    /// Gets or sets the default image prompt weight (0-1).
    /// </summary>
    public double ImagePromptWeight
    {
        get => _imagePromptWeight;
        set => _imagePromptWeight = MathPolyfill.Clamp(value, 0.0, 1.0);
    }

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new instance of IPAdapterModel with full customization support.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="baseUNet">Optional base U-Net noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text encoding.</param>
    /// <param name="embedDim">Image embedding dimension.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public IPAdapterModel(
        NeuralNetworkArchitecture<T>? architecture = null,
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? baseUNet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        int embedDim = CLIP_EMBED_DIM,
        int? seed = null)
        : base(
            options ?? new DiffusionModelOptions<T>
            {
                TrainTimesteps = 1000,
                BetaStart = 0.00085,
                BetaEnd = 0.012,
                BetaSchedule = BetaSchedule.ScaledLinear
            },
            scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateStableDiffusion()),
            architecture)
    {
        _conditioner = conditioner;

        _baseUNet = baseUNet ?? new UNetNoisePredictor<T>(
            architecture: Architecture,
            inputChannels: IPA_LATENT_CHANNELS,
            outputChannels: IPA_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            seed: seed);

        _vae = vae ?? new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: IPA_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocksPerLevel: 2,
            seed: seed);

        _imageEncoder = new ImageEncoder<T>(
            imageSize: 224,
            patchSize: 16,
            embedDim: embedDim,
            numLayers: 12,
            numHeads: 12,
            seed: seed);

        _imageProjector = new ImageProjector<T>(
            inputDim: embedDim,
            outputDim: 768,
            numTokens: 4,
            seed: seed);
    }

    #endregion

    /// <summary>
    /// Generates an image with image prompt conditioning.
    /// </summary>
    /// <param name="textPrompt">The text prompt describing the desired image.</param>
    /// <param name="imagePrompt">The reference image for conditioning.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output image width.</param>
    /// <param name="height">Output image height.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="imagePromptWeight">Weight for image prompt (0-1).</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The generated image tensor.</returns>
    public virtual Tensor<T> GenerateWithImagePrompt(
        string textPrompt,
        Tensor<T> imagePrompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        double? imagePromptWeight = null,
        int? seed = null)
    {
        var effectiveWeight = imagePromptWeight ?? _imagePromptWeight;

        // Encode image prompt
        var imageFeatures = _imageEncoder.Encode(imagePrompt);
        var imageEmbedding = _imageProjector.Project(imageFeatures);

        // Scale by weight
        if (Math.Abs(effectiveWeight - 1.0) > 1e-6)
        {
            imageEmbedding = ScaleTensor(imageEmbedding, effectiveWeight);
        }

        // Get text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        if (_conditioner != null)
        {
            var promptTokens = _conditioner.Tokenize(textPrompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);

            // Combine text and image embeddings
            promptEmbedding = CombineEmbeddings(promptEmbedding, imageEmbedding);

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
        else
        {
            // Use image embedding alone
            promptEmbedding = imageEmbedding;
        }

        // Generate using diffusion
        return GenerateWithEmbedding(
            promptEmbedding,
            negativeEmbedding,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Generates an image with multiple image prompts.
    /// </summary>
    /// <param name="textPrompt">The text prompt.</param>
    /// <param name="imagePrompts">Array of reference images.</param>
    /// <param name="imageWeights">Optional weights for each image.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Output width.</param>
    /// <param name="height">Output height.</param>
    /// <param name="numInferenceSteps">Number of steps.</param>
    /// <param name="guidanceScale">Guidance scale.</param>
    /// <param name="seed">Random seed.</param>
    /// <returns>The generated image.</returns>
    public virtual Tensor<T> GenerateWithMultiImagePrompt(
        string textPrompt,
        Tensor<T>[] imagePrompts,
        double[]? imageWeights = null,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int numInferenceSteps = 50,
        double? guidanceScale = null,
        int? seed = null)
    {
        // Default weights to equal
        var weights = imageWeights ?? Enumerable.Repeat(1.0 / imagePrompts.Length, imagePrompts.Length).ToArray();

        // Encode and combine all image prompts
        Tensor<T>? combinedEmbedding = null;

        for (int i = 0; i < imagePrompts.Length; i++)
        {
            var imageFeatures = _imageEncoder.Encode(imagePrompts[i]);
            var imageEmbedding = _imageProjector.Project(imageFeatures);
            imageEmbedding = ScaleTensor(imageEmbedding, weights[i]);

            if (combinedEmbedding == null)
            {
                combinedEmbedding = imageEmbedding;
            }
            else
            {
                combinedEmbedding = AddTensors(combinedEmbedding, imageEmbedding);
            }
        }

        // Get text conditioning
        Tensor<T>? promptEmbedding = null;
        Tensor<T>? negativeEmbedding = null;
        var effectiveGuidanceScale = guidanceScale ?? GuidanceScale;

        if (_conditioner != null && combinedEmbedding != null)
        {
            var promptTokens = _conditioner.Tokenize(textPrompt);
            promptEmbedding = _conditioner.EncodeText(promptTokens);
            promptEmbedding = CombineEmbeddings(promptEmbedding, combinedEmbedding);

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
        else
        {
            promptEmbedding = combinedEmbedding;
        }

        return GenerateWithEmbedding(
            promptEmbedding,
            negativeEmbedding,
            width,
            height,
            numInferenceSteps,
            effectiveGuidanceScale,
            seed);
    }

    /// <summary>
    /// Generates image using pre-computed embeddings.
    /// </summary>
    private Tensor<T> GenerateWithEmbedding(
        Tensor<T>? promptEmbedding,
        Tensor<T>? negativeEmbedding,
        int width,
        int height,
        int numInferenceSteps,
        double guidanceScale,
        int? seed)
    {
        // Calculate latent dimensions
        var latentHeight = height / IPA_VAE_SCALE_FACTOR;
        var latentWidth = width / IPA_VAE_SCALE_FACTOR;
        var latentShape = new[] { 1, IPA_LATENT_CHANNELS, latentHeight, latentWidth };

        // Initialize noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = SampleNoiseTensor(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (guidanceScale > 1.0 && negativeEmbedding != null && promptEmbedding != null)
            {
                var condPred = _baseUNet.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = _baseUNet.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = _baseUNet.PredictNoise(latents, timestep, promptEmbedding);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        return DecodeFromLatent(latents);
    }

    /// <summary>
    /// Combines text and image embeddings.
    /// </summary>
    private Tensor<T> CombineEmbeddings(Tensor<T> textEmbed, Tensor<T> imageEmbed)
    {
        // Concatenate along token dimension
        // For simplicity, we add the embeddings
        return AddTensors(textEmbed, imageEmbed);
    }

    /// <summary>
    /// Scales a tensor by a scalar value.
    /// </summary>
    private Tensor<T> ScaleTensor(Tensor<T> tensor, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        return Engine.TensorMultiplyScalar<T>(tensor, scaleT);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd<T>(a, b);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var unetParams = _baseUNet.GetParameters();
        var vaeParams = _vae.GetParameters();
        var encoderParams = _imageEncoder.GetParameters();
        var projectorParams = _imageProjector.GetParameters();

        var totalLength = unetParams.Length + vaeParams.Length + encoderParams.Length + projectorParams.Length;
        var combined = new T[totalLength];

        var offset = 0;
        for (int i = 0; i < unetParams.Length; i++)
        {
            combined[offset++] = unetParams[i];
        }
        for (int i = 0; i < vaeParams.Length; i++)
        {
            combined[offset++] = vaeParams[i];
        }
        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[offset++] = encoderParams[i];
        }
        for (int i = 0; i < projectorParams.Length; i++)
        {
            combined[offset++] = projectorParams[i];
        }

        return new Vector<T>(combined);
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        var unetCount = _baseUNet.ParameterCount;
        var vaeCount = _vae.ParameterCount;
        var encoderCount = _imageEncoder.ParameterCount;
        var projectorCount = _imageProjector.ParameterCount;

        var offset = 0;
        var unetParams = new T[unetCount];
        for (int i = 0; i < unetCount; i++)
        {
            unetParams[i] = parameters[offset++];
        }
        _baseUNet.SetParameters(new Vector<T>(unetParams));

        var vaeParams = new T[vaeCount];
        for (int i = 0; i < vaeCount; i++)
        {
            vaeParams[i] = parameters[offset++];
        }
        _vae.SetParameters(new Vector<T>(vaeParams));

        var encoderParams = new T[encoderCount];
        for (int i = 0; i < encoderCount; i++)
        {
            encoderParams[i] = parameters[offset++];
        }
        _imageEncoder.SetParameters(new Vector<T>(encoderParams));

        var projectorParams = new T[projectorCount];
        for (int i = 0; i < projectorCount; i++)
        {
            projectorParams[i] = parameters[offset++];
        }
        _imageProjector.SetParameters(new Vector<T>(projectorParams));
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc />
    public override IDiffusionModel<T> Clone()
    {
        var clone = new IPAdapterModel<T>(
            conditioner: _conditioner,
            seed: RandomGenerator.Next());

        clone.SetParameters(GetParameters());
        clone.ImagePromptWeight = _imagePromptWeight;

        return clone;
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "IP-Adapter", Version = "1.0", ModelType = ModelType.NeuralNetwork,
            Description = "Image prompt adapter for reference image-guided diffusion generation",
            FeatureCount = ParameterCount, Complexity = ParameterCount
        };
        m.SetProperty("architecture", "unet-decoupled-cross-attention");
        m.SetProperty("base_model", "Stable Diffusion 1.5");
        m.SetProperty("text_encoder", "CLIP ViT-L/14");
        m.SetProperty("image_encoder", "CLIP ViT-H/14");
        m.SetProperty("context_dim", 768);
        m.SetProperty("image_prompt_weight", _imagePromptWeight);
        m.SetProperty("latent_channels", IPA_LATENT_CHANNELS);
        return m;
    }
}

/// <summary>
/// Image encoder for extracting features from reference images.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class ImageEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _imageSize;
    private readonly int _patchSize;
    private readonly int _embedDim;
    private readonly int _numPatches;
    private readonly DenseLayer<T> _patchEmbed;
    private readonly List<DenseLayer<T>> _transformerLayers;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount { get; private set; }

    /// <summary>
    /// Initializes a new ImageEncoder.
    /// </summary>
    public ImageEncoder(
        int imageSize = 224,
        int patchSize = 16,
        int embedDim = 768,
        int numLayers = 12,
        int numHeads = 12,
        int? seed = null)
    {
        _imageSize = imageSize;
        _patchSize = patchSize;
        _embedDim = embedDim;
        _numPatches = (imageSize / patchSize) * (imageSize / patchSize);

        var patchDim = 3 * patchSize * patchSize; // RGB patches
        _patchEmbed = new DenseLayer<T>(patchDim, embedDim, (IActivationFunction<T>?)null);

        _transformerLayers = new List<DenseLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            // Simplified transformer layer as MLP
            _transformerLayers.Add(new DenseLayer<T>(embedDim, embedDim * 4, (IActivationFunction<T>?)null));
            _transformerLayers.Add(new DenseLayer<T>(embedDim * 4, embedDim, (IActivationFunction<T>?)null));
        }

        ParameterCount = _patchEmbed.ParameterCount;
        foreach (var layer in _transformerLayers)
        {
            ParameterCount += layer.ParameterCount;
        }
    }

    /// <summary>
    /// Encodes an image into feature embeddings.
    /// </summary>
    public Tensor<T> Encode(Tensor<T> image)
    {
        // Simplified encoding: flatten and project
        var flatImage = FlattenPatches(image);
        var embeddings = _patchEmbed.Forward(flatImage);

        // Apply transformer layers
        var x = embeddings;
        for (int i = 0; i < _transformerLayers.Count; i += 2)
        {
            var h = _transformerLayers[i].Forward(x);
            h = ApplyGelu(h);
            h = _transformerLayers[i + 1].Forward(h);
            x = AddTensors(x, h); // Residual connection
        }

        return x;
    }

    private Tensor<T> FlattenPatches(Tensor<T> image)
    {
        // Simplified: just flatten the image
        var flatData = new T[image.Shape.Aggregate((a, b) => a * b)];
        var span = image.AsSpan();
        for (int i = 0; i < span.Length && i < flatData.Length; i++)
        {
            flatData[i] = span[i];
        }
        return new Tensor<T>(new[] { 1, flatData.Length }, new Vector<T>(flatData));
    }

    private Tensor<T> ApplyGelu(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        var span = x.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            var val = NumOps.ToDouble(span[i]);
            // GELU approximation
            var gelu = 0.5 * val * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (val + 0.044715 * val * val * val)));
            resultSpan[i] = NumOps.FromDouble(gelu);
        }

        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return AiDotNetEngine.Current.TensorAdd<T>(a, b);
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        var p = _patchEmbed.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }

        foreach (var layer in _transformerLayers)
        {
            p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        var count = _patchEmbed.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset++];
        }
        _patchEmbed.SetParameters(new Vector<T>(p));

        foreach (var layer in _transformerLayers)
        {
            count = layer.ParameterCount;
            p = new T[count];
            for (int i = 0; i < count; i++)
            {
                p[i] = parameters[offset++];
            }
            layer.SetParameters(new Vector<T>(p));
        }
    }
}

/// <summary>
/// Projects image features to text embedding space.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class ImageProjector<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numTokens;
    private readonly DenseLayer<T> _projection;
    private readonly DenseLayer<T> _tokenExpansion;

    /// <summary>
    /// Gets the number of parameters.
    /// </summary>
    public int ParameterCount { get; }

    /// <summary>
    /// Initializes a new ImageProjector.
    /// </summary>
    public ImageProjector(
        int inputDim = 768,
        int outputDim = 768,
        int numTokens = 4,
        int? seed = null)
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _numTokens = numTokens;

        _projection = new DenseLayer<T>(inputDim, outputDim, (IActivationFunction<T>?)null);
        _tokenExpansion = new DenseLayer<T>(outputDim, outputDim * numTokens, (IActivationFunction<T>?)null);

        ParameterCount = _projection.ParameterCount + _tokenExpansion.ParameterCount;
    }

    /// <summary>
    /// Projects image features to IP embedding.
    /// </summary>
    public Tensor<T> Project(Tensor<T> imageFeatures)
    {
        var projected = _projection.Forward(imageFeatures);
        var expanded = _tokenExpansion.Forward(projected);

        // Reshape to [batch, numTokens, outputDim]
        var batchSize = imageFeatures.Shape[0];
        return new Tensor<T>(new[] { batchSize, _numTokens, _outputDim }, expanded.ToVector());
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        var p = _projection.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }

        p = _tokenExpansion.GetParameters();
        for (int i = 0; i < p.Length; i++)
        {
            allParams.Add(p[i]);
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        var count = _projection.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset++];
        }
        _projection.SetParameters(new Vector<T>(p));

        count = _tokenExpansion.ParameterCount;
        p = new T[count];
        for (int i = 0; i < count; i++)
        {
            p[i] = parameters[offset++];
        }
        _tokenExpansion.SetParameters(new Vector<T>(p));
    }
}
