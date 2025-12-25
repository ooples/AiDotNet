using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// VideoCrafter model for high-quality text-to-video and image-to-video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VideoCrafter is a video generation model that combines the strengths of text-to-video
/// and image-to-video generation. It uses a dual-conditioning approach that enables
/// both modalities while maintaining high visual quality and temporal coherence.
/// </para>
/// <para>
/// <b>For Beginners:</b> VideoCrafter is like having two video generation modes in one:
///
/// Mode 1 - Text-to-Video:
/// - Input: "A rocket launching into space"
/// - Output: 5-second video of a rocket launch
///
/// Mode 2 - Image-to-Video:
/// - Input: Photo of a rocket on launch pad
/// - Output: 5-second video of the rocket launching
///
/// Key advantages:
/// - High visual quality (up to 1024x576 resolution)
/// - Long video generation (up to 16+ seconds)
/// - Good temporal coherence (smooth motion)
/// - Dual conditioning (text + image together)
///
/// Unlike AnimateDiff which adds motion to SD models, VideoCrafter is trained
/// end-to-end specifically for video generation, resulting in better quality.
/// </para>
/// <para>
/// Architecture:
/// - 3D U-Net with factorized spatial-temporal attention
/// - Dual cross-attention for text and image conditioning
/// - Temporal VAE for consistent frame encoding
/// - DDIM scheduler for fast inference
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create VideoCrafter model
/// var videoCrafter = new VideoCrafterModel&lt;float&gt;();
///
/// // Text-to-video generation
/// var video = videoCrafter.GenerateFromText(
///     prompt: "A beautiful sunset over the ocean, waves crashing",
///     width: 1024,
///     height: 576,
///     numFrames: 16,
///     numInferenceSteps: 50);
///
/// // Image-to-video with text guidance
/// var inputImage = LoadImage("sunset.jpg");
/// var animatedVideo = videoCrafter.GenerateFromImageAndText(
///     image: inputImage,
///     prompt: "waves gently rolling, seagulls flying",
///     numFrames: 16);
/// </code>
/// </example>
public class VideoCrafterModel<T> : VideoDiffusionModelBase<T>
{
    /// <summary>
    /// Default VideoCrafter width.
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default VideoCrafter height.
    /// </summary>
    public const int DefaultHeight = 576;

    /// <summary>
    /// VideoCrafter latent channels.
    /// </summary>
    private const int LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard latent scale factor.
    /// </summary>
    private const double LATENT_SCALE = 0.18215;

    /// <summary>
    /// The VideoUNet noise predictor with dual conditioning.
    /// </summary>
    private readonly VideoUNetPredictor<T> _videoUNet;

    /// <summary>
    /// The temporal VAE for video encoding/decoding.
    /// </summary>
    private readonly TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// The text conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _textConditioner;

    /// <summary>
    /// The image conditioning module.
    /// </summary>
    private readonly IConditioningModule<T>? _imageConditioner;

    /// <summary>
    /// Gets the noise predictor.
    /// </summary>
    public override INoisePredictor<T> NoisePredictor => _videoUNet;

    /// <summary>
    /// Gets the VAE.
    /// </summary>
    public override IVAEModel<T> VAE => _temporalVAE;

    /// <summary>
    /// Gets the temporal VAE.
    /// </summary>
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;

    /// <summary>
    /// Gets the primary conditioning module (text).
    /// </summary>
    public override IConditioningModule<T>? Conditioner => _textConditioner;

    /// <summary>
    /// Gets the image conditioning module.
    /// </summary>
    public IConditioningModule<T>? ImageConditioner => _imageConditioner;

    /// <summary>
    /// Gets the latent channels.
    /// </summary>
    public override int LatentChannels => LATENT_CHANNELS;

    /// <summary>
    /// Gets whether image-to-video is supported.
    /// </summary>
    public override bool SupportsImageToVideo => true;

    /// <summary>
    /// Gets whether text-to-video is supported.
    /// </summary>
    public override bool SupportsTextToVideo => _textConditioner != null;

    /// <summary>
    /// Gets whether video-to-video is supported.
    /// </summary>
    public override bool SupportsVideoToVideo => _textConditioner != null;

    /// <summary>
    /// Gets or sets the image conditioning scale.
    /// </summary>
    /// <remarks>
    /// Controls how strongly the input image influences the output video.
    /// Higher values keep the video closer to the input image.
    /// </remarks>
    public double ImageConditioningScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use dual conditioning (text + image together).
    /// </summary>
    public bool UseDualConditioning { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of VideoCrafterModel with default parameters.
    /// </summary>
    public VideoCrafterModel()
        : this(
            options: null,
            scheduler: null,
            videoUNet: null,
            temporalVAE: null,
            textConditioner: null,
            imageConditioner: null,
            defaultNumFrames: 16,
            defaultFPS: 8)
    {
    }

    /// <summary>
    /// Initializes a new instance of VideoCrafterModel with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="scheduler">Optional scheduler.</param>
    /// <param name="videoUNet">Optional VideoUNet predictor.</param>
    /// <param name="temporalVAE">Optional temporal VAE.</param>
    /// <param name="textConditioner">Optional text conditioning module.</param>
    /// <param name="imageConditioner">Optional image conditioning module.</param>
    /// <param name="defaultNumFrames">Default number of frames.</param>
    /// <param name="defaultFPS">Default FPS.</param>
    public VideoCrafterModel(
        DiffusionModelOptions<T>? options = null,
        IStepScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? textConditioner = null,
        IConditioningModule<T>? imageConditioner = null,
        int defaultNumFrames = 16,
        int defaultFPS = 8)
        : base(options, scheduler ?? CreateDefaultScheduler(), defaultNumFrames, defaultFPS)
    {
        _videoUNet = videoUNet ?? CreateDefaultVideoUNet();
        _temporalVAE = temporalVAE ?? CreateDefaultTemporalVAE();
        _textConditioner = textConditioner;
        _imageConditioner = imageConditioner;
    }

    /// <summary>
    /// Creates the default DDIM scheduler.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default VideoUNet.
    /// </summary>
    private VideoUNetPredictor<T> CreateDefaultVideoUNet()
    {
        return new VideoUNetPredictor<T>(
            inputChannels: LATENT_CHANNELS,
            outputChannels: LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 1024,
            numHeads: 8,
            numTemporalLayers: 2,
            supportsImageConditioning: true);
    }

    /// <summary>
    /// Creates the default TemporalVAE.
    /// </summary>
    private TemporalVAE<T> CreateDefaultTemporalVAE()
    {
        return new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numTemporalLayers: 2,
            temporalKernelSize: 3,
            causalMode: false,
            latentScaleFactor: LATENT_SCALE);
    }

    /// <summary>
    /// Generates video from text prompt.
    /// </summary>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 1024,
        int height = 576,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        int? seed = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Text-to-video requires a text conditioning module.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = _textConditioner.Tokenize(prompt);
        var promptEmbedding = _textConditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _textConditioner.Tokenize(negativePrompt);
                negativeEmbedding = _textConditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _textConditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / _temporalVAE.DownsampleFactor;
        var latentWidth = width / _temporalVAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = PredictVideoNoiseWithText(latents, timestep, promptEmbedding);
                var uncondPred = PredictVideoNoiseWithText(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidanceVideo(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = PredictVideoNoiseWithText(latents, timestep, promptEmbedding);
            }

            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Generates video from image with optional text guidance.
    /// </summary>
    public override Tensor<T> GenerateFromImage(
        Tensor<T> inputImage,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 25,
        int? motionBucketId = null,
        double noiseAugStrength = 0.02,
        int? seed = null)
    {
        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var effectiveMotionBucket = motionBucketId ?? MotionBucketId;

        var imageShape = inputImage.Shape;
        var height = imageShape[2];
        var width = imageShape[3];

        // Encode conditioning image
        var imageEmbedding = EncodeConditioningImage(inputImage, noiseAugStrength, seed);

        // Create motion embedding
        var motionEmbedding = CreateMotionEmbedding(effectiveMotionBucket, fps ?? DefaultFPS);

        // Calculate video latent dimensions
        var latentHeight = height / _temporalVAE.DownsampleFactor;
        var latentWidth = width / _temporalVAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = PredictVideoNoise(latents, timestep, imageEmbedding, motionEmbedding);
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Generates video with dual conditioning (image + text).
    /// </summary>
    /// <param name="image">The conditioning image.</param>
    /// <param name="prompt">The text prompt for guidance.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Text guidance scale.</param>
    /// <param name="imageScale">Image conditioning scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method combines the best of both worlds:
    /// - The image provides the visual style and starting point
    /// - The text describes what motion/action should happen
    ///
    /// Example:
    /// - Image: Photo of a person standing
    /// - Prompt: "person starts dancing energetically"
    /// - Result: Video of that person dancing
    /// </para>
    /// </remarks>
    public Tensor<T> GenerateFromImageAndText(
        Tensor<T> image,
        string prompt,
        string? negativePrompt = null,
        int? numFrames = null,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        double imageScale = 1.0,
        int? seed = null)
    {
        if (_textConditioner == null)
            throw new InvalidOperationException("Dual conditioning requires a text conditioning module.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var useCFG = guidanceScale > 1.0;

        var imageShape = image.Shape;
        var height = imageShape[2];
        var width = imageShape[3];

        // Encode image conditioning
        var imageEmbedding = EncodeConditioningImage(image, 0.02, seed);

        // Encode text conditioning
        var promptTokens = _textConditioner.Tokenize(prompt);
        var promptEmbedding = _textConditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _textConditioner.Tokenize(negativePrompt);
                negativeEmbedding = _textConditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _textConditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / _temporalVAE.DownsampleFactor;
        var latentWidth = width / _temporalVAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                // Dual conditioning: combine image and text embeddings
                var condPred = PredictWithDualConditioning(
                    latents, timestep, imageEmbedding, promptEmbedding, imageScale);
                var uncondPred = PredictWithDualConditioning(
                    latents, timestep, imageEmbedding, negativeEmbedding, imageScale);
                noisePrediction = ApplyGuidanceVideo(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = PredictWithDualConditioning(
                    latents, timestep, imageEmbedding, promptEmbedding, imageScale);
            }

            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Predicts noise with dual conditioning.
    /// </summary>
    private Tensor<T> PredictWithDualConditioning(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> textEmbedding,
        double imageScale)
    {
        // VideoCrafter uses factorized cross-attention
        // Image provides spatial context, text provides semantic guidance
        // Here we combine them by weighted concatenation in the context dimension

        if (UseDualConditioning && imageScale > 0)
        {
            // Combine image and text conditioning
            var combinedConditioning = CombineConditionings(imageEmbedding, textEmbedding, imageScale);
            return _videoUNet.PredictNoiseWithImageCondition(latents, timestep, combinedConditioning, textEmbedding);
        }
        else
        {
            // Text-only conditioning
            return PredictVideoNoiseWithText(latents, timestep, textEmbedding);
        }
    }

    /// <summary>
    /// Combines image and text conditionings.
    /// </summary>
    private Tensor<T> CombineConditionings(
        Tensor<T> imageEmbedding,
        Tensor<T> textEmbedding,
        double imageScale)
    {
        // Weighted combination of embeddings
        // In practice, this would use cross-attention or adapter modules
        var imageScaleT = NumOps.FromDouble(imageScale);
        var textScaleT = NumOps.FromDouble(1.0);

        var imageSpan = imageEmbedding.AsSpan();
        var textSpan = textEmbedding.AsSpan();

        // Create combined embedding (simplified - use larger of two shapes)
        var combinedShape = imageEmbedding.Shape.Length >= textEmbedding.Shape.Length
            ? imageEmbedding.Shape
            : textEmbedding.Shape;

        var combined = new Tensor<T>(combinedShape);
        var combinedSpan = combined.AsWritableSpan();

        var minLength = Math.Min(Math.Min(imageSpan.Length, textSpan.Length), combinedSpan.Length);

        for (int i = 0; i < minLength; i++)
        {
            var imgVal = i < imageSpan.Length ? imageSpan[i] : NumOps.Zero;
            var txtVal = i < textSpan.Length ? textSpan[i] : NumOps.Zero;

            combinedSpan[i] = NumOps.Add(
                NumOps.Multiply(imageScaleT, imgVal),
                NumOps.Multiply(textScaleT, txtVal));
        }

        return combined;
    }

    /// <summary>
    /// Predicts video noise with text conditioning.
    /// </summary>
    protected override Tensor<T> PredictVideoNoiseWithText(
        Tensor<T> latents,
        int timestep,
        Tensor<T> textEmbedding)
    {
        var videoShape = latents.Shape;
        var numFrames = videoShape[1];
        var result = new Tensor<T>(videoShape);

        // Use VideoUNet for temporal-aware prediction
        for (int f = 0; f < numFrames; f++)
        {
            var frameLatent = ExtractFrameLatent(latents, f);
            var frameNoise = _videoUNet.PredictNoise(frameLatent, timestep, textEmbedding);
            InsertFrameLatent(result, frameNoise, f);
        }

        return result;
    }

    /// <summary>
    /// Predicts video noise for image-to-video.
    /// </summary>
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        return _videoUNet.PredictNoiseWithImageCondition(
            latents,
            timestep,
            imageEmbedding,
            textConditioning: null);
    }

    /// <summary>
    /// Decodes video latents using temporal VAE.
    /// </summary>
    protected override Tensor<T> DecodeVideoLatents(Tensor<T> latents)
    {
        return _temporalVAE.Decode(latents);
    }

    #region IParameterizable Implementation

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public override int ParameterCount
    {
        get
        {
            var count = 0;
            var unetParams = _videoUNet.GetParameters();
            if (unetParams != null)
            {
                count += unetParams.Length;
            }

            var vaeParams = _temporalVAE.GetParameters();
            if (vaeParams != null)
            {
                count += vaeParams.Length;
            }

            return count;
        }
    }

    /// <summary>
    /// Gets all parameters.
    /// </summary>
    public override Vector<T> GetParameters()
    {
        var unetParams = _videoUNet.GetParameters();
        var vaeParams = _temporalVAE.GetParameters();

        var totalLength = (unetParams?.Length ?? 0) + (vaeParams?.Length ?? 0);
        var combined = new Vector<T>(totalLength);
        var offset = 0;

        if (unetParams != null)
        {
            for (int i = 0; i < unetParams.Length; i++)
            {
                combined[offset + i] = unetParams[i];
            }
            offset += unetParams.Length;
        }

        if (vaeParams != null)
        {
            for (int i = 0; i < vaeParams.Length; i++)
            {
                combined[offset + i] = vaeParams[i];
            }
        }

        return combined;
    }

    /// <summary>
    /// Sets all parameters.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        var unetParamCount = _videoUNet.GetParameters()?.Length ?? 0;
        var vaeParamCount = _temporalVAE.GetParameters()?.Length ?? 0;

        if (parameters.Length != unetParamCount + vaeParamCount)
        {
            throw new ArgumentException(
                $"Expected {unetParamCount + vaeParamCount} parameters but got {parameters.Length}.",
                nameof(parameters));
        }

        var offset = 0;

        if (unetParamCount > 0)
        {
            var unetParams = new Vector<T>(unetParamCount);
            for (int i = 0; i < unetParamCount; i++)
            {
                unetParams[i] = parameters[offset + i];
            }
            _videoUNet.SetParameters(unetParams);
            offset += unetParamCount;
        }

        if (vaeParamCount > 0)
        {
            var vaeParams = new Vector<T>(vaeParamCount);
            for (int i = 0; i < vaeParamCount; i++)
            {
                vaeParams[i] = parameters[offset + i];
            }
            _temporalVAE.SetParameters(vaeParams);
        }
    }

    #endregion

    #region ICloneable Implementation

    /// <summary>
    /// Clones this model.
    /// </summary>
    public override IDiffusionModel<T> Clone()
    {
        var clone = new VideoCrafterModel<T>(
            options: null,
            scheduler: null,
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            textConditioner: _textConditioner,
            imageConditioner: _imageConditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);

        clone.ImageConditioningScale = ImageConditioningScale;
        clone.UseDualConditioning = UseDualConditioning;
        clone.SetMotionBucketId(MotionBucketId);

        return clone;
    }

    /// <summary>
    /// Creates a deep copy.
    /// </summary>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return (IFullModel<T, Tensor<T>, Tensor<T>>)Clone();
    }

    #endregion
}
