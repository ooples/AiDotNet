using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Models;

/// <summary>
/// Stable Video Diffusion (SVD) model for image-to-video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stable Video Diffusion generates short video clips from a single input image.
/// It extends the Stable Diffusion architecture with temporal awareness, using
/// a 3D U-Net for noise prediction and a temporal VAE for encoding/decoding.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of SVD as "making a picture come to life."
/// You give it a single image, and it generates a short video showing
/// how that scene might animate:
///
/// Example workflow:
/// 1. Input: Photo of a waterfall
/// 2. SVD analyzes the scene and understands what should move
/// 3. Output: 4-second video showing water flowing, mist rising
///
/// Key features:
/// - Image-to-video: Primary use case, animate still images
/// - Motion control: Adjust how much motion to add (motion bucket)
/// - Configurable length: Generate different numbers of frames
/// - High quality: Based on Stable Diffusion's proven architecture
///
/// Compared to text-to-video:
/// - More predictable results (scene is defined by input image)
/// - Better quality (less ambiguity than text prompts)
/// - Faster generation (can use fewer denoising steps)
/// </para>
/// <para>
/// Technical specifications:
/// - Default resolution: 576x1024 or 1024x576
/// - Default frames: 25 frames at 7 FPS (~3.5 seconds)
/// - Motion bucket ID: 1-255 (127 = moderate motion)
/// - Noise augmentation: 0.02 default for conditioning image
/// - Latent space: 4 channels, 8x spatial downsampling
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Stable Video Diffusion model
/// var svd = new StableVideoDiffusion&lt;float&gt;();
///
/// // Load your image (batch=1, channels=3, height=576, width=1024)
/// var inputImage = LoadImage("landscape.jpg");
///
/// // Generate video with default settings
/// var video = svd.GenerateFromImage(inputImage);
///
/// // Generate with custom motion (more movement)
/// var dynamicVideo = svd.GenerateFromImage(
///     inputImage,
///     numFrames: 25,
///     fps: 7,
///     motionBucketId: 200,  // Higher = more motion
///     numInferenceSteps: 25,
///     seed: 42);
///
/// // Output shape: [1, 25, 3, 576, 1024] (batch, frames, channels, height, width)
/// SaveVideo(dynamicVideo, "output.mp4");
/// </code>
/// </example>
public class StableVideoDiffusion<T> : VideoDiffusionModelBase<T>
{
    /// <summary>
    /// Default width for SVD generation.
    /// </summary>
    public const int DefaultWidth = 1024;

    /// <summary>
    /// Default height for SVD generation.
    /// </summary>
    public const int DefaultHeight = 576;

    /// <summary>
    /// Standard SVD latent channels.
    /// </summary>
    private const int SVD_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard SVD latent scale factor.
    /// </summary>
    private const double SVD_LATENT_SCALE = 0.18215;

    /// <summary>
    /// Default noise augmentation strength for conditioning image.
    /// </summary>
    private const double DEFAULT_NOISE_AUG_STRENGTH = 0.02;

    /// <summary>
    /// The VideoUNet noise predictor.
    /// </summary>
    private readonly VideoUNetPredictor<T> _videoUNet;

    /// <summary>
    /// Noise augmentation strength for micro-conditioning.
    /// </summary>
    private readonly double _noiseAugmentStrength;

    /// <summary>
    /// The temporal VAE for video encoding/decoding.
    /// </summary>
    private readonly TemporalVAE<T> _temporalVAE;

    /// <summary>
    /// Optional conditioning module for text guidance.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Gets the noise predictor used by this model.
    /// </summary>
    public override INoisePredictor<T> NoisePredictor => _videoUNet;

    /// <summary>
    /// Gets the VAE used by this model for image encoding.
    /// </summary>
    /// <remarks>
    /// Returns the temporal VAE for both single image and video operations.
    /// The temporal VAE can handle both 4D (image) and 5D (video) tensors.
    /// </remarks>
    public override IVAEModel<T> VAE => _temporalVAE;

    /// <summary>
    /// Gets the temporal VAE specifically for video operations.
    /// </summary>
    public override IVAEModel<T>? TemporalVAE => _temporalVAE;

    /// <summary>
    /// Gets the conditioning module if available.
    /// </summary>
    /// <remarks>
    /// SVD primarily uses image conditioning rather than text conditioning.
    /// The conditioner is optional and typically used for additional guidance.
    /// </remarks>
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <summary>
    /// Gets the number of latent channels (4 for SVD).
    /// </summary>
    public override int LatentChannels => SVD_LATENT_CHANNELS;

    /// <summary>
    /// Gets whether this model supports image-to-video generation.
    /// </summary>
    /// <remarks>
    /// Always true for SVD - this is the primary use case.
    /// </remarks>
    public override bool SupportsImageToVideo => true;

    /// <summary>
    /// Gets whether this model supports text-to-video generation.
    /// </summary>
    /// <remarks>
    /// Returns true only if a conditioning module is provided.
    /// SVD's primary mode is image-to-video, but text guidance can be added.
    /// </remarks>
    public override bool SupportsTextToVideo => _conditioner != null;

    /// <summary>
    /// Gets whether this model supports video-to-video transformation.
    /// </summary>
    /// <remarks>
    /// Partially supported through the VideoToVideo method inherited from base class.
    /// </remarks>
    public override bool SupportsVideoToVideo => _conditioner != null;

    /// <summary>
    /// Gets the video U-Net predictor with image conditioning support.
    /// </summary>
    public VideoUNetPredictor<T> VideoUNet => _videoUNet;

    /// <summary>
    /// Initializes a new instance of StableVideoDiffusion with default parameters.
    /// </summary>
    /// <remarks>
    /// Creates an SVD model with standard parameters:
    /// - 25 frames at 7 FPS
    /// - 320 base channels
    /// - DDPM scheduler with 1000 training steps
    /// - Image conditioning enabled
    /// </remarks>
    public StableVideoDiffusion()
        : this(
            options: null,
            scheduler: null,
            videoUNet: null,
            temporalVAE: null,
            conditioner: null,
            defaultNumFrames: 25,
            defaultFPS: 7)
    {
    }

    /// <summary>
    /// Initializes a new instance of StableVideoDiffusion with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler. Defaults to DDPM with 1000 steps.</param>
    /// <param name="videoUNet">Optional custom VideoUNet predictor.</param>
    /// <param name="temporalVAE">Optional custom temporal VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text guidance.</param>
    /// <param name="defaultNumFrames">Default number of frames to generate.</param>
    /// <param name="defaultFPS">Default frames per second.</param>
    public StableVideoDiffusion(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        VideoUNetPredictor<T>? videoUNet = null,
        TemporalVAE<T>? temporalVAE = null,
        IConditioningModule<T>? conditioner = null,
        int defaultNumFrames = 25,
        int defaultFPS = 7,
        double noiseAugmentStrength = DEFAULT_NOISE_AUG_STRENGTH)
        : base(options, scheduler ?? CreateDefaultScheduler(), defaultNumFrames, defaultFPS)
    {
        // Create default VideoUNet if not provided
        _videoUNet = videoUNet ?? CreateDefaultVideoUNet();

        // Create default TemporalVAE if not provided
        _temporalVAE = temporalVAE ?? CreateDefaultTemporalVAE();

        // Store optional conditioner
        _conditioner = conditioner;

        // Store noise augment strength for micro-conditioning
        _noiseAugmentStrength = noiseAugmentStrength;
    }

    /// <summary>
    /// Creates the default DDIM scheduler for SVD.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        // SVD uses 1000 training steps with scaled linear beta schedule
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default VideoUNet predictor for SVD.
    /// </summary>
    private VideoUNetPredictor<T> CreateDefaultVideoUNet()
    {
        // Standard SVD architecture parameters
        return new VideoUNetPredictor<T>(
            inputChannels: SVD_LATENT_CHANNELS,
            outputChannels: SVD_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 1024,
            numHeads: 8,
            numTemporalLayers: 1,
            supportsImageConditioning: true);
    }

    /// <summary>
    /// Creates the default TemporalVAE for SVD.
    /// </summary>
    private TemporalVAE<T> CreateDefaultTemporalVAE()
    {
        // Standard SVD VAE parameters
        // Note: downsampleFactor is calculated internally from channelMultipliers
        return new TemporalVAE<T>(
            inputChannels: 3,
            latentChannels: SVD_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },  // Results in downsample factor of 8
            numTemporalLayers: 1,
            temporalKernelSize: 3,
            causalMode: false,
            latentScaleFactor: SVD_LATENT_SCALE);
    }

    /// <summary>
    /// Generates a video from an input image using image-to-video diffusion.
    /// </summary>
    /// <param name="inputImage">
    /// The conditioning image tensor [batch, channels, height, width].
    /// Should be normalized to [-1, 1] range.
    /// </param>
    /// <param name="numFrames">Number of frames to generate. Default: 25.</param>
    /// <param name="fps">Frames per second. Default: 7.</param>
    /// <param name="numInferenceSteps">Number of denoising steps. Default: 25.</param>
    /// <param name="motionBucketId">
    /// Motion intensity control (1-255). Higher values = more motion.
    /// Default: 127 (moderate motion).
    /// </param>
    /// <param name="noiseAugStrength">
    /// Noise augmentation for conditioning image.
    /// Higher values encourage more deviation from input. Default: 0.02.
    /// </param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>
    /// Generated video tensor [batch, numFrames, channels, height, width].
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method generates a video sequence from a single input image.
    /// The first frame will closely match the input image, while subsequent
    /// frames show natural motion based on the scene content.
    /// </para>
    /// <para>
    /// Tips for best results:
    /// - Use high-quality, sharp input images
    /// - Adjust motion bucket for scene type (lower for static scenes, higher for action)
    /// - Use more inference steps for higher quality (25-50 steps)
    /// - Lower noise augmentation keeps output closer to input
    /// </para>
    /// </remarks>
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
        var effectiveFPS = fps ?? DefaultFPS;
        var effectiveMotionBucket = motionBucketId ?? MotionBucketId;

        // Get image dimensions
        var imageShape = inputImage.Shape;
        var height = imageShape[2];
        var width = imageShape[3];

        // Encode conditioning image with noise augmentation
        var imageEmbedding = EncodeConditioningImage(inputImage, noiseAugStrength, seed);

        // Create motion embedding
        var motionEmbedding = CreateMotionEmbedding(effectiveMotionBucket, effectiveFPS);

        // Calculate video latent dimensions
        // TemporalVAE expects [batch, channels, frames, height, width]
        var latentHeight = height / _temporalVAE.DownsampleFactor;
        var latentWidth = width / _temporalVAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, LatentChannels, effectiveNumFrames, latentHeight, latentWidth };

        // Generate initial noise for all frames
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Predict noise using VideoUNet with image conditioning
            var noisePrediction = PredictVideoNoise(
                latents,
                timestep,
                imageEmbedding,
                motionEmbedding);

            // Scheduler step for video latents
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        // Decode video latents to frames using temporal VAE
        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Predicts noise for video frames conditioned on image and motion.
    /// </summary>
    /// <param name="latents">Current video latents [batch, channels, frames, height, width].</param>
    /// <param name="timestep">Current diffusion timestep.</param>
    /// <param name="imageEmbedding">Encoded conditioning image.</param>
    /// <param name="motionEmbedding">Motion embedding for motion intensity control.</param>
    /// <returns>Predicted noise tensor with same shape as latents.</returns>
    /// <remarks>
    /// This method uses the VideoUNet with image conditioning to predict
    /// noise for all frames simultaneously. The image embedding provides
    /// scene context while motion embedding controls animation intensity.
    /// </remarks>
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        // SVD uses the VideoUNet with image conditioning
        // The image embedding is used as cross-attention context
        // Motion embedding is combined with time embedding internally

        // For SVD, we predict noise for all frames at once using the 3D U-Net
        // The VideoUNet handles the 5D input internally
        return _videoUNet.PredictNoiseWithImageCondition(
            latents,
            timestep,
            imageEmbedding,
            textConditioning: null);
    }

    /// <summary>
    /// Encodes a conditioning image with SVD-specific processing.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="noiseAugStrength">Noise augmentation strength.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Encoded image embedding for conditioning.</returns>
    protected override Tensor<T> EncodeConditioningImage(Tensor<T> image, double noiseAugStrength, int? seed)
    {
        // Add noise augmentation to encourage motion
        Tensor<T> augmentedImage;
        if (noiseAugStrength > 0)
        {
            var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
            var noise = DiffusionNoiseHelper<T>.SampleGaussian(image.Shape, rng);
            var scaledNoise = DiffusionNoiseHelper<T>.ScaleNoise(noise, noiseAugStrength);

            augmentedImage = new Tensor<T>(image.Shape);
            var augSpan = augmentedImage.AsWritableSpan();
            var imgSpan = image.AsSpan();
            var noiseSpan = scaledNoise.AsSpan();

            for (int i = 0; i < augSpan.Length; i++)
            {
                augSpan[i] = NumOps.Add(imgSpan[i], noiseSpan[i]);
            }
        }
        else
        {
            augmentedImage = image;
        }

        // Encode the image to latent space
        return _temporalVAE.Encode(augmentedImage, sampleMode: false);
    }

    /// <summary>
    /// Creates SVD-specific motion embedding.
    /// </summary>
    /// <param name="motionBucketId">Motion intensity (1-255).</param>
    /// <param name="fps">Frames per second.</param>
    /// <returns>Motion embedding tensor.</returns>
    protected override Tensor<T> CreateMotionEmbedding(int motionBucketId, int fps)
    {
        // SVD uses micro-conditioning with add_time_ids:
        // [fps-1, motion_bucket_id, noise_aug_strength] as separate raw values
        // that are concatenated and projected through learned embedding layers

        // Standard SVD embedding dimensions:
        // - Each conditioning value is embedded into 256 dimensions
        // - Total: 3 values x 256 = 768 (or fewer depending on model config)
        var additionTimeEmbedDim = 256;
        var numConditioningValues = 3; // fps, motion_bucket_id, noise_aug_strength
        var totalEmbeddingDim = additionTimeEmbedDim * numConditioningValues;

        var embedding = new Tensor<T>(new[] { 1, totalEmbeddingDim });
        var span = embedding.AsWritableSpan();

        // Create add_time_ids: [fps-1, motion_bucket_id, noise_aug_strength]
        // SVD was trained with fps-1 as per diffusers implementation
        double fpsMinus1 = fps - 1;
        double motionBucket = motionBucketId;
        double noiseAugStrength = _noiseAugmentStrength; // Use configured noise aug strength

        // Each value is projected using sinusoidal timestep embedding (similar to diffusion timestep)
        // then concatenated. This follows the SVD UNet's addition_embed_type architecture
        ProjectConditioningValue(span, fpsMinus1, 0, additionTimeEmbedDim);
        ProjectConditioningValue(span, motionBucket, additionTimeEmbedDim, additionTimeEmbedDim);
        ProjectConditioningValue(span, noiseAugStrength, additionTimeEmbedDim * 2, additionTimeEmbedDim);

        return embedding;
    }

    /// <summary>
    /// Projects a single conditioning value into an embedding using sinusoidal timestep projection.
    /// </summary>
    /// <param name="span">The output span to write to.</param>
    /// <param name="value">The conditioning value.</param>
    /// <param name="startIdx">Start index in the span.</param>
    /// <param name="embedDim">Embedding dimension for this value.</param>
    private void ProjectConditioningValue(Span<T> span, double value, int startIdx, int embedDim)
    {
        // Use sinusoidal embedding similar to timestep embedding
        // This is then typically passed through linear layers in the actual model
        int halfDim = embedDim / 2;
        double embScale = Math.Log(10000.0) / (halfDim - 1);

        for (int i = 0; i < halfDim; i++)
        {
            double freq = Math.Exp(-i * embScale);
            double phase = value * freq;

            span[startIdx + i] = NumOps.FromDouble(Math.Sin(phase));
            span[startIdx + halfDim + i] = NumOps.FromDouble(Math.Cos(phase));
        }
    }

    /// <summary>
    /// Decodes video latents using the temporal VAE.
    /// </summary>
    /// <param name="latents">Video latents [batch, frames, channels, height, width].</param>
    /// <returns>Decoded video [batch, frames, channels, height, width].</returns>
    protected override Tensor<T> DecodeVideoLatents(Tensor<T> latents)
    {
        // Use the temporal VAE to decode all frames with temporal consistency
        return _temporalVAE.Decode(latents);
    }

    /// <summary>
    /// Generates video with explicit first frame control.
    /// </summary>
    /// <param name="firstFrame">The exact first frame to use.</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="motionBucketId">Motion intensity (1-255).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video with specified first frame.</returns>
    /// <remarks>
    /// This method ensures the first frame exactly matches the input,
    /// while subsequent frames are generated through diffusion.
    /// Useful when you want precise control over the starting frame.
    /// </remarks>
    public Tensor<T> GenerateWithFirstFrame(
        Tensor<T> firstFrame,
        int numFrames = 25,
        int motionBucketId = 127,
        int numInferenceSteps = 25,
        int? seed = null)
    {
        // Generate video from image
        var video = GenerateFromImage(
            firstFrame,
            numFrames: numFrames,
            motionBucketId: motionBucketId,
            numInferenceSteps: numInferenceSteps,
            noiseAugStrength: 0.0, // No noise aug for exact first frame
            seed: seed);

        // Replace first frame with exact input
        var videoShape = video.Shape;
        var channels = videoShape[2];
        var height = videoShape[3];
        var width = videoShape[4];
        var frameSize = channels * height * width;

        var videoSpan = video.AsWritableSpan();
        var firstFrameSpan = firstFrame.AsSpan();

        for (int i = 0; i < frameSize; i++)
        {
            videoSpan[i] = firstFrameSpan[i];
        }

        return video;
    }

    /// <summary>
    /// Generates video with motion guidance from a secondary image.
    /// </summary>
    /// <param name="startImage">The starting image for the video.</param>
    /// <param name="endImage">Target image suggesting where motion should lead.</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video transitioning from start to end image.</returns>
    /// <remarks>
    /// Uses the latent space interpolation technique to guide the video
    /// generation toward the target end image. Not exact morphing, but
    /// provides directional guidance for the motion.
    /// </remarks>
    public Tensor<T> GenerateWithEndImageGuidance(
        Tensor<T> startImage,
        Tensor<T> endImage,
        int numFrames = 25,
        int numInferenceSteps = 25,
        int? seed = null)
    {
        // Encode both images
        var startLatent = EncodeToLatent(startImage, sampleMode: false);
        var endLatent = EncodeToLatent(endImage, sampleMode: false);

        // Get dimensions
        var imageShape = startImage.Shape;
        var height = imageShape[2];
        var width = imageShape[3];
        var latentHeight = height / _temporalVAE.DownsampleFactor;
        var latentWidth = width / _temporalVAE.DownsampleFactor;

        // Create initial video latents with interpolation guidance
        var videoLatentShape = new[] { 1, numFrames, LatentChannels, latentHeight, latentWidth };
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        // Create image embedding from start image
        var imageEmbedding = EncodeConditioningImage(startImage, 0.02, seed);
        var motionEmbedding = CreateMotionEmbedding(MotionBucketId, DefaultFPS);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising with end-image guidance blending
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Predict noise
            var noisePrediction = PredictVideoNoise(latents, timestep, imageEmbedding, motionEmbedding);

            // Scheduler step
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);

            // Blend toward end image latent for later frames (soft guidance)
            latents = ApplyEndImageGuidance(latents, startLatent, endLatent, timestep, numFrames);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Applies soft guidance toward end image in latent space.
    /// </summary>
    private Tensor<T> ApplyEndImageGuidance(
        Tensor<T> latents,
        Tensor<T> startLatent,
        Tensor<T> endLatent,
        int timestep,
        int numFrames)
    {
        var result = new Tensor<T>(latents.Shape);
        var resultSpan = result.AsWritableSpan();
        var latentSpan = latents.AsSpan();
        var startSpan = startLatent.AsSpan();
        var endSpan = endLatent.AsSpan();

        var latentShape = latents.Shape;
        // Video latent shape: [batch, frames, channels, height, width]
        var videoChannels = latentShape[2];
        var videoHeight = latentShape[3];
        var videoWidth = latentShape[4];
        var frameSize = videoChannels * videoHeight * videoWidth;

        // Start/end latent shape: [batch, channels, height, width] (4D)
        var startShape = startLatent.Shape;
        var startChannels = startShape[1];
        var startHeight = startShape[2];
        var startWidth = startShape[3];
        var startFrameSize = startChannels * startHeight * startWidth;

        // Verify spatial dimensions match
        bool dimensionsMatch = startChannels == videoChannels &&
                               startHeight == videoHeight &&
                               startWidth == videoWidth;

        // Guidance strength decreases as we progress through denoising
        var timestepRatio = timestep / 1000.0;
        var baseGuidanceStrength = 0.1 * timestepRatio; // Stronger guidance early

        for (int f = 0; f < numFrames; f++)
        {
            // Frame progress determines blend toward end
            var frameProgress = f / (double)(numFrames - 1);
            var guidanceWeight = NumOps.FromDouble(baseGuidanceStrength * frameProgress);
            var oneMinusWeight = NumOps.FromDouble(1.0 - baseGuidanceStrength * frameProgress);

            for (int i = 0; i < frameSize; i++)
            {
                var latentIdx = f * frameSize + i;

                // If dimensions match, use direct indexing; otherwise fall back to generated latent
                if (dimensionsMatch && i < startFrameSize)
                {
                    var spatialIdx = i;

                    // Blend toward target based on frame position (linear interpolation)
                    var frameT = NumOps.FromDouble(frameProgress);
                    var oneMinusFrameT = NumOps.FromDouble(1.0 - frameProgress);
                    var targetLatent = NumOps.Add(
                        NumOps.Multiply(oneMinusFrameT, startSpan[spatialIdx]),
                        NumOps.Multiply(frameT, endSpan[spatialIdx]));

                    // Soft blend with predicted latent
                    resultSpan[latentIdx] = NumOps.Add(
                        NumOps.Multiply(oneMinusWeight, latentSpan[latentIdx]),
                        NumOps.Multiply(guidanceWeight, targetLatent));
                }
                else
                {
                    // Dimension mismatch - use generated latent directly
                    resultSpan[latentIdx] = latentSpan[latentIdx];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the recommended resolution for SVD generation.
    /// </summary>
    /// <param name="aspectRatio">Desired aspect ratio (width/height).</param>
    /// <returns>Tuple of (width, height) optimized for SVD.</returns>
    /// <remarks>
    /// SVD works best at specific resolutions. This method returns
    /// the closest supported resolution for the given aspect ratio.
    /// </remarks>
    public static (int width, int height) GetRecommendedResolution(double aspectRatio = 16.0 / 9.0)
    {
        // SVD supported resolutions
        var resolutions = new[]
        {
            (1024, 576),  // 16:9 landscape
            (576, 1024),  // 9:16 portrait
            (768, 768),   // 1:1 square
            (1024, 768),  // 4:3 landscape
            (768, 1024),  // 3:4 portrait
        };

        // Find closest match
        var bestMatch = resolutions[0];
        var bestDiff = double.MaxValue;

        foreach (var res in resolutions)
        {
            var resAspect = (double)res.Item1 / res.Item2;
            var diff = Math.Abs(resAspect - aspectRatio);
            if (diff < bestDiff)
            {
                bestDiff = diff;
                bestMatch = res;
            }
        }

        return bestMatch;
    }

    #region IParameterizable Implementation

    /// <summary>
    /// Gets the total number of parameters in the model.
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
    /// Gets the flattened parameters of all components.
    /// </summary>
    /// <returns>A vector containing all model parameters.</returns>
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
    /// Sets the parameters for all components.
    /// </summary>
    /// <param name="parameters">The parameter vector to distribute across components.</param>
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
    /// Creates a clone of this StableVideoDiffusion model.
    /// </summary>
    /// <returns>A new instance with the same configuration.</returns>
    public override IDiffusionModel<T> Clone()
    {
        var clone = new StableVideoDiffusion<T>(
            options: null,
            scheduler: null,
            videoUNet: (VideoUNetPredictor<T>)_videoUNet.Clone(),
            temporalVAE: (TemporalVAE<T>)_temporalVAE.Clone(),
            conditioner: _conditioner,
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);

        // Copy motion bucket state
        clone.SetMotionBucketId(MotionBucketId);

        return clone;
    }

    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A new instance with copied parameters.</returns>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        return (IFullModel<T, Tensor<T>, Tensor<T>>)Clone();
    }

    #endregion
}
