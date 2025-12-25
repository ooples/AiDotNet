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
/// AnimateDiff model for text-to-video and image-to-video generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// AnimateDiff extends Stable Diffusion with motion modules that enable temporal consistency
/// in video generation. Unlike SVD which is trained end-to-end for video, AnimateDiff adds
/// motion modules to existing text-to-image models, making it highly flexible.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of AnimateDiff as "teaching an image generator to make videos."
///
/// How it works:
/// 1. Start with a text-to-image model (like Stable Diffusion)
/// 2. Add special "motion modules" between the layers
/// 3. These modules learn how things move in videos
/// 4. The original image quality is preserved while adding motion
///
/// Key advantages:
/// - Works with any Stable Diffusion model/checkpoint
/// - Can use existing LoRAs, ControlNets, etc.
/// - Flexible: text-to-video, image-to-video, or both
/// - Lower training requirements than full video models
///
/// Example use cases:
/// - Generate a short animation from a text prompt
/// - Animate a still image with natural motion
/// - Create consistent character animations
/// - Style transfer for videos using SD checkpoints
/// </para>
/// <para>
/// Architecture overview:
/// - Base: Standard Stable Diffusion U-Net
/// - Motion Modules: Temporal attention layers inserted after spatial attention
/// - VAE: Standard SD VAE (per-frame encoding/decoding)
/// - Optional: LoRA adapters for style customization
///
/// Supported modes:
/// - Text-to-Video: Generate video from text prompt
/// - Image-to-Video: Animate an input image with text guidance
/// - Video-to-Video: Style transfer or modify existing video
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create AnimateDiff with default motion modules
/// var animateDiff = new AnimateDiffModel&lt;float&gt;();
///
/// // Text-to-video generation
/// var video = animateDiff.GenerateFromText(
///     prompt: "A beautiful sunset over the ocean, waves gently rolling",
///     width: 512,
///     height: 512,
///     numFrames: 16,
///     numInferenceSteps: 25);
///
/// // Image-to-video with text guidance
/// var inputImage = LoadImage("beach.jpg");
/// var animatedVideo = animateDiff.AnimateImage(
///     inputImage,
///     prompt: "gentle waves, moving clouds",
///     numFrames: 16);
/// </code>
/// </example>
public class AnimateDiffModel<T> : VideoDiffusionModelBase<T>
{
    /// <summary>
    /// Default AnimateDiff width (SD compatible).
    /// </summary>
    public const int DefaultWidth = 512;

    /// <summary>
    /// Default AnimateDiff height (SD compatible).
    /// </summary>
    public const int DefaultHeight = 512;

    /// <summary>
    /// Standard AnimateDiff latent channels.
    /// </summary>
    private const int ANIMATEDIFF_LATENT_CHANNELS = 4;

    /// <summary>
    /// Standard latent scale factor.
    /// </summary>
    private const double LATENT_SCALE = 0.18215;

    /// <summary>
    /// The U-Net noise predictor with motion modules.
    /// </summary>
    private readonly UNetNoisePredictor<T> _unet;

    /// <summary>
    /// The motion module weights/state.
    /// </summary>
    private readonly MotionModuleConfig _motionConfig;

    /// <summary>
    /// The standard VAE for frame encoding/decoding.
    /// </summary>
    private readonly StandardVAE<T> _vae;

    /// <summary>
    /// Optional conditioning module for text guidance.
    /// </summary>
    private readonly IConditioningModule<T>? _conditioner;

    /// <summary>
    /// Gets the noise predictor.
    /// </summary>
    public override INoisePredictor<T> NoisePredictor => _unet;

    /// <summary>
    /// Gets the VAE for frame encoding/decoding.
    /// </summary>
    public override IVAEModel<T> VAE => _vae;

    /// <summary>
    /// Gets the conditioning module.
    /// </summary>
    public override IConditioningModule<T>? Conditioner => _conditioner;

    /// <summary>
    /// Gets the number of latent channels.
    /// </summary>
    public override int LatentChannels => ANIMATEDIFF_LATENT_CHANNELS;

    /// <summary>
    /// Gets whether image-to-video is supported.
    /// </summary>
    /// <remarks>
    /// AnimateDiff supports animating still images when a conditioner is available.
    /// </remarks>
    public override bool SupportsImageToVideo => _conditioner != null;

    /// <summary>
    /// Gets whether text-to-video is supported.
    /// </summary>
    /// <remarks>
    /// AnimateDiff's primary mode is text-to-video.
    /// </remarks>
    public override bool SupportsTextToVideo => _conditioner != null;

    /// <summary>
    /// Gets whether video-to-video is supported.
    /// </summary>
    public override bool SupportsVideoToVideo => _conditioner != null;

    /// <summary>
    /// Gets the motion module configuration.
    /// </summary>
    public MotionModuleConfig MotionConfig => _motionConfig;

    /// <summary>
    /// Gets or sets the context length for temporal attention.
    /// </summary>
    /// <remarks>
    /// Controls how many frames are processed together in the motion modules.
    /// Larger values provide better temporal consistency but require more memory.
    /// </remarks>
    public int ContextLength { get; set; } = 16;

    /// <summary>
    /// Gets or sets the context overlap for sliding window generation.
    /// </summary>
    /// <remarks>
    /// When generating more frames than ContextLength, this controls
    /// the overlap between windows to maintain smooth transitions.
    /// </remarks>
    public int ContextOverlap { get; set; } = 4;

    /// <summary>
    /// Initializes a new instance of AnimateDiffModel with default parameters.
    /// </summary>
    public AnimateDiffModel()
        : this(
            options: null,
            scheduler: null,
            unet: null,
            vae: null,
            conditioner: null,
            motionConfig: null,
            defaultNumFrames: 16,
            defaultFPS: 8)
    {
    }

    /// <summary>
    /// Initializes a new instance of AnimateDiffModel with custom parameters.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="unet">Optional custom U-Net noise predictor.</param>
    /// <param name="vae">Optional custom VAE.</param>
    /// <param name="conditioner">Optional conditioning module for text guidance.</param>
    /// <param name="motionConfig">Optional motion module configuration.</param>
    /// <param name="defaultNumFrames">Default number of frames to generate.</param>
    /// <param name="defaultFPS">Default frames per second.</param>
    public AnimateDiffModel(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        UNetNoisePredictor<T>? unet = null,
        StandardVAE<T>? vae = null,
        IConditioningModule<T>? conditioner = null,
        MotionModuleConfig? motionConfig = null,
        int defaultNumFrames = 16,
        int defaultFPS = 8)
        : base(options, scheduler ?? CreateDefaultScheduler(), defaultNumFrames, defaultFPS)
    {
        _motionConfig = motionConfig ?? new MotionModuleConfig();
        _unet = unet ?? CreateDefaultUNet();
        _vae = vae ?? CreateDefaultVAE();
        _conditioner = conditioner;
    }

    /// <summary>
    /// Creates the default scheduler for AnimateDiff.
    /// </summary>
    private static DDIMScheduler<T> CreateDefaultScheduler()
    {
        var config = SchedulerConfig<T>.CreateStableDiffusion();
        return new DDIMScheduler<T>(config);
    }

    /// <summary>
    /// Creates the default U-Net predictor.
    /// </summary>
    private UNetNoisePredictor<T> CreateDefaultUNet()
    {
        // Standard SD U-Net architecture
        return new UNetNoisePredictor<T>(
            inputChannels: ANIMATEDIFF_LATENT_CHANNELS,
            outputChannels: ANIMATEDIFF_LATENT_CHANNELS,
            baseChannels: 320,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            numResBlocks: 2,
            attentionResolutions: new[] { 4, 2, 1 },
            contextDim: 768,
            numHeads: 8);
    }

    /// <summary>
    /// Creates the default VAE.
    /// </summary>
    private StandardVAE<T> CreateDefaultVAE()
    {
        return new StandardVAE<T>(
            inputChannels: 3,
            latentChannels: ANIMATEDIFF_LATENT_CHANNELS,
            baseChannels: 128,
            channelMultipliers: new[] { 1, 2, 4, 4 },
            latentScaleFactor: LATENT_SCALE);
    }

    /// <summary>
    /// Generates video from text using AnimateDiff.
    /// </summary>
    /// <param name="prompt">The text prompt describing the video.</param>
    /// <param name="negativePrompt">Optional negative prompt.</param>
    /// <param name="width">Video width.</param>
    /// <param name="height">Video height.</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="fps">Frames per second (for motion module).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video tensor [batch, numFrames, channels, height, width].</returns>
    public override Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        int? seed = null)
    {
        if (_conditioner == null)
            throw new InvalidOperationException("Text-to-video generation requires a conditioning module.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var effectiveFPS = fps ?? DefaultFPS;
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = _conditioner.Tokenize(prompt);
        var promptEmbedding = _conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = _conditioner.Tokenize(negativePrompt);
                negativeEmbedding = _conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = _conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate latent dimensions
        var latentHeight = height / _vae.DownsampleFactor;
        var latentWidth = width / _vae.DownsampleFactor;

        // Use context windowing for longer videos
        if (effectiveNumFrames <= ContextLength)
        {
            // Generate all frames at once
            return GenerateVideoWindow(
                effectiveNumFrames,
                latentHeight,
                latentWidth,
                promptEmbedding,
                negativeEmbedding,
                guidanceScale,
                numInferenceSteps,
                seed);
        }
        else
        {
            // Use sliding window approach for longer videos
            return GenerateWithSlidingWindow(
                effectiveNumFrames,
                latentHeight,
                latentWidth,
                promptEmbedding,
                negativeEmbedding,
                guidanceScale,
                numInferenceSteps,
                seed);
        }
    }

    /// <summary>
    /// Generates a single window of video frames.
    /// </summary>
    private Tensor<T> GenerateVideoWindow(
        int numFrames,
        int latentHeight,
        int latentWidth,
        Tensor<T> promptEmbedding,
        Tensor<T>? negativeEmbedding,
        double guidanceScale,
        int numInferenceSteps,
        int? seed)
    {
        var videoLatentShape = new[] { 1, numFrames, LatentChannels, latentHeight, latentWidth };
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (negativeEmbedding != null && guidanceScale > 1.0)
            {
                // Process all frames with motion-aware prediction
                var condPred = PredictWithMotionModules(latents, timestep, promptEmbedding);
                var uncondPred = PredictWithMotionModules(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidanceVideo(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = PredictWithMotionModules(latents, timestep, promptEmbedding);
            }

            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Generates video using sliding window approach for longer sequences.
    /// </summary>
    private Tensor<T> GenerateWithSlidingWindow(
        int totalFrames,
        int latentHeight,
        int latentWidth,
        Tensor<T> promptEmbedding,
        Tensor<T>? negativeEmbedding,
        double guidanceScale,
        int numInferenceSteps,
        int? seed)
    {
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;

        // Initialize latents for all frames
        var videoLatentShape = new[] { 1, totalFrames, LatentChannels, latentHeight, latentWidth };
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        Scheduler.SetTimesteps(numInferenceSteps);

        // Calculate window positions
        var windowStride = ContextLength - ContextOverlap;
        var numWindows = (int)Math.Ceiling((double)(totalFrames - ContextOverlap) / windowStride);

        foreach (var timestep in Scheduler.Timesteps)
        {
            // Accumulator for blending overlapping predictions
            var accumulatedNoise = new Tensor<T>(videoLatentShape);
            var accumulatedWeights = new Tensor<T>(videoLatentShape);

            for (int w = 0; w < numWindows; w++)
            {
                var startFrame = w * windowStride;
                var endFrame = Math.Min(startFrame + ContextLength, totalFrames);
                var windowSize = endFrame - startFrame;

                // Extract window latents
                var windowLatents = ExtractFrameWindow(latents, startFrame, windowSize);

                // Predict noise for this window
                Tensor<T> windowNoise;
                if (negativeEmbedding != null && guidanceScale > 1.0)
                {
                    var condPred = PredictWithMotionModules(windowLatents, timestep, promptEmbedding);
                    var uncondPred = PredictWithMotionModules(windowLatents, timestep, negativeEmbedding);
                    windowNoise = ApplyGuidanceVideo(uncondPred, condPred, guidanceScale);
                }
                else
                {
                    windowNoise = PredictWithMotionModules(windowLatents, timestep, promptEmbedding);
                }

                // Blend window noise into accumulator with linear weights
                BlendWindowNoise(accumulatedNoise, accumulatedWeights, windowNoise, startFrame, windowSize);
            }

            // Normalize accumulated noise by weights
            var noisePrediction = NormalizeAccumulatedNoise(accumulatedNoise, accumulatedWeights);

            // Scheduler step
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Predicts noise using the U-Net with motion module awareness.
    /// </summary>
    private Tensor<T> PredictWithMotionModules(
        Tensor<T> latents,
        int timestep,
        Tensor<T> textEmbedding)
    {
        var videoShape = latents.Shape;
        var numFrames = videoShape[1];

        // AnimateDiff processes frames through the U-Net with motion modules
        // For each frame, we pass it through the spatial U-Net layers
        // and apply temporal attention in the motion modules

        var result = new Tensor<T>(videoShape);
        var resultSpan = result.AsWritableSpan();
        var latentSpan = latents.AsSpan();

        var batchSize = videoShape[0];
        var channels = videoShape[2];
        var height = videoShape[3];
        var width = videoShape[4];
        var frameSize = channels * height * width;

        // Process each frame through the U-Net
        // In a full implementation, motion modules would apply temporal attention
        // across frames. Here we use a simplified per-frame + blending approach.
        for (int f = 0; f < numFrames; f++)
        {
            // Extract frame latent
            var frameShape = new[] { batchSize, channels, height, width };
            var frameLatent = new Tensor<T>(frameShape);
            var frameLatentSpan = frameLatent.AsWritableSpan();

            for (int b = 0; b < batchSize; b++)
            {
                var batchOffset = b * numFrames * frameSize + f * frameSize;
                for (int i = 0; i < frameSize; i++)
                {
                    frameLatentSpan[b * frameSize + i] = latentSpan[batchOffset + i];
                }
            }

            // Predict noise for this frame
            var frameNoise = _unet.PredictNoise(frameLatent, timestep, textEmbedding);
            var frameNoiseSpan = frameNoise.AsSpan();

            // Apply temporal smoothing from motion modules
            // This is a simplified version - full implementation would use temporal attention
            var temporalWeight = ApplyMotionModuleWeight(f, numFrames);

            // Insert into result with temporal weighting
            for (int b = 0; b < batchSize; b++)
            {
                var batchOffset = b * numFrames * frameSize + f * frameSize;
                for (int i = 0; i < frameSize; i++)
                {
                    var weighted = NumOps.Multiply(temporalWeight, frameNoiseSpan[b * frameSize + i]);
                    resultSpan[batchOffset + i] = weighted;
                }
            }
        }

        // Apply temporal smoothing across frames
        ApplyTemporalSmoothing(result, numFrames);

        return result;
    }

    /// <summary>
    /// Applies motion module temporal weighting.
    /// </summary>
    private T ApplyMotionModuleWeight(int frameIndex, int totalFrames)
    {
        // Use motion module configuration to compute temporal weight
        // This helps with temporal consistency at the edges
        if (totalFrames <= 1)
            return NumOps.One;

        var position = (double)frameIndex / (totalFrames - 1);

        // Cosine bell weighting - stronger in the middle
        var weight = 0.5 - 0.5 * Math.Cos(2 * Math.PI * position);

        // Scale based on motion module strength
        weight = 1.0 - _motionConfig.TemporalFalloff * (1.0 - weight);

        return NumOps.FromDouble(Math.Max(0.5, Math.Min(1.0, weight)));
    }

    /// <summary>
    /// Applies temporal smoothing across predicted noise.
    /// </summary>
    private void ApplyTemporalSmoothing(Tensor<T> noise, int numFrames)
    {
        if (numFrames <= 2 || _motionConfig.TemporalSmoothing <= 0)
            return;

        var shape = noise.Shape;
        var batchSize = shape[0];
        var channels = shape[2];
        var height = shape[3];
        var width = shape[4];
        var frameSize = channels * height * width;

        var noiseSpan = noise.AsWritableSpan();
        var smoothingWeight = NumOps.FromDouble(_motionConfig.TemporalSmoothing);
        var mainWeight = NumOps.FromDouble(1.0 - _motionConfig.TemporalSmoothing);

        // Simple 1D temporal convolution (average with neighbors)
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 1; f < numFrames - 1; f++)
            {
                for (int i = 0; i < frameSize; i++)
                {
                    var prevIdx = b * numFrames * frameSize + (f - 1) * frameSize + i;
                    var currIdx = b * numFrames * frameSize + f * frameSize + i;
                    var nextIdx = b * numFrames * frameSize + (f + 1) * frameSize + i;

                    var avg = NumOps.Multiply(
                        NumOps.FromDouble(0.5),
                        NumOps.Add(noiseSpan[prevIdx], noiseSpan[nextIdx]));

                    noiseSpan[currIdx] = NumOps.Add(
                        NumOps.Multiply(mainWeight, noiseSpan[currIdx]),
                        NumOps.Multiply(smoothingWeight, avg));
                }
            }
        }
    }

    /// <summary>
    /// Extracts a window of frames from video latents.
    /// </summary>
    private Tensor<T> ExtractFrameWindow(Tensor<T> latents, int startFrame, int windowSize)
    {
        var shape = latents.Shape;
        var batchSize = shape[0];
        var channels = shape[2];
        var height = shape[3];
        var width = shape[4];
        var totalFrames = shape[1];
        var frameSize = channels * height * width;

        var windowShape = new[] { batchSize, windowSize, channels, height, width };
        var window = new Tensor<T>(windowShape);
        var windowSpan = window.AsWritableSpan();
        var latentSpan = latents.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < windowSize; f++)
            {
                var srcFrame = startFrame + f;
                if (srcFrame >= totalFrames) break;

                var srcOffset = b * totalFrames * frameSize + srcFrame * frameSize;
                var dstOffset = b * windowSize * frameSize + f * frameSize;

                for (int i = 0; i < frameSize; i++)
                {
                    windowSpan[dstOffset + i] = latentSpan[srcOffset + i];
                }
            }
        }

        return window;
    }

    /// <summary>
    /// Blends window noise into the accumulator with linear weights.
    /// </summary>
    private void BlendWindowNoise(
        Tensor<T> accumulated,
        Tensor<T> weights,
        Tensor<T> windowNoise,
        int startFrame,
        int windowSize)
    {
        var shape = accumulated.Shape;
        var batchSize = shape[0];
        var totalFrames = shape[1];
        var channels = shape[2];
        var height = shape[3];
        var width = shape[4];
        var frameSize = channels * height * width;

        var accSpan = accumulated.AsWritableSpan();
        var weightSpan = weights.AsWritableSpan();
        var noiseSpan = windowNoise.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < windowSize; f++)
            {
                var globalFrame = startFrame + f;
                if (globalFrame >= totalFrames) break;

                // Linear blend weight (higher in the middle of the window)
                var windowPos = (double)f / (windowSize - 1);
                var blendWeight = 1.0 - Math.Abs(windowPos - 0.5) * 0.5;

                var globalOffset = b * totalFrames * frameSize + globalFrame * frameSize;
                var windowOffset = b * windowSize * frameSize + f * frameSize;

                var blendT = NumOps.FromDouble(blendWeight);

                for (int i = 0; i < frameSize; i++)
                {
                    accSpan[globalOffset + i] = NumOps.Add(
                        accSpan[globalOffset + i],
                        NumOps.Multiply(blendT, noiseSpan[windowOffset + i]));
                    weightSpan[globalOffset + i] = NumOps.Add(
                        weightSpan[globalOffset + i],
                        blendT);
                }
            }
        }
    }

    /// <summary>
    /// Normalizes accumulated noise by weights.
    /// </summary>
    private Tensor<T> NormalizeAccumulatedNoise(Tensor<T> accumulated, Tensor<T> weights)
    {
        var result = new Tensor<T>(accumulated.Shape);
        var resultSpan = result.AsWritableSpan();
        var accSpan = accumulated.AsSpan();
        var weightSpan = weights.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            var weight = weightSpan[i];
            if (NumOps.ToDouble(weight) > 1e-8)
            {
                resultSpan[i] = NumOps.Divide(accSpan[i], weight);
            }
            else
            {
                resultSpan[i] = accSpan[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Predicts video noise for image-to-video generation.
    /// </summary>
    protected override Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding)
    {
        // For AnimateDiff, image conditioning is typically done through
        // IP-Adapter or image cross-attention. Here we use a simplified approach.
        return PredictWithMotionModules(latents, timestep, imageEmbedding);
    }

    /// <summary>
    /// Generates video from an input image.
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
        if (_conditioner == null)
            throw new InvalidOperationException("Image-to-video requires a conditioning module.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;

        // Get image dimensions
        var imageShape = inputImage.Shape;
        var height = imageShape[2];
        var width = imageShape[3];

        // Encode image as conditioning
        var imageLatent = EncodeToLatent(inputImage, sampleMode: false);

        // Create motion embedding based on motion bucket
        var effectiveMotionBucket = motionBucketId ?? MotionBucketId;
        var motionEmbed = CreateMotionEmbedding(effectiveMotionBucket, fps ?? DefaultFPS);

        // Calculate latent dimensions
        var latentHeight = height / _vae.DownsampleFactor;
        var latentWidth = width / _vae.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        // Copy first frame from image latent (with noise aug)
        if (noiseAugStrength > 0)
        {
            var noise = DiffusionNoiseHelper<T>.SampleGaussian(imageLatent.Shape, rng);
            var scaledNoise = DiffusionNoiseHelper<T>.ScaleNoise(noise, noiseAugStrength);
            var augLatent = AddTensors(imageLatent, scaledNoise);
            InsertFrameLatent(latents, augLatent, 0);
        }
        else
        {
            InsertFrameLatent(latents, imageLatent, 0);
        }

        Scheduler.SetTimesteps(numInferenceSteps);

        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = PredictVideoNoise(latents, timestep, imageLatent, motionEmbed);
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);

            // Preserve first frame by re-injecting (optional, can be controlled)
            if (_motionConfig.PreserveFirstFrame)
            {
                InsertFrameLatent(latents, imageLatent, 0);
            }
        }

        return DecodeVideoLatents(latents);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        var resultSpan = result.AsWritableSpan();
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Add(aSpan[i], bSpan[i]);
        }

        return result;
    }

    /// <summary>
    /// Decodes video latents to frames.
    /// </summary>
    protected override Tensor<T> DecodeVideoLatents(Tensor<T> latents)
    {
        // AnimateDiff uses per-frame VAE decoding
        var latentShape = latents.Shape;
        var numFrames = latentShape[1];

        var frames = new Tensor<T>[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            var frameLatent = ExtractFrameLatent(latents, f);
            frames[f] = DecodeFromLatent(frameLatent);
        }

        return FramesToVideo(frames);
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
            var unetParams = _unet.GetParameters();
            if (unetParams != null)
            {
                count += unetParams.Length;
            }

            var vaeParams = _vae.GetParameters();
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
        var unetParams = _unet.GetParameters();
        var vaeParams = _vae.GetParameters();

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
        var unetParamCount = _unet.GetParameters()?.Length ?? 0;
        var vaeParamCount = _vae.GetParameters()?.Length ?? 0;

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
            _unet.SetParameters(unetParams);
            offset += unetParamCount;
        }

        if (vaeParamCount > 0)
        {
            var vaeParams = new Vector<T>(vaeParamCount);
            for (int i = 0; i < vaeParamCount; i++)
            {
                vaeParams[i] = parameters[offset + i];
            }
            _vae.SetParameters(vaeParams);
        }
    }

    #endregion

    #region ICloneable Implementation

    /// <summary>
    /// Clones this AnimateDiff model.
    /// </summary>
    public override IDiffusionModel<T> Clone()
    {
        var clone = new AnimateDiffModel<T>(
            options: null,
            scheduler: null,
            unet: (UNetNoisePredictor<T>)_unet.Clone(),
            vae: (StandardVAE<T>)_vae.Clone(),
            conditioner: _conditioner,
            motionConfig: _motionConfig.Clone(),
            defaultNumFrames: DefaultNumFrames,
            defaultFPS: DefaultFPS);

        clone.ContextLength = ContextLength;
        clone.ContextOverlap = ContextOverlap;
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

/// <summary>
/// Configuration for AnimateDiff motion modules.
/// </summary>
public class MotionModuleConfig
{
    /// <summary>
    /// Gets or sets the temporal smoothing strength (0-1).
    /// </summary>
    public double TemporalSmoothing { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the temporal falloff for edge frames.
    /// </summary>
    public double TemporalFalloff { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether to preserve the first frame in image-to-video.
    /// </summary>
    public bool PreserveFirstFrame { get; set; } = true;

    /// <summary>
    /// Gets or sets the motion module version.
    /// </summary>
    public string Version { get; set; } = "v2";

    /// <summary>
    /// Clones this configuration.
    /// </summary>
    public MotionModuleConfig Clone()
    {
        return new MotionModuleConfig
        {
            TemporalSmoothing = TemporalSmoothing,
            TemporalFalloff = TemporalFalloff,
            PreserveFirstFrame = PreserveFirstFrame,
            Version = Version
        };
    }
}
