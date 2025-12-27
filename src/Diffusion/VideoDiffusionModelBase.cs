using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Diffusion;
using AiDotNet.NeuralNetworks.Diffusion.Schedulers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for video diffusion models that generate temporal sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all video diffusion models,
/// including image-to-video generation, text-to-video generation, video-to-video transformation,
/// and frame interpolation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for video generation models like Stable Video Diffusion
/// and AnimateDiff. It extends latent diffusion to handle the temporal dimension, generating
/// coherent video sequences where frames are consistent over time.
/// </para>
/// <para>
/// Key capabilities:
/// - Image-to-Video: Animate a still image
/// - Text-to-Video: Generate video from text description
/// - Video-to-Video: Transform existing video style/content
/// - Frame interpolation: Increase frame rate smoothly
/// </para>
/// </remarks>
public abstract class VideoDiffusionModelBase<T> : LatentDiffusionModelBase<T>, IVideoDiffusionModel<T>
{
    /// <summary>
    /// The motion bucket ID for controlling motion intensity.
    /// </summary>
    private int _motionBucketId = 127;

    /// <summary>
    /// Default number of frames to generate.
    /// </summary>
    private readonly int _defaultNumFrames;

    /// <summary>
    /// Default frames per second.
    /// </summary>
    private readonly int _defaultFPS;

    /// <inheritdoc />
    public virtual int DefaultNumFrames => _defaultNumFrames;

    /// <inheritdoc />
    public virtual int DefaultFPS => _defaultFPS;

    /// <inheritdoc />
    public abstract bool SupportsImageToVideo { get; }

    /// <inheritdoc />
    public abstract bool SupportsTextToVideo { get; }

    /// <inheritdoc />
    public abstract bool SupportsVideoToVideo { get; }

    /// <inheritdoc />
    public virtual int MotionBucketId => _motionBucketId;

    /// <summary>
    /// Gets the temporal VAE for video encoding/decoding.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A temporal VAE processes video frames together,
    /// maintaining consistency across time. It's better than processing each frame
    /// independently because it avoids flickering.
    /// </para>
    /// </remarks>
    public virtual IVAEModel<T>? TemporalVAE => null;

    /// <summary>
    /// Gets the noise augmentation strength for input images.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Adding slight noise to the conditioning image encourages the model
    /// to generate motion rather than static frames.
    /// </para>
    /// </remarks>
    public virtual double NoiseAugStrength { get; protected set; } = 0.02;

    /// <summary>
    /// Initializes a new instance of the VideoDiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="defaultNumFrames">Default number of frames to generate.</param>
    /// <param name="defaultFPS">Default frames per second.</param>
    protected VideoDiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int defaultNumFrames = 25,
        int defaultFPS = 7)
        : base(options, scheduler)
    {
        _defaultNumFrames = defaultNumFrames;
        _defaultFPS = defaultFPS;
    }

    #region IVideoDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> GenerateFromImage(
        Tensor<T> inputImage,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 25,
        int? motionBucketId = null,
        double noiseAugStrength = 0.02,
        int? seed = null)
    {
        if (!SupportsImageToVideo)
            throw new NotSupportedException("This model does not support image-to-video generation.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var effectiveFPS = fps ?? DefaultFPS;
        var effectiveMotionBucket = motionBucketId ?? MotionBucketId;

        // Get image dimensions
        var imageShape = inputImage.Shape;
        var height = imageShape[2];
        var width = imageShape[3];

        // Encode conditioning image
        var imageEmbedding = EncodeConditioningImage(inputImage, noiseAugStrength, seed);

        // Create motion embedding
        var motionEmbedding = CreateMotionEmbedding(effectiveMotionBucket, effectiveFPS);

        // Calculate video latent dimensions
        var latentHeight = height / VAE.DownsampleFactor;
        var latentWidth = width / VAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise for all frames
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop with temporal conditioning
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Predict noise for all frames conditioned on image and motion
            var noisePrediction = PredictVideoNoise(
                latents,
                timestep,
                imageEmbedding,
                motionEmbedding);

            // Scheduler step for each frame
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        // Decode video latents to frames
        return DecodeVideoLatents(latents);
    }

    /// <inheritdoc />
    public virtual Tensor<T> GenerateFromText(
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
        if (!SupportsTextToVideo)
            throw new NotSupportedException("This model does not support text-to-video generation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Text-to-video generation requires a conditioning module.");

        var effectiveNumFrames = numFrames ?? DefaultNumFrames;
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate video latent dimensions
        var latentHeight = height / VAE.DownsampleFactor;
        var latentWidth = width / VAE.DownsampleFactor;
        var videoLatentShape = new[] { 1, effectiveNumFrames, LatentChannels, latentHeight, latentWidth };

        // Generate initial noise for all frames
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(videoLatentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                // Classifier-free guidance
                var condPred = PredictVideoNoiseWithText(latents, timestep, promptEmbedding);
                var uncondPred = PredictVideoNoiseWithText(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidanceVideo(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = PredictVideoNoiseWithText(latents, timestep, promptEmbedding);
            }

            // Scheduler step
            latents = SchedulerStepVideo(latents, noisePrediction, timestep);
        }

        // Decode video latents to frames
        return DecodeVideoLatents(latents);
    }

    /// <inheritdoc />
    public virtual Tensor<T> VideoToVideo(
        Tensor<T> inputVideo,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.7,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        int? seed = null)
    {
        if (!SupportsVideoToVideo)
            throw new NotSupportedException("This model does not support video-to-video transformation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Video-to-video transformation requires a conditioning module.");

        strength = MathPolyfill.Clamp(strength, 0.0, 1.0);
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode input video to latents
        var latents = EncodeVideoToLatent(inputVideo);

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate starting timestep based on strength
        Scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - strength));
        var startTimestep = Scheduler.Timesteps.Skip(startStep).First();

        // Add noise to latents at starting timestep
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        latents = AddNoiseToVideoLatents(latents, startTimestep, rng);

        // Denoising loop (starting from startStep)
        foreach (var timestep in Scheduler.Timesteps.Skip(startStep))
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

    /// <inheritdoc />
    public virtual Tensor<T> InterpolateFrames(
        Tensor<T> video,
        int targetFPS,
        FrameInterpolationMethod interpolationMethod = FrameInterpolationMethod.Diffusion)
    {
        var videoShape = video.Shape;
        var currentFrames = videoShape[1];
        var currentFPS = DefaultFPS;
        var targetFrames = (int)Math.Ceiling((double)currentFrames * targetFPS / currentFPS);

        if (targetFrames <= currentFrames)
            return video;

        return interpolationMethod switch
        {
            FrameInterpolationMethod.Diffusion => InterpolateFramesDiffusion(video, targetFrames),
            FrameInterpolationMethod.OpticalFlow => InterpolateFramesOpticalFlow(video, targetFrames),
            FrameInterpolationMethod.Linear => InterpolateFramesLinear(video, targetFrames),
            FrameInterpolationMethod.Blend => InterpolateFramesBlend(video, targetFrames),
            _ => InterpolateFramesLinear(video, targetFrames)
        };
    }

    /// <inheritdoc />
    public virtual void SetMotionBucketId(int bucketId)
    {
        if (bucketId < 1 || bucketId > 255)
            throw new ArgumentOutOfRangeException(nameof(bucketId), "Motion bucket ID must be between 1 and 255.");
        _motionBucketId = bucketId;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ExtractFrame(Tensor<T> video, int frameIndex)
    {
        var videoShape = video.Shape;
        var numFrames = videoShape[1];

        if (frameIndex < 0 || frameIndex >= numFrames)
            throw new ArgumentOutOfRangeException(nameof(frameIndex), $"Frame index must be between 0 and {numFrames - 1}.");

        var batchSize = videoShape[0];
        var channels = videoShape[2];
        var height = videoShape[3];
        var width = videoShape[4];

        var frameShape = new[] { batchSize, channels, height, width };
        var frame = new Tensor<T>(frameShape);
        var frameSpan = frame.AsWritableSpan();
        var videoSpan = video.AsSpan();

        var frameSize = channels * height * width;
        var frameOffset = frameIndex * frameSize;

        for (int b = 0; b < batchSize; b++)
        {
            var batchOffset = b * numFrames * frameSize;
            for (int i = 0; i < frameSize; i++)
            {
                frameSpan[b * frameSize + i] = videoSpan[batchOffset + frameOffset + i];
            }
        }

        return frame;
    }

    /// <inheritdoc />
    public virtual Tensor<T> FramesToVideo(Tensor<T>[] frames)
    {
        if (frames == null || frames.Length == 0)
            throw new ArgumentException("Frames array must not be empty.", nameof(frames));

        var firstFrame = frames[0];
        var frameShape = firstFrame.Shape;
        var batchSize = frameShape[0];
        var channels = frameShape[1];
        var height = frameShape[2];
        var width = frameShape[3];
        var numFrames = frames.Length;

        var videoShape = new[] { batchSize, numFrames, channels, height, width };
        var video = new Tensor<T>(videoShape);
        var videoSpan = video.AsWritableSpan();

        var frameSize = channels * height * width;

        for (int f = 0; f < numFrames; f++)
        {
            var frameSpan = frames[f].AsSpan();
            for (int b = 0; b < batchSize; b++)
            {
                var batchOffset = b * numFrames * frameSize + f * frameSize;
                for (int i = 0; i < frameSize; i++)
                {
                    videoSpan[batchOffset + i] = frameSpan[b * frameSize + i];
                }
            }
        }

        return video;
    }

    #endregion

    #region Protected Methods for Derived Classes

    /// <summary>
    /// Encodes a conditioning image for image-to-video generation.
    /// </summary>
    /// <param name="image">The conditioning image.</param>
    /// <param name="noiseAugStrength">Noise augmentation strength.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>The encoded image embedding.</returns>
    protected virtual Tensor<T> EncodeConditioningImage(Tensor<T> image, double noiseAugStrength, int? seed)
    {
        // Add noise augmentation to encourage motion
        if (noiseAugStrength > 0)
        {
            var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
            var noise = DiffusionNoiseHelper<T>.SampleGaussian(image.Shape, rng);
            var scaledNoise = DiffusionNoiseHelper<T>.ScaleNoise(noise, noiseAugStrength);

            var augmented = new Tensor<T>(image.Shape);
            var augSpan = augmented.AsWritableSpan();
            var imgSpan = image.AsSpan();
            var noiseSpan = scaledNoise.AsSpan();

            for (int i = 0; i < augSpan.Length; i++)
            {
                augSpan[i] = NumOps.Add(imgSpan[i], noiseSpan[i]);
            }

            return EncodeToLatent(augmented, sampleMode: false);
        }

        return EncodeToLatent(image, sampleMode: false);
    }

    /// <summary>
    /// Creates a motion embedding from the motion bucket ID and FPS.
    /// </summary>
    /// <param name="motionBucketId">The motion intensity.</param>
    /// <param name="fps">Frames per second.</param>
    /// <returns>Motion embedding tensor.</returns>
    protected virtual Tensor<T> CreateMotionEmbedding(int motionBucketId, int fps)
    {
        // Create a simple motion embedding combining bucket ID and FPS
        var embedding = new Tensor<T>(new[] { 1, 2 });
        var span = embedding.AsWritableSpan();

        span[0] = NumOps.FromDouble(motionBucketId / 255.0);
        span[1] = NumOps.FromDouble(fps / 30.0);

        return embedding;
    }

    /// <summary>
    /// Predicts noise for video frames conditioned on image and motion.
    /// </summary>
    /// <param name="latents">Current video latents.</param>
    /// <param name="timestep">Current timestep.</param>
    /// <param name="imageEmbedding">Conditioning image embedding.</param>
    /// <param name="motionEmbedding">Motion embedding.</param>
    /// <returns>Predicted noise for all frames.</returns>
    protected abstract Tensor<T> PredictVideoNoise(
        Tensor<T> latents,
        int timestep,
        Tensor<T> imageEmbedding,
        Tensor<T> motionEmbedding);

    /// <summary>
    /// Predicts noise for video frames conditioned on text.
    /// </summary>
    /// <param name="latents">Current video latents.</param>
    /// <param name="timestep">Current timestep.</param>
    /// <param name="textEmbedding">Text embedding.</param>
    /// <returns>Predicted noise for all frames.</returns>
    protected virtual Tensor<T> PredictVideoNoiseWithText(
        Tensor<T> latents,
        int timestep,
        Tensor<T> textEmbedding)
    {
        // Default: predict noise for each frame independently
        var videoShape = latents.Shape;
        var numFrames = videoShape[1];
        var result = new Tensor<T>(videoShape);

        for (int f = 0; f < numFrames; f++)
        {
            var frame = ExtractFrameLatent(latents, f);
            var frameNoise = NoisePredictor.PredictNoise(frame, timestep, textEmbedding);
            InsertFrameLatent(result, frameNoise, f);
        }

        return result;
    }

    /// <summary>
    /// Encodes a video to latent space.
    /// </summary>
    /// <param name="video">The video tensor [batch, numFrames, channels, height, width].</param>
    /// <returns>Video latents.</returns>
    protected virtual Tensor<T> EncodeVideoToLatent(Tensor<T> video)
    {
        var videoShape = video.Shape;
        var numFrames = videoShape[1];

        // Use temporal VAE if available, otherwise encode per-frame
        if (TemporalVAE != null)
        {
            return TemporalVAE.Encode(video, sampleMode: true);
        }

        // Encode each frame independently
        var firstFrame = ExtractFrame(video, 0);
        var firstLatent = EncodeToLatent(firstFrame, sampleMode: true);
        var latentShape = firstLatent.Shape;

        var videoLatentShape = new[] { videoShape[0], numFrames, latentShape[1], latentShape[2], latentShape[3] };
        var videoLatents = new Tensor<T>(videoLatentShape);

        InsertFrameLatent(videoLatents, firstLatent, 0);

        for (int f = 1; f < numFrames; f++)
        {
            var frame = ExtractFrame(video, f);
            var latent = EncodeToLatent(frame, sampleMode: true);
            InsertFrameLatent(videoLatents, latent, f);
        }

        return videoLatents;
    }

    /// <summary>
    /// Decodes video latents to frames.
    /// </summary>
    /// <param name="latents">Video latents [batch, numFrames, latentChannels, height, width].</param>
    /// <returns>Decoded video [batch, numFrames, channels, height, width].</returns>
    protected virtual Tensor<T> DecodeVideoLatents(Tensor<T> latents)
    {
        var latentShape = latents.Shape;
        var numFrames = latentShape[1];

        // Use temporal VAE if available
        if (TemporalVAE != null)
        {
            return TemporalVAE.Decode(latents);
        }

        // Decode each frame independently
        var frames = new Tensor<T>[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            var frameLatent = ExtractFrameLatent(latents, f);
            frames[f] = DecodeFromLatent(frameLatent);
        }

        return FramesToVideo(frames);
    }

    /// <summary>
    /// Adds noise to video latents at a specific timestep.
    /// </summary>
    /// <param name="latents">The original latents.</param>
    /// <param name="timestep">The timestep for noise level.</param>
    /// <param name="rng">Random number generator.</param>
    /// <returns>Noisy latents.</returns>
    protected virtual Tensor<T> AddNoiseToVideoLatents(Tensor<T> latents, int timestep, Random rng)
    {
        var noise = DiffusionNoiseHelper<T>.SampleGaussian(latents.Shape, rng);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), timestep);
        return new Tensor<T>(latents.Shape, noisyLatents);
    }

    /// <summary>
    /// Performs a scheduler step for video latents.
    /// </summary>
    /// <param name="latents">Current latents.</param>
    /// <param name="noisePrediction">Predicted noise.</param>
    /// <param name="timestep">Current timestep.</param>
    /// <returns>Updated latents.</returns>
    protected virtual Tensor<T> SchedulerStepVideo(Tensor<T> latents, Tensor<T> noisePrediction, int timestep)
    {
        var latentVector = latents.ToVector();
        var noiseVector = noisePrediction.ToVector();
        latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
        return new Tensor<T>(latents.Shape, latentVector);
    }

    /// <summary>
    /// Applies classifier-free guidance to video noise predictions.
    /// </summary>
    protected virtual Tensor<T> ApplyGuidanceVideo(Tensor<T> unconditional, Tensor<T> conditional, double scale)
    {
        return ApplyGuidance(unconditional, conditional, scale);
    }

    /// <summary>
    /// Extracts a single frame's latent from video latents.
    /// </summary>
    protected virtual Tensor<T> ExtractFrameLatent(Tensor<T> videoLatents, int frameIndex)
    {
        var shape = videoLatents.Shape;
        var batchSize = shape[0];
        var channels = shape[2];
        var height = shape[3];
        var width = shape[4];

        var frameShape = new[] { batchSize, channels, height, width };
        var frame = new Tensor<T>(frameShape);
        var frameSpan = frame.AsWritableSpan();
        var videoSpan = videoLatents.AsSpan();

        var frameSize = channels * height * width;
        var numFrames = shape[1];

        for (int b = 0; b < batchSize; b++)
        {
            var batchOffset = b * numFrames * frameSize;
            var srcOffset = batchOffset + frameIndex * frameSize;
            var dstOffset = b * frameSize;

            for (int i = 0; i < frameSize; i++)
            {
                frameSpan[dstOffset + i] = videoSpan[srcOffset + i];
            }
        }

        return frame;
    }

    /// <summary>
    /// Inserts a frame latent into video latents at the specified index.
    /// </summary>
    protected virtual void InsertFrameLatent(Tensor<T> videoLatents, Tensor<T> frameLatent, int frameIndex)
    {
        var shape = videoLatents.Shape;
        var batchSize = shape[0];
        var channels = shape[2];
        var height = shape[3];
        var width = shape[4];

        var frameSize = channels * height * width;
        var numFrames = shape[1];
        var videoSpan = videoLatents.AsWritableSpan();
        var frameSpan = frameLatent.AsSpan();

        for (int b = 0; b < batchSize; b++)
        {
            var batchOffset = b * numFrames * frameSize;
            var dstOffset = batchOffset + frameIndex * frameSize;
            var srcOffset = b * frameSize;

            for (int i = 0; i < frameSize; i++)
            {
                videoSpan[dstOffset + i] = frameSpan[srcOffset + i];
            }
        }
    }

    #endregion

    #region Frame Interpolation Methods

    /// <summary>
    /// Interpolates frames using diffusion-based method.
    /// </summary>
    protected virtual Tensor<T> InterpolateFramesDiffusion(Tensor<T> video, int targetFrames)
    {
        // Use diffusion to generate intermediate frames
        // This is a simplified implementation
        var videoShape = video.Shape;
        var currentFrames = videoShape[1];

        var interpolatedFrames = new List<Tensor<T>>();
        var ratio = (double)(currentFrames - 1) / (targetFrames - 1);

        for (int i = 0; i < targetFrames; i++)
        {
            var srcIdx = i * ratio;
            var frame0Idx = (int)Math.Floor(srcIdx);
            var frame1Idx = Math.Min(frame0Idx + 1, currentFrames - 1);
            var t = srcIdx - frame0Idx;

            var frame0 = ExtractFrame(video, frame0Idx);
            var frame1 = ExtractFrame(video, frame1Idx);

            // For diffusion interpolation, use SLERP in latent space
            var latent0 = EncodeToLatent(frame0, sampleMode: false);
            var latent1 = EncodeToLatent(frame1, sampleMode: false);
            var interpolated = DiffusionNoiseHelper<T>.SlerpNoise(latent0, latent1, t);
            interpolatedFrames.Add(DecodeFromLatent(interpolated));
        }

        return FramesToVideo(interpolatedFrames.ToArray());
    }

    /// <summary>
    /// Interpolates frames using optical flow (simplified).
    /// </summary>
    protected virtual Tensor<T> InterpolateFramesOpticalFlow(Tensor<T> video, int targetFrames)
    {
        // Simplified: fall back to blend for now
        return InterpolateFramesBlend(video, targetFrames);
    }

    /// <summary>
    /// Interpolates frames using linear interpolation.
    /// </summary>
    protected virtual Tensor<T> InterpolateFramesLinear(Tensor<T> video, int targetFrames)
    {
        var videoShape = video.Shape;
        var currentFrames = videoShape[1];

        var interpolatedFrames = new List<Tensor<T>>();
        var ratio = (double)(currentFrames - 1) / (targetFrames - 1);

        for (int i = 0; i < targetFrames; i++)
        {
            var srcIdx = i * ratio;
            var frame0Idx = (int)Math.Floor(srcIdx);
            var frame1Idx = Math.Min(frame0Idx + 1, currentFrames - 1);
            var t = srcIdx - frame0Idx;

            var frame0 = ExtractFrame(video, frame0Idx);
            var frame1 = ExtractFrame(video, frame1Idx);

            var interpolated = LinearBlend(frame0, frame1, t);
            interpolatedFrames.Add(interpolated);
        }

        return FramesToVideo(interpolatedFrames.ToArray());
    }

    /// <summary>
    /// Interpolates frames using blend method.
    /// </summary>
    protected virtual Tensor<T> InterpolateFramesBlend(Tensor<T> video, int targetFrames)
    {
        return InterpolateFramesLinear(video, targetFrames);
    }

    /// <summary>
    /// Linearly blends two frames.
    /// </summary>
    protected virtual Tensor<T> LinearBlend(Tensor<T> frame0, Tensor<T> frame1, double t)
    {
        var result = new Tensor<T>(frame0.Shape);
        var span0 = frame0.AsSpan();
        var span1 = frame1.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var tVal = NumOps.FromDouble(t);
        var oneMinusT = NumOps.FromDouble(1.0 - t);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Add(
                NumOps.Multiply(oneMinusT, span0[i]),
                NumOps.Multiply(tVal, span1[i]));
        }

        return result;
    }

    #endregion
}
