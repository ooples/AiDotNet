namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for video diffusion models that generate temporal sequences.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video diffusion models extend image diffusion to handle the temporal dimension,
/// generating coherent video sequences. They model both spatial (within-frame) and
/// temporal (across-frame) dependencies.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video diffusion is like image diffusion, but it creates videos
/// instead of single images. The main challenge is making the frames look consistent
/// over time (no flickering or teleporting objects).
///
/// How video diffusion works:
/// 1. The model generates multiple frames at once (typically 14-25 frames)
/// 2. Special "temporal attention" ensures frames are consistent
/// 3. The model can be conditioned on a starting image, text, or both
///
/// Common approaches:
/// - Image-to-Video (SVD): Start from an image, generate motion
/// - Text-to-Video (VideoCrafter): Generate video from text description
/// - Video-to-Video: Transform existing video with new style/content
///
/// Key challenges solved by these models:
/// - Temporal consistency (no flickering)
/// - Motion coherence (objects move naturally)
/// - Long-range dependencies (beginning and end are related)
/// </para>
/// <para>
/// This interface extends <see cref="IDiffusionModel{T}"/> with video-specific operations.
/// </para>
/// </remarks>
public interface IVideoDiffusionModel<T> : IDiffusionModel<T>
{
    /// <summary>
    /// Gets the default number of frames generated.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typical values: 14, 16, 25 frames. Limited by GPU memory.
    /// </para>
    /// </remarks>
    int DefaultNumFrames { get; }

    /// <summary>
    /// Gets the default frames per second for generated videos.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Typical values: 7 FPS for SVD, 8 FPS for AnimateDiff.
    /// Lower FPS = smoother but slower apparent motion.
    /// </para>
    /// </remarks>
    int DefaultFPS { get; }

    /// <summary>
    /// Gets whether this model supports image-to-video generation.
    /// </summary>
    bool SupportsImageToVideo { get; }

    /// <summary>
    /// Gets whether this model supports text-to-video generation.
    /// </summary>
    bool SupportsTextToVideo { get; }

    /// <summary>
    /// Gets whether this model supports video-to-video transformation.
    /// </summary>
    bool SupportsVideoToVideo { get; }

    /// <summary>
    /// Gets the motion bucket ID for controlling motion intensity (SVD-specific).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Controls amount of motion in generated video.
    /// Lower values = less motion, higher values = more motion.
    /// Range: 1-255, default: 127.
    /// </para>
    /// </remarks>
    int MotionBucketId { get; }

    /// <summary>
    /// Generates a video from a conditioning image.
    /// </summary>
    /// <param name="inputImage">The conditioning image [batch, channels, height, width].</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="fps">Target frames per second.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="motionBucketId">Motion intensity (1-255).</param>
    /// <param name="noiseAugStrength">Noise augmentation for input image.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video tensor [batch, numFrames, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This animates a still image:
    /// - Input: A single image (photo, artwork, etc.)
    /// - Output: A video where the scene comes to life
    ///
    /// Tips:
    /// - motionBucketId controls how much movement happens
    /// - noiseAugStrength slightly varies the input to encourage motion
    /// - Higher inference steps = smoother motion but slower
    /// </para>
    /// </remarks>
    Tensor<T> GenerateFromImage(
        Tensor<T> inputImage,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 25,
        int? motionBucketId = null,
        double noiseAugStrength = 0.02,
        int? seed = null);

    /// <summary>
    /// Generates a video from a text prompt.
    /// </summary>
    /// <param name="prompt">Text description of the video to generate.</param>
    /// <param name="negativePrompt">What to avoid in the video.</param>
    /// <param name="width">Video width in pixels.</param>
    /// <param name="height">Video height in pixels.</param>
    /// <param name="numFrames">Number of frames to generate.</param>
    /// <param name="fps">Target frames per second.</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Generated video tensor [batch, numFrames, channels, height, width].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a video from a description:
    /// - prompt: What you want ("a dog running on a beach")
    /// - The model generates both the visual content and the motion
    /// </para>
    /// </remarks>
    Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        int width = 512,
        int height = 512,
        int? numFrames = null,
        int? fps = null,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        int? seed = null);

    /// <summary>
    /// Transforms an existing video.
    /// </summary>
    /// <param name="inputVideo">The input video [batch, numFrames, channels, height, width].</param>
    /// <param name="prompt">Text prompt describing the transformation.</param>
    /// <param name="negativePrompt">What to avoid.</param>
    /// <param name="strength">Transformation strength (0.0-1.0).</param>
    /// <param name="numInferenceSteps">Number of denoising steps.</param>
    /// <param name="guidanceScale">Classifier-free guidance scale.</param>
    /// <param name="seed">Optional random seed.</param>
    /// <returns>Transformed video tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This changes an existing video's style or content:
    /// - strength=0.3: Minor style changes, motion preserved
    /// - strength=0.7: Major changes, but timing preserved
    /// - strength=1.0: Complete regeneration guided by original
    /// </para>
    /// </remarks>
    Tensor<T> VideoToVideo(
        Tensor<T> inputVideo,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.7,
        int numInferenceSteps = 50,
        double guidanceScale = 7.5,
        int? seed = null);

    /// <summary>
    /// Interpolates between frames to increase frame rate.
    /// </summary>
    /// <param name="video">The input video [batch, numFrames, channels, height, width].</param>
    /// <param name="targetFPS">Target frame rate.</param>
    /// <param name="interpolationMethod">Method for frame interpolation.</param>
    /// <returns>Interpolated video with more frames.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This makes videos smoother by adding in-between frames:
    /// - Input: 7 FPS video (a bit choppy)
    /// - Output: 30 FPS video (smooth playback)
    /// The AI figures out what the in-between frames should look like.
    /// </para>
    /// </remarks>
    Tensor<T> InterpolateFrames(
        Tensor<T> video,
        int targetFPS,
        FrameInterpolationMethod interpolationMethod = FrameInterpolationMethod.Diffusion);

    /// <summary>
    /// Sets the motion intensity for generation.
    /// </summary>
    /// <param name="bucketId">Motion bucket ID (1-255).</param>
    void SetMotionBucketId(int bucketId);

    /// <summary>
    /// Extracts a frame from the video tensor.
    /// </summary>
    /// <param name="video">The video tensor [batch, numFrames, channels, height, width].</param>
    /// <param name="frameIndex">Index of the frame to extract.</param>
    /// <returns>The frame as an image tensor [batch, channels, height, width].</returns>
    Tensor<T> ExtractFrame(Tensor<T> video, int frameIndex);

    /// <summary>
    /// Concatenates frames into a video tensor.
    /// </summary>
    /// <param name="frames">Array of frame tensors [batch, channels, height, width].</param>
    /// <returns>Video tensor [batch, numFrames, channels, height, width].</returns>
    Tensor<T> FramesToVideo(Tensor<T>[] frames);
}

/// <summary>
/// Methods for interpolating between video frames.
/// </summary>
public enum FrameInterpolationMethod
{
    /// <summary>
    /// Use diffusion-based interpolation (highest quality, slowest).
    /// </summary>
    Diffusion,

    /// <summary>
    /// Use optical flow-based interpolation (good quality, fast).
    /// </summary>
    OpticalFlow,

    /// <summary>
    /// Simple linear interpolation (fast, lower quality).
    /// </summary>
    Linear,

    /// <summary>
    /// Blend-based interpolation (fast, reasonable quality).
    /// </summary>
    Blend
}
