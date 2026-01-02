namespace AiDotNet.Video.Interfaces;

/// <summary>
/// Interface for video enhancement models that improve video quality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Video enhancers take degraded or low-quality video and produce improved versions.
/// Common enhancement tasks include super-resolution (upscaling), denoising,
/// stabilization, and frame interpolation.
/// </para>
/// <para>
/// <b>For Beginners:</b> Video enhancement is like photo editing, but for videos.
/// These models can:
/// - Make blurry videos sharper (super-resolution)
/// - Remove grain/noise from old or low-light videos
/// - Smooth out shaky camera footage (stabilization)
/// - Make choppy videos smoother (frame interpolation)
///
/// The enhanced video has the same content but looks much better!
///
/// Example:
/// <code>
/// var enhancer = new VideoSuperResolution&lt;double&gt;(new VideoEnhancementOptions&lt;double&gt;
/// {
///     ScaleFactor = 4  // 4x upscaling
/// });
///
/// // Upscale a 480p video to 1920p (4x larger)
/// var hdVideo = enhancer.Enhance(lowResVideo);
/// </code>
/// </para>
/// </remarks>
public interface IVideoEnhancer<T> : IVideoModel<T>
{
    #region Properties

    /// <summary>
    /// Gets the scale factor for spatial enhancement (e.g., 2x, 4x upscaling).
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 1 means no spatial scaling (same resolution output).
    /// A value of 4 means the output is 4x larger in both width and height.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how much bigger the output video will be.
    /// ScaleFactor = 2 means a 720x480 video becomes 1440x960.
    /// ScaleFactor = 4 means a 720x480 video becomes 2880x1920.
    /// </para>
    /// </remarks>
    int ScaleFactor { get; }

    /// <summary>
    /// Gets the temporal scale factor for frame rate enhancement.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A value of 1 means no frame rate change.
    /// A value of 2 means the output has 2x more frames (e.g., 30fps becomes 60fps).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many extra frames will be added.
    /// TemporalScaleFactor = 2 doubles the frame rate for smoother playback.
    /// </para>
    /// </remarks>
    int TemporalScaleFactor { get; }

    /// <summary>
    /// Gets whether this enhancer can process video in real-time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Real-time means the model can process frames fast enough to keep up with
    /// live video playback (typically 24-30 frames per second).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If this is true, you can use the enhancer for live video.
    /// If false, it's better suited for batch processing of recorded videos.
    /// </para>
    /// </remarks>
    bool SupportsRealTime { get; }

    /// <summary>
    /// Gets the type of enhancement this model performs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different enhancers specialize in different improvements.
    /// Check this to know what kind of enhancement the model will apply.
    /// </para>
    /// </remarks>
    VideoEnhancementType EnhancementType { get; }

    #endregion

    #region Enhancement Methods

    /// <summary>
    /// Enhances a video and returns the improved version.
    /// </summary>
    /// <param name="inputVideo">The video to enhance [batch, numFrames, channels, height, width].</param>
    /// <returns>The enhanced video tensor.</returns>
    /// <remarks>
    /// <para>
    /// The output shape depends on the enhancement type:
    /// - Super-resolution: [batch, numFrames, channels, height * scale, width * scale]
    /// - Frame interpolation: [batch, numFrames * temporalScale, channels, height, width]
    /// - Denoising/stabilization: Same shape as input
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main method to improve your video.
    /// Just pass in your video and get back an enhanced version!
    /// </para>
    /// </remarks>
    Tensor<T> Enhance(Tensor<T> inputVideo);

    /// <summary>
    /// Enhances a video asynchronously with progress reporting.
    /// </summary>
    /// <param name="inputVideo">The video to enhance.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A task containing the enhanced video tensor.</returns>
    Task<Tensor<T>> EnhanceAsync(
        Tensor<T> inputVideo,
        IProgress<EnhancementProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Enhances a single frame from a video.
    /// </summary>
    /// <param name="frame">A single frame [batch, channels, height, width].</param>
    /// <returns>The enhanced frame.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for real-time processing or when you only
    /// need to enhance individual frames. Note that some enhancements (like
    /// frame interpolation) require multiple frames and won't work with this method.
    /// </para>
    /// </remarks>
    Tensor<T> EnhanceFrame(Tensor<T> frame);

    /// <summary>
    /// Enhances a video using a sliding window approach for memory efficiency.
    /// </summary>
    /// <param name="inputVideo">The video to enhance.</param>
    /// <param name="windowSize">Number of frames to process at once.</param>
    /// <param name="overlap">Number of overlapping frames between windows.</param>
    /// <returns>The enhanced video.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Long videos might not fit in GPU memory. This method
    /// processes the video in chunks (windows), which uses less memory but may
    /// take longer. The overlap helps avoid visible seams between chunks.
    /// </para>
    /// </remarks>
    Tensor<T> EnhanceWithSlidingWindow(Tensor<T> inputVideo, int windowSize = 16, int overlap = 4);

    #endregion

    #region Quality Metrics

    /// <summary>
    /// Computes the Peak Signal-to-Noise Ratio between original and enhanced video.
    /// </summary>
    /// <param name="original">The original video.</param>
    /// <param name="enhanced">The enhanced video.</param>
    /// <returns>The PSNR value in decibels (higher is better).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PSNR measures how much the enhanced video differs from
    /// the original. For super-resolution, you'd compare against a known high-res
    /// reference. Values above 30 dB generally indicate good quality.
    /// </para>
    /// </remarks>
    T ComputePSNR(Tensor<T> original, Tensor<T> enhanced);

    /// <summary>
    /// Computes the Structural Similarity Index between original and enhanced video.
    /// </summary>
    /// <param name="original">The original video.</param>
    /// <param name="enhanced">The enhanced video.</param>
    /// <returns>The SSIM value (0-1, higher is better).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SSIM measures how similar two videos look to human eyes.
    /// Unlike PSNR, it considers structure, luminance, and contrast. Values above
    /// 0.9 indicate very high similarity.
    /// </para>
    /// </remarks>
    T ComputeSSIM(Tensor<T> original, Tensor<T> enhanced);

    #endregion
}

/// <summary>
/// Types of video enhancement operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Each type represents a different way to improve video quality.
/// Some enhancers may support multiple types.
/// </para>
/// </remarks>
public enum VideoEnhancementType
{
    /// <summary>
    /// Increases video resolution (e.g., 480p to 4K).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Super-resolution uses AI to add detail that wasn't in the original video.
    /// It's like making a blurry photo sharp by "imagining" the missing details.
    /// </para>
    /// </remarks>
    SuperResolution,

    /// <summary>
    /// Removes noise and grain from video.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Denoising cleans up video shot in low light or with high ISO settings.
    /// The AI learns to distinguish real detail from random noise.
    /// </para>
    /// </remarks>
    Denoising,

    /// <summary>
    /// Removes camera shake and stabilizes footage.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stabilization smooths out shaky handheld footage.
    /// The AI predicts a smooth camera path and warps frames to match.
    /// </para>
    /// </remarks>
    Stabilization,

    /// <summary>
    /// Increases frame rate by generating intermediate frames.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Frame interpolation makes choppy video smoother (e.g., 24fps to 60fps).
    /// The AI predicts what frames between existing frames should look like.
    /// </para>
    /// </remarks>
    FrameInterpolation,

    /// <summary>
    /// Estimates motion between frames (optical flow).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Optical flow computes how each pixel moves between frames.
    /// This is useful for motion analysis, video editing, and as input to other enhancements.
    /// </para>
    /// </remarks>
    OpticalFlow,

    /// <summary>
    /// General quality enhancement (combination of techniques).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Some models combine multiple enhancement techniques (denoising + sharpening + color correction)
    /// into a single operation for overall quality improvement.
    /// </para>
    /// </remarks>
    GeneralEnhancement
}

/// <summary>
/// Reports progress during video enhancement operations.
/// </summary>
public class EnhancementProgress
{
    /// <summary>
    /// Gets or sets the current frame being processed.
    /// </summary>
    public int CurrentFrame { get; set; }

    /// <summary>
    /// Gets or sets the total number of frames to process.
    /// </summary>
    public int TotalFrames { get; set; }

    /// <summary>
    /// Gets or sets the current processing stage description.
    /// </summary>
    public string? Stage { get; set; }

    /// <summary>
    /// Gets the progress as a percentage (0-100).
    /// </summary>
    public double ProgressPercentage => TotalFrames > 0 ? (double)CurrentFrame / TotalFrames * 100 : 0;
}
