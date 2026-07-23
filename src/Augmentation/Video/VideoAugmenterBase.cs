using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Base class for video data augmentations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Video augmentation transforms sequences of frames to improve
/// model robustness to temporal variations. It combines:
/// <list type="bullet">
/// <item>Temporal augmentations (time-based): cropping, reversing, speed changes</item>
/// <item>Spatial augmentations (frame-based): flips, rotations, color changes applied consistently across frames</item>
/// </list>
/// </para>
/// <para>Video data is represented as an array of ImageTensor frames.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class VideoAugmenterBase<T> : AugmentationBase<T, ImageTensor<T>[]>
{
    /// <summary>
    /// Gets or sets the frame rate of the video in frames per second.
    /// </summary>
    /// <remarks>
    /// <para>Default: 30 fps (common for most video)</para>
    /// <para>Other common values: 24 fps (film), 25 fps (PAL), 60 fps (high frame rate)</para>
    /// </remarks>
    public double FrameRate { get; set; } = 30.0;

    /// <summary>
    /// Initializes a new video augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    /// <param name="frameRate">The frame rate of the video in fps.</param>
    protected VideoAugmenterBase(double probability = 1.0, double frameRate = 30.0) : base(probability)
    {
        FrameRate = frameRate;
    }

    /// <summary>
    /// Gets the duration of the video in seconds.
    /// </summary>
    /// <param name="frames">The video frames.</param>
    /// <returns>The duration in seconds.</returns>
    protected double GetDuration(ImageTensor<T>[] frames)
    {
        return frames.Length / FrameRate;
    }

    /// <summary>
    /// Gets the number of frames in the video.
    /// </summary>
    /// <param name="frames">The video frames.</param>
    /// <returns>The frame count.</returns>
    protected int GetFrameCount(ImageTensor<T>[] frames)
    {
        return frames.Length;
    }

    /// <summary>
    /// Validates that all frames have the same dimensions.
    /// </summary>
    /// <param name="frames">The video frames to validate.</param>
    /// <exception cref="ArgumentException">Thrown if frames have inconsistent dimensions.</exception>
    protected void ValidateFrameDimensions(ImageTensor<T>[] frames)
    {
        if (frames.Length == 0) return;

        var firstFrame = frames[0];
        var height = firstFrame.Height;
        var width = firstFrame.Width;
        var channels = firstFrame.Channels;

        for (int i = 1; i < frames.Length; i++)
        {
            if (frames[i].Height != height || frames[i].Width != width || frames[i].Channels != channels)
            {
                throw new ArgumentException($"Frame {i} has different dimensions than frame 0. " +
                    $"Expected ({height}, {width}, {channels}), got ({frames[i].Height}, {frames[i].Width}, {frames[i].Channels}).");
            }
        }
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["frameRate"] = FrameRate;
        return parameters;
    }
}

/// <summary>
/// Base class for temporal video augmentations that modify the frame sequence.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class TemporalAugmenterBase<T> : VideoAugmenterBase<T>
{
    /// <summary>
    /// Initializes a new temporal augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <param name="frameRate">The frame rate of the video.</param>
    protected TemporalAugmenterBase(double probability = 1.0, double frameRate = 30.0)
        : base(probability, frameRate)
    {
    }
}

/// <summary>
/// Base class for spatial video augmentations that apply image transforms to all frames.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public abstract class SpatialVideoAugmenterBase<T> : VideoAugmenterBase<T>
{
    /// <summary>
    /// Gets or sets whether to use consistent random parameters across all frames.
    /// </summary>
    /// <remarks>
    /// <para>Default: true</para>
    /// <para>When true, the same random transformation is applied to all frames.
    /// When false, each frame gets different random parameters (rarely desired for video).</para>
    /// </remarks>
    public bool ConsistentAcrossFrames { get; set; } = true;

    /// <summary>
    /// Initializes a new spatial video augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <param name="frameRate">The frame rate of the video.</param>
    protected SpatialVideoAugmenterBase(double probability = 1.0, double frameRate = 30.0)
        : base(probability, frameRate)
    {
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["consistentAcrossFrames"] = ConsistentAcrossFrames;
        return parameters;
    }
}
