using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Changes the playback speed of video by resampling frames.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Speed change makes video play faster or slower by
/// keeping/skipping/duplicating frames. This simulates different action speeds
/// and helps models recognize actions regardless of how fast they're performed.</para>
/// <para><b>Speed factors:</b>
/// <list type="bullet">
/// <item>2.0 = double speed (skip every other frame)</item>
/// <item>1.0 = normal speed</item>
/// <item>0.5 = half speed (duplicate frames)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpeedChange<T> : TemporalAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum speed factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.8 (20% slower)</para>
    /// </remarks>
    public double MinSpeed { get; }

    /// <summary>
    /// Gets the maximum speed factor.
    /// </summary>
    /// <remarks>
    /// <para>Default: 1.2 (20% faster)</para>
    /// </remarks>
    public double MaxSpeed { get; }

    /// <summary>
    /// Creates a new speed change augmentation.
    /// </summary>
    /// <param name="minSpeed">Minimum speed factor (default: 0.8).</param>
    /// <param name="maxSpeed">Maximum speed factor (default: 1.2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public SpeedChange(
        double minSpeed = 0.8,
        double maxSpeed = 1.2,
        double probability = 0.5,
        double frameRate = 30.0) : base(probability, frameRate)
    {
        if (minSpeed <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(minSpeed),
                "Minimum speed must be positive.");
        }

        if (maxSpeed <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSpeed),
                "Maximum speed must be positive.");
        }

        if (minSpeed > maxSpeed)
        {
            throw new ArgumentException("Minimum speed must be less than or equal to maximum speed.");
        }

        MinSpeed = minSpeed;
        MaxSpeed = maxSpeed;
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        int originalFrameCount = GetFrameCount(data);
        if (originalFrameCount <= 1) return data;

        double speed = context.GetRandomDouble(MinSpeed, MaxSpeed);

        // New frame count after speed change
        // Faster = fewer frames, slower = more frames
        int newFrameCount = Math.Max(1, (int)(originalFrameCount / speed));

        var result = new ImageTensor<T>[newFrameCount];

        for (int i = 0; i < newFrameCount; i++)
        {
            // Map new frame index to original frame index
            double srcPos = (double)i * originalFrameCount / newFrameCount;
            int srcIndex = (int)Math.Round(srcPos);
            srcIndex = Math.Min(srcIndex, originalFrameCount - 1);

            result[i] = data[srcIndex];
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSpeed"] = MinSpeed;
        parameters["maxSpeed"] = MaxSpeed;
        return parameters;
    }
}
