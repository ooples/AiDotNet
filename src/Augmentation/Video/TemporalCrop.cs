using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Randomly crops a temporal segment from video.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Temporal cropping takes a random consecutive portion of the video.
/// For example, from a 10-second video, it might extract 8 seconds starting at a random point.
/// This helps models learn to recognize actions regardless of when they occur in the video.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Action recognition where action can occur anywhere in the video</item>
/// <item>Training with variable-length videos</item>
/// <item>Reducing overfitting to specific temporal positions</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TemporalCrop<T> : TemporalAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum crop ratio (fraction of original length to keep).
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.7 (keep at least 70% of frames)</para>
    /// </remarks>
    public double MinCropRatio { get; }

    /// <summary>
    /// Gets the maximum crop ratio (fraction of original length to keep).
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.9 (keep at most 90% of frames)</para>
    /// </remarks>
    public double MaxCropRatio { get; }

    /// <summary>
    /// Creates a new temporal crop augmentation.
    /// </summary>
    /// <param name="minCropRatio">Minimum fraction of frames to keep (default: 0.7).</param>
    /// <param name="maxCropRatio">Maximum fraction of frames to keep (default: 0.9).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public TemporalCrop(
        double minCropRatio = 0.7,
        double maxCropRatio = 0.9,
        double probability = 0.5,
        double frameRate = 30.0) : base(probability, frameRate)
    {
        if (minCropRatio <= 0 || minCropRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minCropRatio),
                "Minimum crop ratio must be between 0 and 1.");
        }

        if (maxCropRatio <= 0 || maxCropRatio > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxCropRatio),
                "Maximum crop ratio must be between 0 and 1.");
        }

        if (minCropRatio > maxCropRatio)
        {
            throw new ArgumentException("Minimum crop ratio must be less than or equal to maximum.");
        }

        MinCropRatio = minCropRatio;
        MaxCropRatio = maxCropRatio;
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        int originalFrameCount = GetFrameCount(data);
        if (originalFrameCount <= 1) return data;

        // Determine crop ratio and resulting frame count
        double cropRatio = context.GetRandomDouble(MinCropRatio, MaxCropRatio);
        int cropFrameCount = Math.Max(1, (int)(originalFrameCount * cropRatio));

        // Determine random start position
        int maxStart = originalFrameCount - cropFrameCount;
        int startFrame = maxStart > 0 ? context.Random.Next(maxStart + 1) : 0;

        // Extract the crop
        var result = new ImageTensor<T>[cropFrameCount];
        for (int i = 0; i < cropFrameCount; i++)
        {
            result[i] = data[startFrame + i];
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minCropRatio"] = MinCropRatio;
        parameters["maxCropRatio"] = MaxCropRatio;
        return parameters;
    }
}
