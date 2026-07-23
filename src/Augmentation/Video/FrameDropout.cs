using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Randomly drops frames from video.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Frame dropout randomly removes some frames from the video,
/// simulating frame drops in real-world video capture or network streaming.
/// This helps models become robust to missing frames.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Training for real-world video processing with potential frame drops</item>
/// <item>Reducing overfitting to exact frame sequences</item>
/// <item>Simulating lower frame rate videos</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FrameDropout<T> : TemporalAugmenterBase<T>
{
    /// <summary>
    /// Gets the probability of dropping each frame.
    /// </summary>
    /// <remarks>
    /// <para>Default: 0.1 (10% of frames dropped)</para>
    /// </remarks>
    public double DropoutRate { get; }

    /// <summary>
    /// Gets the minimum number of frames to keep.
    /// </summary>
    /// <remarks>
    /// <para>Default: 2</para>
    /// </remarks>
    public int MinFramesToKeep { get; }

    /// <summary>
    /// Creates a new frame dropout augmentation.
    /// </summary>
    /// <param name="dropoutRate">Probability of dropping each frame (default: 0.1).</param>
    /// <param name="minFramesToKeep">Minimum frames to keep (default: 2).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.3).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public FrameDropout(
        double dropoutRate = 0.1,
        int minFramesToKeep = 2,
        double probability = 0.3,
        double frameRate = 30.0) : base(probability, frameRate)
    {
        if (dropoutRate < 0 || dropoutRate > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(dropoutRate),
                "Dropout rate must be between 0 and 1.");
        }

        if (minFramesToKeep < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(minFramesToKeep),
                "Minimum frames to keep must be at least 1.");
        }

        DropoutRate = dropoutRate;
        MinFramesToKeep = minFramesToKeep;
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        int frameCount = GetFrameCount(data);
        if (frameCount <= MinFramesToKeep) return data;

        var keptFrames = new List<ImageTensor<T>>();

        foreach (var frame in data)
        {
            if (context.Random.NextDouble() >= DropoutRate)
            {
                keptFrames.Add(frame);
            }
        }

        // Ensure minimum frames are kept
        if (keptFrames.Count < MinFramesToKeep)
        {
            // Randomly add back some dropped frames
            var allIndices = Enumerable.Range(0, frameCount).ToList();
            var keptIndices = new HashSet<int>();

            // Mark already kept frames
            for (int i = 0; i < keptFrames.Count; i++)
            {
                // Find original index (simplified - assumes no duplicates)
                for (int j = 0; j < frameCount; j++)
                {
                    if (ReferenceEquals(data[j], keptFrames[i]))
                    {
                        keptIndices.Add(j);
                        break;
                    }
                }
            }

            // Add random frames until minimum
            while (keptIndices.Count < MinFramesToKeep)
            {
                int idx = context.Random.Next(frameCount);
                keptIndices.Add(idx);
            }

            // Rebuild in original order
            keptFrames.Clear();
            foreach (int idx in keptIndices.OrderBy(x => x))
            {
                keptFrames.Add(data[idx]);
            }
        }

        return keptFrames.ToArray();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["dropoutRate"] = DropoutRate;
        parameters["minFramesToKeep"] = MinFramesToKeep;
        return parameters;
    }
}
