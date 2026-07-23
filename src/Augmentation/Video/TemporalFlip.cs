using AiDotNet.Augmentation.Image;

namespace AiDotNet.Augmentation.Video;

/// <summary>
/// Reverses the order of frames in a video.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Temporal flip plays the video backwards by reversing
/// the frame order. This can help models become invariant to action direction,
/// though it should be used carefully as some actions are not time-reversible.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Symmetric actions (walking back and forth, waving)</item>
/// <item>Scene recognition where temporal direction doesn't matter</item>
/// <item>Data augmentation for limited video datasets</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>Actions with clear direction (falling down vs. jumping up)</item>
/// <item>Tasks involving causality or temporal reasoning</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TemporalFlip<T> : TemporalAugmenterBase<T>
{
    /// <summary>
    /// Creates a new temporal flip augmentation.
    /// </summary>
    /// <param name="probability">Probability of applying this augmentation (default: 0.3).</param>
    /// <param name="frameRate">Frame rate of the video (default: 30).</param>
    public TemporalFlip(
        double probability = 0.3,
        double frameRate = 30.0) : base(probability, frameRate)
    {
    }

    /// <inheritdoc />
    protected override ImageTensor<T>[] ApplyAugmentation(ImageTensor<T>[] data, AugmentationContext<T> context)
    {
        int frameCount = GetFrameCount(data);
        if (frameCount <= 1) return data;

        // Reverse the frame order
        var result = new ImageTensor<T>[frameCount];
        for (int i = 0; i < frameCount; i++)
        {
            result[i] = data[frameCount - 1 - i];
        }

        return result;
    }
}
