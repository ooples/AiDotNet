namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly shuffles the order of color channels.
/// </summary>
/// <remarks>
/// <para>Randomly permutes the RGB channels (e.g., RGB → BRG or GBR). This teaches models
/// to be robust to color channel ordering and can prevent overfitting to specific color patterns.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChannelShuffle<T> : ImageAugmenterBase<T>
{
    public ChannelShuffle(double probability = 0.5) : base(probability) { }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 2) return data.Clone();

        // Generate random permutation
        var indices = new int[data.Channels];
        for (int i = 0; i < data.Channels; i++) indices[i] = i;

        // Fisher-Yates shuffle
        for (int i = data.Channels - 1; i > 0; i--)
        {
            int j = context.GetRandomInt(0, i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                    result.SetPixel(y, x, c, data.GetPixel(y, x, indices[c]));

        return result;
    }
}
