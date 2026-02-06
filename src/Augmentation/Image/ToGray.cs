namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts to grayscale with random channel weights, outputting 3 channels.
/// </summary>
/// <remarks>
/// <para>Unlike RgbToGrayscale which uses fixed weights and can output 1 channel, ToGray
/// uses random weights for the RGB channels and always outputs 3 channels (grayscale
/// replicated). This adds stochastic color invariance as augmentation.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ToGray<T> : ImageAugmenterBase<T>
{
    public ToGray(double probability = 0.5) : base(probability) { }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        // Random weights that sum to 1
        double w1 = context.GetRandomDouble(0, 1);
        double w2 = context.GetRandomDouble(0, 1 - w1);
        double w3 = 1.0 - w1 - w2;

        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double r = NumOps.ToDouble(data.GetPixel(y, x, 0));
                double g = NumOps.ToDouble(data.GetPixel(y, x, 1));
                double b = NumOps.ToDouble(data.GetPixel(y, x, 2));
                double gray = r * w1 + g * w2 + b * w3;

                result.SetPixel(y, x, 0, NumOps.FromDouble(gray));
                result.SetPixel(y, x, 1, NumOps.FromDouble(gray));
                result.SetPixel(y, x, 2, NumOps.FromDouble(gray));
            }

        return result;
    }
}
