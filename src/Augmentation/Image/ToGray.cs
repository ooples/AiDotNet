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

        var image = data.ToPlanarTensor();
        var gray = ImageTensor<T>.MixChannels(image, w1, w2, w3);
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image, gray, gray, gray));

        return result;
    }
}
