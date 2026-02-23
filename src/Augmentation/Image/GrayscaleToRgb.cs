namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts a grayscale image to RGB by replicating the single channel across all three channels.
/// </summary>
/// <remarks>
/// <para>
/// This creates a 3-channel image from a single-channel grayscale image by duplicating the
/// luminance value into the R, G, and B channels. The resulting image will look identical
/// to the grayscale original but is compatible with models that expect 3-channel input.
/// </para>
/// <para><b>For Beginners:</b> Some models require 3-channel (RGB) input. If you have a
/// grayscale image with only 1 channel, this copies that single channel three times to
/// create a compatible 3-channel image. The image still looks gray, but now has the right
/// number of channels.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When feeding grayscale images to models trained on RGB data</item>
/// <item>When using pretrained models that expect 3-channel input</item>
/// <item>Mixing grayscale and color datasets in the same pipeline</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GrayscaleToRgb<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new grayscale to RGB conversion.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public GrayscaleToRgb(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Converts the grayscale image to RGB.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels == 3 && data.ColorSpace == ColorSpace.RGB)
            return data.Clone();

        if (data.Channels != 1)
            throw new InvalidOperationException(
                $"Expected single-channel grayscale image, got {data.Channels} channels.");

        var result = new ImageTensor<T>(data.Height, data.Width, 3, data.ChannelOrder, ColorSpace.RGB)
        {
            IsNormalized = data.IsNormalized,
            OriginalRange = data.OriginalRange
        };

        for (int y = 0; y < data.Height; y++)
        {
            for (int x = 0; x < data.Width; x++)
            {
                T value = data.GetPixel(y, x, 0);
                result.SetPixel(y, x, 0, value);
                result.SetPixel(y, x, 1, value);
                result.SetPixel(y, x, 2, value);
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
