namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image between RGB and YUV color spaces.
/// </summary>
/// <remarks>
/// <para>
/// YUV separates luminance (Y) from chrominance (U and V). This encoding was originally
/// designed for analog television to allow backwards compatibility with black-and-white sets.
/// The Y channel carries all the brightness information, while U and V carry color.
/// </para>
/// <para><b>For Beginners:</b> YUV splits an image into brightness (Y) and color (U, V).
/// The Y channel alone looks like a grayscale version of the image. This is useful because
/// the human eye is more sensitive to brightness changes than color changes, so you can
/// compress the U and V channels more aggressively.</para>
/// <para><b>Channel layout (BT.601 standard):</b>
/// <list type="bullet">
/// <item><b>Y (Luminance)</b>: [0, 1] - Brightness</item>
/// <item><b>U (Cb)</b>: [-0.436, 0.436] - Blue difference</item>
/// <item><b>V (Cr)</b>: [-0.615, 0.615] - Red difference</item>
/// </list>
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Video processing and JPEG/MPEG encoding</item>
/// <item>Separating luminance for brightness-invariant processing</item>
/// <item>Chroma subsampling applications</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToYuv<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new RGB to YUV conversion.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToYuv(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Converts the image from RGB to YUV color space (BT.601 standard).
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to YUV conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, ColorSpace.YCbCr)
        {
            IsNormalized = data.IsNormalized,
            OriginalRange = data.OriginalRange
        };

        for (int y = 0; y < data.Height; y++)
        {
            for (int x = 0; x < data.Width; x++)
            {
                double r = NumOps.ToDouble(data.GetPixel(y, x, 0));
                double g = NumOps.ToDouble(data.GetPixel(y, x, 1));
                double b = NumOps.ToDouble(data.GetPixel(y, x, 2));

                if (!data.IsNormalized)
                {
                    r /= 255.0;
                    g /= 255.0;
                    b /= 255.0;
                }

                // BT.601 conversion matrix
                double yLum = 0.299 * r + 0.587 * g + 0.114 * b;
                double u = -0.14713 * r - 0.28886 * g + 0.436 * b;
                double v = 0.615 * r - 0.51499 * g - 0.10001 * b;

                result.SetPixel(y, x, 0, NumOps.FromDouble(yLum));
                result.SetPixel(y, x, 1, NumOps.FromDouble(u));
                result.SetPixel(y, x, 2, NumOps.FromDouble(v));

                for (int c = 3; c < data.Channels; c++)
                {
                    result.SetPixel(y, x, c, data.GetPixel(y, x, c));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts YUV values back to RGB.
    /// </summary>
    /// <param name="yLum">Y (luminance) [0, 1].</param>
    /// <param name="u">U (blue difference) [-0.436, 0.436].</param>
    /// <param name="v">V (red difference) [-0.615, 0.615].</param>
    /// <returns>RGB values in [0, 1].</returns>
    public static (double r, double g, double b) YuvToRgb(double yLum, double u, double v)
    {
        double r = yLum + 1.13983 * v;
        double g = yLum - 0.39465 * u - 0.58060 * v;
        double b = yLum + 2.03211 * u;

        return (Math.Max(0, Math.Min(1, r)),
                Math.Max(0, Math.Min(1, g)),
                Math.Max(0, Math.Min(1, b)));
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
