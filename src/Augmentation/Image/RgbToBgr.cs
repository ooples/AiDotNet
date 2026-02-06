namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts between RGB and BGR color channel orderings.
/// </summary>
/// <remarks>
/// <para>
/// RGB and BGR differ only in the order of color channels. RGB (Red, Green, Blue) is the
/// standard for most frameworks (PyTorch, TensorFlow), while BGR (Blue, Green, Red) is
/// used by OpenCV. This transform swaps the first and third channels.
/// </para>
/// <para><b>For Beginners:</b> Different libraries store color channels in different orders.
/// Most deep learning frameworks use RGB, but OpenCV uses BGR. If you loaded an image with
/// OpenCV and want to use it with a PyTorch model, you need to swap the channel order.
/// This transform handles that conversion in both directions.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When loading images with OpenCV (BGR) for use with PyTorch/TF models (RGB)</item>
/// <item>When preprocessing for models trained with BGR input (some older Caffe models)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToBgr<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Creates a new RGB/BGR channel swap.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public RgbToBgr(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Swaps the R and B channels.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB/BGR conversion.");

        var targetColorSpace = data.ColorSpace == ColorSpace.RGB ? ColorSpace.BGR : ColorSpace.RGB;

        var result = new ImageTensor<T>(data.Height, data.Width, data.Channels, data.ChannelOrder, targetColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        for (int y = 0; y < data.Height; y++)
        {
            for (int x = 0; x < data.Width; x++)
            {
                // Swap channel 0 (R/B) and channel 2 (B/R), keep channel 1 (G) and alpha
                result.SetPixel(y, x, 0, data.GetPixel(y, x, 2));
                result.SetPixel(y, x, 1, data.GetPixel(y, x, 1));
                result.SetPixel(y, x, 2, data.GetPixel(y, x, 0));

                // Copy alpha channel if present
                if (data.Channels > 3)
                {
                    for (int c = 3; c < data.Channels; c++)
                    {
                        result.SetPixel(y, x, c, data.GetPixel(y, x, c));
                    }
                }
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
