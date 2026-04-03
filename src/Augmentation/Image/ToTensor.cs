namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an image to a normalized tensor with values in [0, 1].
/// </summary>
/// <remarks>
/// <para>
/// ToTensor converts pixel values from [0, 255] integer range to [0, 1] floating point range
/// by dividing by 255. This is the standard first step in most image preprocessing pipelines,
/// equivalent to <c>torchvision.transforms.ToTensor()</c>.
/// </para>
/// <para><b>For Beginners:</b> Digital images store colors as numbers from 0 to 255.
/// Neural networks prefer small numbers close to zero, so we divide by 255 to get
/// values between 0 and 1. This is almost always the first thing you do to an image
/// before feeding it to a model.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>As the first preprocessing step for images with [0, 255] values</item>
/// <item>Before applying normalization (Normalize expects [0, 1] input)</item>
/// </list>
/// </para>
/// <para><b>When NOT to use:</b>
/// <list type="bullet">
/// <item>If values are already in [0, 1] range</item>
/// <item>If the image was loaded as float (many frameworks do this automatically)</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ToTensor<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the scale factor used for conversion. Default is 1/255.
    /// </summary>
    public double ScaleFactor { get; }

    /// <summary>
    /// Creates a new ToTensor augmentation.
    /// </summary>
    /// <param name="scaleFactor">
    /// The scale factor to divide pixel values by. Default is 255.0 (maps [0,255] to [0,1]).
    /// </param>
    /// <param name="probability">The probability of applying this augmentation. Default is 1.0.</param>
    public ToTensor(double scaleFactor = 255.0, double probability = 1.0)
        : base(probability)
    {
        if (scaleFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be positive.");

        ScaleFactor = scaleFactor;
    }

    /// <summary>
    /// Converts pixel values to [0, 1] range.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.IsNormalized)
            return data.Clone();

        var result = data.Clone();
        int height = data.Height;
        int width = data.Width;
        int channels = data.Channels;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < channels; c++)
                {
                    double value = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double scaled = value / ScaleFactor;
                    scaled = Math.Max(0.0, Math.Min(1.0, scaled));
                    result.SetPixel(y, x, c, NumOps.FromDouble(scaled));
                }
            }
        }

        result.IsNormalized = true;

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["scale_factor"] = ScaleFactor;
        return parameters;
    }
}
