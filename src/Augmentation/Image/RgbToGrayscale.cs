namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Converts an RGB image to grayscale using configurable channel weights.
/// </summary>
/// <remarks>
/// <para>
/// Grayscale conversion reduces a 3-channel RGB image to a single luminance channel
/// using weighted combination of the color channels. The default weights (0.2989, 0.5870, 0.1140)
/// follow the ITU-R BT.601 standard, which accounts for human perception of brightness
/// (green appears brightest, blue appears darkest).
/// </para>
/// <para><b>For Beginners:</b> Converting to grayscale removes color information, keeping
/// only brightness. This is useful when color doesn't matter for your task (like reading
/// text or detecting edges) and reduces computation by working with 1 channel instead of 3.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Document analysis and OCR</item>
/// <item>Edge detection and structural analysis</item>
/// <item>Reducing model size when color is not informative</item>
/// <item>As a data augmentation to teach color invariance</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RgbToGrayscale<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the weight for the red channel.
    /// </summary>
    public double RedWeight { get; }

    /// <summary>
    /// Gets the weight for the green channel.
    /// </summary>
    public double GreenWeight { get; }

    /// <summary>
    /// Gets the weight for the blue channel.
    /// </summary>
    public double BlueWeight { get; }

    /// <summary>
    /// Gets the number of output channels (1 or 3).
    /// When 3, the grayscale value is replicated across all channels.
    /// </summary>
    public int OutputChannels { get; }

    /// <summary>
    /// Creates a new RGB to grayscale conversion.
    /// </summary>
    /// <param name="redWeight">Weight for red channel. Default is 0.2989 (ITU-R BT.601).</param>
    /// <param name="greenWeight">Weight for green channel. Default is 0.5870 (ITU-R BT.601).</param>
    /// <param name="blueWeight">Weight for blue channel. Default is 0.1140 (ITU-R BT.601).</param>
    /// <param name="outputChannels">Number of output channels (1 or 3). Default is 1.</param>
    /// <param name="probability">The probability of applying this augmentation.</param>
    public RgbToGrayscale(
        double redWeight = 0.2989,
        double greenWeight = 0.5870,
        double blueWeight = 0.1140,
        int outputChannels = 1,
        double probability = 1.0) : base(probability)
    {
        if (outputChannels != 1 && outputChannels != 3)
            throw new ArgumentException("Output channels must be 1 or 3.", nameof(outputChannels));

        RedWeight = redWeight;
        GreenWeight = greenWeight;
        BlueWeight = blueWeight;
        OutputChannels = outputChannels;
    }

    /// <summary>
    /// Converts the RGB image to grayscale.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.ColorSpace == ColorSpace.Grayscale)
            return data.Clone();

        if (data.Channels < 3)
            throw new InvalidOperationException("Image must have at least 3 channels for RGB to grayscale conversion.");

        var result = new ImageTensor<T>(data.Height, data.Width, OutputChannels, data.ChannelOrder,
            OutputChannels == 1 ? ColorSpace.Grayscale : data.ColorSpace)
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
                double gray = r * RedWeight + g * GreenWeight + b * BlueWeight;

                if (OutputChannels == 1)
                {
                    result.SetPixel(y, x, 0, NumOps.FromDouble(gray));
                }
                else
                {
                    result.SetPixel(y, x, 0, NumOps.FromDouble(gray));
                    result.SetPixel(y, x, 1, NumOps.FromDouble(gray));
                    result.SetPixel(y, x, 2, NumOps.FromDouble(gray));
                }
            }
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["red_weight"] = RedWeight;
        parameters["green_weight"] = GreenWeight;
        parameters["blue_weight"] = BlueWeight;
        parameters["output_channels"] = OutputChannels;
        return parameters;
    }
}
