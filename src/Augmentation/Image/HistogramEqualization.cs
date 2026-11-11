namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies standard histogram equalization to improve image contrast.
/// </summary>
/// <remarks>
/// <para>
/// Histogram equalization redistributes pixel intensity values to produce a more uniform
/// histogram, effectively spreading out the most frequent intensity values. This improves
/// contrast, especially for images that are too dark or too bright.
/// </para>
/// <para><b>For Beginners:</b> If your image looks washed out or too dark, histogram
/// equalization stretches the range of colors to use the full spectrum, making details
/// more visible.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HistogramEqualization<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets whether to apply per-channel equalization.
    /// When false, converts to luminance, equalizes, then converts back.
    /// </summary>
    public bool PerChannel { get; }

    /// <summary>
    /// Creates a new histogram equalization augmentation.
    /// </summary>
    /// <param name="perChannel">Whether to equalize each channel independently. Default is false.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public HistogramEqualization(bool perChannel = false, double probability = 0.5) : base(probability)
    {
        PerChannel = perChannel;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int h = data.Height;
        int w = data.Width;
        int numBins = 256;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        if (PerChannel || data.Channels == 1)
        {
            for (int c = 0; c < data.Channels; c++)
            {
                EqualizeChannel(data, result, h, w, c, numBins, maxVal);
            }
        }
        else
        {
            // Equalize based on luminance: build histogram from average luminance,
            // then apply the same mapping to all channels proportionally
            EqualizeLuminance(data, result, h, w, numBins, maxVal);
        }

        return result;
    }

    private static void EqualizeLuminance(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int numBins, double maxVal)
    {
        // Build luminance histogram (average of all channels)
        var engine = AiDotNetEngine.Current;
        var planar = src.ToPlanarTensor();
        var lum = engine.TensorDivideScalar(engine.ReduceSum(planar, [0], false), NumOps.FromDouble(src.Channels));
        var bins = ImageTensor<T>.HistogramBins(lum, maxVal, numBins);
        var levels = ImageTensor<T>.EqualizationLevels(bins, numBins, maxVal);

        // Apply the same luminance-based mapping to all channels: scale = level / lum (1 for black pixels)
        var level = engine.TensorTake(levels, bins);
        var scale = engine.TensorWhere(
            engine.TensorGreaterThan(lum, NumOps.FromDouble(1e-10)),
            engine.TensorDivide(level, lum),
            ImageTensor<T>.Constant(lum, 1.0));
        var scaled = engine.TensorMultiply(planar, engine.Reshape(scale, [1, h, w]));
        dst.SetFromPlanarTensor(engine.TensorClamp(scaled, NumOps.Zero, NumOps.FromDouble(maxVal)));
    }

    private static void EqualizeChannel(ImageTensor<T> src, ImageTensor<T> dst,
        int h, int w, int channel, int numBins, double maxVal)
    {
        // Build histogram and equalized levels, then map every pixel through its bin
        var engine = AiDotNetEngine.Current;
        var bins = ImageTensor<T>.HistogramBins(src.GetChannelPlane(channel), maxVal, numBins);
        var levels = ImageTensor<T>.EqualizationLevels(bins, numBins, maxVal);
        dst.SetChannelPlane(channel, engine.TensorTake(levels, bins).Cast<T>());
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["per_channel"] = PerChannel;
        return p;
    }
}
