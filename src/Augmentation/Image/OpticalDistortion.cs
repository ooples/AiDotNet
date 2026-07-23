namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates barrel and pincushion lens distortion.
/// </summary>
/// <remarks>
/// <para>
/// Optical distortion simulates the radial distortion produced by camera lenses. Barrel
/// distortion (positive k) makes straight lines bow outward, while pincushion distortion
/// (negative k) makes them bow inward. This is a common effect in wide-angle and telephoto lenses.
/// </para>
/// <para><b>For Beginners:</b> Camera lenses aren't perfect — they bend straight lines
/// slightly. Wide-angle lenses make edges bulge outward (barrel distortion), while telephoto
/// lenses make edges pinch inward (pincushion). This augmentation simulates these effects.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Training models robust to different camera lenses</item>
/// <item>Autonomous driving (wide-angle dash cameras)</item>
/// <item>Surveillance camera footage processing</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OpticalDistortion<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the minimum distortion coefficient.
    /// </summary>
    public double MinDistortionK { get; }

    /// <summary>
    /// Gets the maximum distortion coefficient.
    /// </summary>
    public double MaxDistortionK { get; }

    /// <summary>
    /// Gets the fill value for out-of-bounds pixels.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new optical distortion augmentation.
    /// </summary>
    /// <param name="minDistortionK">Minimum distortion coefficient. Default is -0.3 (pincushion).</param>
    /// <param name="maxDistortionK">Maximum distortion coefficient. Default is 0.3 (barrel).</param>
    /// <param name="fillValue">Fill value for out-of-bounds pixels. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public OpticalDistortion(
        double minDistortionK = -0.3,
        double maxDistortionK = 0.3,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (minDistortionK > maxDistortionK)
            throw new ArgumentException("minDistortionK must be <= maxDistortionK.");

        MinDistortionK = minDistortionK;
        MaxDistortionK = maxDistortionK;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies the optical distortion.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;
        double k = context.GetRandomDouble(MinDistortionK, MaxDistortionK);

        double cx = w / 2.0;
        double cy = h / 2.0;
        double maxRadius = Math.Sqrt(cx * cx + cy * cy);

        var result = new ImageTensor<T>(h, w, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        T fill = NumOps.FromDouble(FillValue);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                // Normalize coordinates to [-1, 1]
                double nx = (x - cx) / maxRadius;
                double ny = (y - cy) / maxRadius;
                double r2 = nx * nx + ny * ny;

                // Apply radial distortion
                double distortion = 1 + k * r2;
                double srcX = cx + nx * distortion * maxRadius;
                double srcY = cy + ny * distortion * maxRadius;

                for (int c = 0; c < data.Channels; c++)
                {
                    if (srcX >= 0 && srcX < w - 1 && srcY >= 0 && srcY < h - 1)
                    {
                        result.SetPixel(y, x, c, BilinearSample(data, srcX, srcY, c));
                    }
                    else
                    {
                        result.SetPixel(y, x, c, fill);
                    }
                }
            }
        }

        return result;
    }

    private static T BilinearSample(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = (int)Math.Floor(x), x1 = x0 + 1;
        int y0 = (int)Math.Floor(y), y1 = y0 + 1;
        x0 = Math.Max(0, Math.Min(image.Width - 1, x0));
        x1 = Math.Max(0, Math.Min(image.Width - 1, x1));
        y0 = Math.Max(0, Math.Min(image.Height - 1, y0));
        y1 = Math.Max(0, Math.Min(image.Height - 1, y1));

        double fx = x - Math.Floor(x);
        double fy = y - Math.Floor(y);

        double v00 = NumOps.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = NumOps.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = NumOps.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = NumOps.ToDouble(image.GetPixel(y1, x1, channel));

        return NumOps.FromDouble(v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                                  v10 * (1 - fx) * fy + v11 * fx * fy);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["min_distortion_k"] = MinDistortionK;
        parameters["max_distortion_k"] = MaxDistortionK;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
