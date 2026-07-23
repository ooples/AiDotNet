namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Resizes an image to a target size using configurable interpolation.
/// </summary>
/// <remarks>
/// <para>
/// Resize changes the spatial dimensions of an image using various interpolation methods.
/// Different interpolation modes trade off between speed and quality:
/// <list type="bullet">
/// <item><b>Nearest</b>: Fastest, produces blocky results. Best for masks/labels.</item>
/// <item><b>Bilinear</b>: Good balance of speed and quality. Default for most tasks.</item>
/// <item><b>Bicubic</b>: Smoother results, slower. Good for high-quality resizing.</item>
/// <item><b>Lanczos</b>: Highest quality, slowest. Best for final output.</item>
/// <item><b>Area</b>: Best for downscaling, averages pixel areas.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Resizing changes the pixel dimensions of an image. When making
/// an image smaller, pixels must be combined. When making it larger, new pixels must be
/// created by interpolating between existing ones. The interpolation mode controls how
/// this is done.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Preparing images for model input (most models require fixed input size)</item>
/// <item>Downscaling large images to reduce memory and computation</item>
/// <item>Upscaling small images when needed</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Resize<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the target height.
    /// </summary>
    public int TargetHeight { get; }

    /// <summary>
    /// Gets the target width.
    /// </summary>
    public int TargetWidth { get; }

    /// <summary>
    /// Gets the interpolation mode used for resizing.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Creates a new resize augmentation.
    /// </summary>
    /// <param name="targetHeight">The target height in pixels. Must be positive.</param>
    /// <param name="targetWidth">The target width in pixels. Must be positive.</param>
    /// <param name="interpolation">
    /// The interpolation mode. Industry standard default is Bilinear.
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Default is 1.0 (always apply) since resize is typically deterministic preprocessing.
    /// </param>
    public Resize(
        int targetHeight,
        int targetWidth,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        double probability = 1.0) : base(probability)
    {
        if (targetHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), "Target height must be positive.");
        if (targetWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetWidth), "Target width must be positive.");

        TargetHeight = targetHeight;
        TargetWidth = targetWidth;
        Interpolation = interpolation;
    }

    /// <summary>
    /// Applies the resize operation to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Height == TargetHeight && data.Width == TargetWidth)
            return data.Clone();

        var result = new ImageTensor<T>(TargetHeight, TargetWidth, data.Channels, data.ChannelOrder, data.ColorSpace)
        {
            IsNormalized = data.IsNormalized,
            NormalizationMean = data.NormalizationMean,
            NormalizationStd = data.NormalizationStd,
            OriginalRange = data.OriginalRange
        };

        double scaleY = (double)data.Height / TargetHeight;
        double scaleX = (double)data.Width / TargetWidth;

        for (int y = 0; y < TargetHeight; y++)
        {
            for (int x = 0; x < TargetWidth; x++)
            {
                // Map target pixel to source pixel centers for accurate interpolation
                double srcY = (y + 0.5) * scaleY - 0.5;
                double srcX = (x + 0.5) * scaleX - 0.5;

                double maxVal = data.IsNormalized ? 1.0 : 255.0;
                for (int c = 0; c < data.Channels; c++)
                {
                    double value = Interpolation switch
                    {
                        InterpolationMode.Nearest => SampleNearest(data, srcX, srcY, c),
                        InterpolationMode.Bilinear => SampleBilinear(data, srcX, srcY, c),
                        InterpolationMode.Bicubic => SampleBicubic(data, srcX, srcY, c),
                        InterpolationMode.Area => SampleArea(data, x, y, scaleX, scaleY, c),
                        InterpolationMode.Lanczos => SampleLanczos(data, srcX, srcY, c),
                        _ => SampleBilinear(data, srcX, srcY, c)
                    };

                    // Clamp to valid range (bicubic and Lanczos can overshoot)
                    value = Math.Max(0, Math.Min(maxVal, value));
                    result.SetPixel(y, x, c, NumOps.FromDouble(value));
                }
            }
        }

        return result;
    }

    private static double SampleNearest(ImageTensor<T> image, double x, double y, int channel)
    {
        int ix = Math.Max(0, Math.Min((int)Math.Round(x), image.Width - 1));
        int iy = Math.Max(0, Math.Min((int)Math.Round(y), image.Height - 1));
        return NumOps.ToDouble(image.GetPixel(iy, ix, channel));
    }

    private static double SampleBilinear(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = Math.Max(0, Math.Min((int)Math.Floor(x), image.Width - 1));
        int x1 = Math.Max(0, Math.Min(x0 + 1, image.Width - 1));
        int y0 = Math.Max(0, Math.Min((int)Math.Floor(y), image.Height - 1));
        int y1 = Math.Max(0, Math.Min(y0 + 1, image.Height - 1));

        double xFrac = x - Math.Floor(x);
        double yFrac = y - Math.Floor(y);

        double v00 = NumOps.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = NumOps.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = NumOps.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = NumOps.ToDouble(image.GetPixel(y1, x1, channel));

        double top = v00 * (1 - xFrac) + v01 * xFrac;
        double bottom = v10 * (1 - xFrac) + v11 * xFrac;
        return top * (1 - yFrac) + bottom * yFrac;
    }

    private static double SampleBicubic(ImageTensor<T> image, double x, double y, int channel)
    {
        int ix = (int)Math.Floor(x);
        int iy = (int)Math.Floor(y);
        double fx = x - ix;
        double fy = y - iy;

        double result = 0;
        for (int dy = -1; dy <= 2; dy++)
        {
            double rowVal = 0;
            for (int dx = -1; dx <= 2; dx++)
            {
                int px = Math.Max(0, Math.Min(ix + dx, image.Width - 1));
                int py = Math.Max(0, Math.Min(iy + dy, image.Height - 1));
                double val = NumOps.ToDouble(image.GetPixel(py, px, channel));
                rowVal += val * CubicWeight(fx - dx);
            }
            result += rowVal * CubicWeight(fy - dy);
        }

        return result;
    }

    private static double CubicWeight(double t)
    {
        t = Math.Abs(t);
        if (t <= 1.0)
            return (1.5 * t * t * t) - (2.5 * t * t) + 1.0;
        if (t <= 2.0)
            return (-0.5 * t * t * t) + (2.5 * t * t) - (4.0 * t) + 2.0;
        return 0.0;
    }

    private static double SampleArea(ImageTensor<T> image, int destX, int destY,
        double scaleX, double scaleY, int channel)
    {
        double srcX0 = destX * scaleX;
        double srcY0 = destY * scaleY;
        double srcX1 = (destX + 1) * scaleX;
        double srcY1 = (destY + 1) * scaleY;

        int x0 = Math.Max(0, Math.Min((int)Math.Floor(srcX0), image.Width - 1));
        int x1 = Math.Max(0, Math.Min((int)Math.Ceiling(srcX1) - 1, image.Width - 1));
        int y0 = Math.Max(0, Math.Min((int)Math.Floor(srcY0), image.Height - 1));
        int y1 = Math.Max(0, Math.Min((int)Math.Ceiling(srcY1) - 1, image.Height - 1));

        double sum = 0;
        int count = 0;
        for (int sy = y0; sy <= y1; sy++)
        {
            for (int sx = x0; sx <= x1; sx++)
            {
                sum += NumOps.ToDouble(image.GetPixel(sy, sx, channel));
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    private static double SampleLanczos(ImageTensor<T> image, double x, double y, int channel)
    {
        const int a = 3; // Lanczos-3
        int ix = (int)Math.Floor(x);
        int iy = (int)Math.Floor(y);

        double result = 0;
        double weightSum = 0;

        for (int dy = -(a - 1); dy <= a; dy++)
        {
            for (int dx = -(a - 1); dx <= a; dx++)
            {
                int px = Math.Max(0, Math.Min(ix + dx, image.Width - 1));
                int py = Math.Max(0, Math.Min(iy + dy, image.Height - 1));

                double wx = LanczosKernel(x - (ix + dx), a);
                double wy = LanczosKernel(y - (iy + dy), a);
                double weight = wx * wy;

                result += NumOps.ToDouble(image.GetPixel(py, px, channel)) * weight;
                weightSum += weight;
            }
        }

        return weightSum > 0 ? result / weightSum : 0;
    }

    private static double LanczosKernel(double x, int a)
    {
        if (Math.Abs(x) < 1e-8) return 1.0;
        if (Math.Abs(x) >= a) return 0.0;
        double pix = Math.PI * x;
        return a * Math.Sin(pix) * Math.Sin(pix / a) / (pix * pix);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["target_height"] = TargetHeight;
        parameters["target_width"] = TargetWidth;
        parameters["interpolation"] = Interpolation.ToString();
        return parameters;
    }
}
