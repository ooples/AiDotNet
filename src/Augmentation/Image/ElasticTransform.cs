namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies elastic deformation to an image (Simard et al., 2003).
/// </summary>
/// <remarks>
/// <para>
/// Elastic deformation generates random displacement fields, smooths them with a Gaussian
/// filter, and applies them to warp the image. This creates realistic local distortions
/// similar to handwriting variations or biological tissue deformation.
/// </para>
/// <para><b>For Beginners:</b> Imagine placing your image on a rubber sheet and randomly
/// pushing and pulling different parts. The result looks naturally distorted, like how
/// handwriting varies slightly each time you write the same letter. This is one of the most
/// effective augmentations for digit/character recognition.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Handwritten digit/character recognition (MNIST, EMNIST)</item>
/// <item>Medical image segmentation (tissue deformation)</item>
/// <item>Any task where local shape variations are natural</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ElasticTransform<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the alpha parameter controlling displacement magnitude.
    /// </summary>
    public double Alpha { get; }

    /// <summary>
    /// Gets the sigma parameter controlling smoothness of the displacement field.
    /// </summary>
    public double Sigma { get; }

    /// <summary>
    /// Gets the fill value for out-of-bounds pixels.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new elastic transform.
    /// </summary>
    /// <param name="alpha">
    /// Displacement magnitude. Larger values create more distortion.
    /// Industry standard default is 50.0 for MNIST-like images.
    /// </param>
    /// <param name="sigma">
    /// Gaussian smoothing sigma for the displacement field. Larger values create
    /// smoother, more global distortions. Industry standard default is 5.0.
    /// </param>
    /// <param name="fillValue">Fill value for pixels outside bounds. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public ElasticTransform(
        double alpha = 50.0,
        double sigma = 5.0,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (alpha < 0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be non-negative.");
        if (sigma <= 0)
            throw new ArgumentOutOfRangeException(nameof(sigma), "Sigma must be positive.");

        Alpha = alpha;
        Sigma = sigma;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies the elastic deformation.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;

        // Generate random displacement fields
        var dx = new double[h, w];
        var dy = new double[h, w];

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                dx[y, x] = context.GetRandomDouble(-1, 1);
                dy[y, x] = context.GetRandomDouble(-1, 1);
            }
        }

        // Smooth with Gaussian filter
        dx = GaussianSmooth(dx, h, w, Sigma);
        dy = GaussianSmooth(dy, h, w, Sigma);

        // Scale by alpha
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                dx[y, x] *= Alpha;
                dy[y, x] *= Alpha;
            }
        }

        // Apply displacement
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
                double srcX = x + dx[y, x];
                double srcY = y + dy[y, x];

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

    private static double[,] GaussianSmooth(double[,] field, int h, int w, double sigma)
    {
        int kSize = (int)(sigma * 6) | 1;
        if (kSize < 3) kSize = 3;
        int halfK = kSize / 2;

        // Build 1D kernel
        var kernel = new double[kSize];
        double sum = 0;
        for (int i = 0; i < kSize; i++)
        {
            double d = i - halfK;
            kernel[i] = Math.Exp(-d * d / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < kSize; i++) kernel[i] /= sum;

        // Horizontal pass
        var temp = new double[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = 0;
                for (int k = -halfK; k <= halfK; k++)
                {
                    int sx = Math.Max(0, Math.Min(w - 1, x + k));
                    val += field[y, sx] * kernel[k + halfK];
                }
                temp[y, x] = val;
            }
        }

        // Vertical pass
        var result = new double[h, w];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double val = 0;
                for (int k = -halfK; k <= halfK; k++)
                {
                    int sy = Math.Max(0, Math.Min(h - 1, y + k));
                    val += temp[sy, x] * kernel[k + halfK];
                }
                result[y, x] = val;
            }
        }

        return result;
    }

    private static T BilinearSample(ImageTensor<T> image, double x, double y, int channel)
    {
        int x0 = (int)Math.Floor(x);
        int x1 = x0 + 1;
        int y0 = (int)Math.Floor(y);
        int y1 = y0 + 1;

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

        double top = v00 * (1 - fx) + v01 * fx;
        double bottom = v10 * (1 - fx) + v11 * fx;
        return NumOps.FromDouble(top * (1 - fy) + bottom * fy);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["alpha"] = Alpha;
        parameters["sigma"] = Sigma;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
