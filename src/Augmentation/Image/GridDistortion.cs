namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies grid-based distortion to an image.
/// </summary>
/// <remarks>
/// <para>
/// Grid distortion divides the image into a grid of cells and randomly displaces the grid
/// intersection points, then interpolates the displacement across each cell. This creates
/// smooth, locally varying distortions.
/// </para>
/// <para><b>For Beginners:</b> Imagine overlaying a flexible grid on your image, then randomly
/// nudging each grid intersection point. The image warps smoothly between the displaced points,
/// creating natural-looking distortions.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>OCR and document recognition</item>
/// <item>Medical image augmentation</item>
/// <item>Any task requiring smooth local distortions</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GridDistortion<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of grid steps in each dimension.
    /// </summary>
    public int NumSteps { get; }

    /// <summary>
    /// Gets the maximum distortion magnitude.
    /// </summary>
    public double DistortLimit { get; }

    /// <summary>
    /// Gets the fill value for out-of-bounds pixels.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new grid distortion augmentation.
    /// </summary>
    /// <param name="numSteps">Number of grid divisions per side. Default is 5.</param>
    /// <param name="distortLimit">Maximum displacement as fraction of step size. Default is 0.3.</param>
    /// <param name="fillValue">Fill value for out-of-bounds pixels. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public GridDistortion(
        int numSteps = 5,
        double distortLimit = 0.3,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (numSteps < 2)
            throw new ArgumentOutOfRangeException(nameof(numSteps), "Must have at least 2 grid steps.");
        if (distortLimit < 0 || distortLimit > 1)
            throw new ArgumentOutOfRangeException(nameof(distortLimit), "Must be between 0 and 1.");

        NumSteps = numSteps;
        DistortLimit = distortLimit;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies the grid distortion.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;

        double stepX = (double)w / NumSteps;
        double stepY = (double)h / NumSteps;

        // Generate random displacements at grid points
        var gridDx = new double[NumSteps + 1, NumSteps + 1];
        var gridDy = new double[NumSteps + 1, NumSteps + 1];

        for (int gy = 0; gy <= NumSteps; gy++)
        {
            for (int gx = 0; gx <= NumSteps; gx++)
            {
                // Don't displace border points
                if (gy == 0 || gy == NumSteps || gx == 0 || gx == NumSteps)
                {
                    gridDx[gy, gx] = 0;
                    gridDy[gy, gx] = 0;
                }
                else
                {
                    gridDx[gy, gx] = context.GetRandomDouble(-DistortLimit, DistortLimit) * stepX;
                    gridDy[gy, gx] = context.GetRandomDouble(-DistortLimit, DistortLimit) * stepY;
                }
            }
        }

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
                // Find which grid cell we're in
                double gxf = x / stepX;
                double gyf = y / stepY;
                int gx0 = Math.Min((int)gxf, NumSteps - 1);
                int gy0 = Math.Min((int)gyf, NumSteps - 1);
                int gx1 = gx0 + 1;
                int gy1 = gy0 + 1;

                double fx = gxf - gx0;
                double fy = gyf - gy0;

                // Bilinear interpolation of displacement
                double dx = gridDx[gy0, gx0] * (1 - fx) * (1 - fy) +
                            gridDx[gy0, gx1] * fx * (1 - fy) +
                            gridDx[gy1, gx0] * (1 - fx) * fy +
                            gridDx[gy1, gx1] * fx * fy;

                double dy = gridDy[gy0, gx0] * (1 - fx) * (1 - fy) +
                            gridDy[gy0, gx1] * fx * (1 - fy) +
                            gridDy[gy1, gx0] * (1 - fx) * fy +
                            gridDy[gy1, gx1] * fx * fy;

                double srcX = x + dx;
                double srcY = y + dy;

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
        parameters["num_steps"] = NumSteps;
        parameters["distort_limit"] = DistortLimit;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
