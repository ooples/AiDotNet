namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies piecewise affine transformation by dividing the image into triangular regions.
/// </summary>
/// <remarks>
/// <para>
/// Piecewise affine transformation places a regular grid over the image, randomly displaces
/// the grid points, triangulates the grid, then applies a separate affine transform within
/// each triangle. This creates smooth, locally-varying warps that are more controlled than
/// elastic deformation.
/// </para>
/// <para><b>For Beginners:</b> This divides your image into a grid of triangles, then slightly
/// shifts each grid point. Each triangle is then stretched/warped independently but connects
/// smoothly to its neighbors, creating a natural-looking distortion.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Character/handwriting recognition augmentation</item>
/// <item>Face augmentation with controlled deformation</item>
/// <item>When you need smoother distortion than elastic transform</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PiecewiseAffine<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of grid rows.
    /// </summary>
    public int GridRows { get; }

    /// <summary>
    /// Gets the number of grid columns.
    /// </summary>
    public int GridCols { get; }

    /// <summary>
    /// Gets the maximum displacement scale as fraction of grid cell size.
    /// </summary>
    public double Scale { get; }

    /// <summary>
    /// Gets the fill value for out-of-bounds pixels.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new piecewise affine augmentation.
    /// </summary>
    /// <param name="gridRows">Number of grid rows. Default is 4.</param>
    /// <param name="gridCols">Number of grid columns. Default is 4.</param>
    /// <param name="scale">Maximum displacement as fraction of cell size. Default is 0.05.</param>
    /// <param name="fillValue">Fill value for out-of-bounds pixels. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public PiecewiseAffine(
        int gridRows = 4,
        int gridCols = 4,
        double scale = 0.05,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (gridRows < 2) throw new ArgumentOutOfRangeException(nameof(gridRows), "Must be at least 2.");
        if (gridCols < 2) throw new ArgumentOutOfRangeException(nameof(gridCols), "Must be at least 2.");
        if (scale < 0) throw new ArgumentOutOfRangeException(nameof(scale), "Must be non-negative.");

        GridRows = gridRows;
        GridCols = gridCols;
        Scale = scale;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies the piecewise affine transformation using grid-based interpolation.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;

        double cellW = (double)w / GridCols;
        double cellH = (double)h / GridRows;

        // Generate displaced grid points
        var srcPts = new double[GridRows + 1, GridCols + 1, 2];
        var dstPts = new double[GridRows + 1, GridCols + 1, 2];

        for (int gy = 0; gy <= GridRows; gy++)
        {
            for (int gx = 0; gx <= GridCols; gx++)
            {
                double px = gx * cellW;
                double py = gy * cellH;
                srcPts[gy, gx, 0] = px;
                srcPts[gy, gx, 1] = py;

                // Don't displace border points
                if (gy == 0 || gy == GridRows || gx == 0 || gx == GridCols)
                {
                    dstPts[gy, gx, 0] = px;
                    dstPts[gy, gx, 1] = py;
                }
                else
                {
                    dstPts[gy, gx, 0] = px + context.GetRandomDouble(-Scale, Scale) * cellW;
                    dstPts[gy, gx, 1] = py + context.GetRandomDouble(-Scale, Scale) * cellH;
                }
            }
        }

        // Use bilinear grid interpolation for the mapping
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
                // Find grid cell
                double gxf = x / cellW;
                double gyf = y / cellH;
                int gx0 = Math.Min((int)gxf, GridCols - 1);
                int gy0 = Math.Min((int)gyf, GridRows - 1);
                int gx1 = gx0 + 1;
                int gy1 = gy0 + 1;

                double fx = gxf - gx0;
                double fy = gyf - gy0;

                // Bilinear interpolation of source points from destination grid
                double srcX = dstPts[gy0, gx0, 0] * (1 - fx) * (1 - fy) +
                              dstPts[gy0, gx1, 0] * fx * (1 - fy) +
                              dstPts[gy1, gx0, 0] * (1 - fx) * fy +
                              dstPts[gy1, gx1, 0] * fx * fy;

                // Map back: find where this destination pixel came from in source
                // Use inverse mapping: source = original grid position + (dst displacement inverted)
                double origX = srcPts[gy0, gx0, 0] * (1 - fx) * (1 - fy) +
                               srcPts[gy0, gx1, 0] * fx * (1 - fy) +
                               srcPts[gy1, gx0, 0] * (1 - fx) * fy +
                               srcPts[gy1, gx1, 0] * fx * fy;

                double origY = srcPts[gy0, gx0, 1] * (1 - fx) * (1 - fy) +
                               srcPts[gy0, gx1, 1] * fx * (1 - fy) +
                               srcPts[gy1, gx0, 1] * (1 - fx) * fy +
                               srcPts[gy1, gx1, 1] * fx * fy;

                double dstY = dstPts[gy0, gx0, 1] * (1 - fx) * (1 - fy) +
                              dstPts[gy0, gx1, 1] * fx * (1 - fy) +
                              dstPts[gy1, gx0, 1] * (1 - fx) * fy +
                              dstPts[gy1, gx1, 1] * fx * fy;

                // Forward warp: map source to destination displacement
                double mapX = x + (origX - srcX);
                double mapY = y + (origY - dstY);

                for (int c = 0; c < data.Channels; c++)
                {
                    if (mapX >= 0 && mapX < w - 1 && mapY >= 0 && mapY < h - 1)
                    {
                        result.SetPixel(y, x, c, BilinearSample(data, mapX, mapY, c));
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
        double fx = x - Math.Floor(x), fy = y - Math.Floor(y);
        double v = NumOps.ToDouble(image.GetPixel(y0, x0, channel)) * (1 - fx) * (1 - fy) +
                   NumOps.ToDouble(image.GetPixel(y0, x1, channel)) * fx * (1 - fy) +
                   NumOps.ToDouble(image.GetPixel(y1, x0, channel)) * (1 - fx) * fy +
                   NumOps.ToDouble(image.GetPixel(y1, x1, channel)) * fx * fy;
        return NumOps.FromDouble(v);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["grid_rows"] = GridRows;
        parameters["grid_cols"] = GridCols;
        parameters["scale"] = Scale;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
