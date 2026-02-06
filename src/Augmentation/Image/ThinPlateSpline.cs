namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies thin plate spline (TPS) transformation to an image.
/// </summary>
/// <remarks>
/// <para>
/// Thin plate spline is a smooth interpolation method that warps an image by specifying
/// control point displacements. TPS minimizes the bending energy of the deformation,
/// producing the smoothest possible warp that passes through all control points.
/// Named after the physical analogy of bending a thin metal plate.
/// </para>
/// <para><b>For Beginners:</b> Imagine pinning a flexible sheet at several points and then
/// moving some pins. The sheet bends smoothly between the pins. TPS creates the smoothest
/// possible distortion that matches all the specified control point movements.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Shape deformation augmentation</item>
/// <item>Medical image registration</item>
/// <item>Face warping and morphing</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ThinPlateSpline<T> : ImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the number of control points per dimension.
    /// </summary>
    public int NumControlPoints { get; }

    /// <summary>
    /// Gets the maximum displacement scale.
    /// </summary>
    public double Scale { get; }

    /// <summary>
    /// Gets the fill value for out-of-bounds pixels.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new thin plate spline augmentation.
    /// </summary>
    /// <param name="numControlPoints">Control points per dimension. Default is 3 (gives 3x3=9 points).</param>
    /// <param name="scale">Maximum displacement as fraction of image size. Default is 0.05.</param>
    /// <param name="fillValue">Fill value for out-of-bounds. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public ThinPlateSpline(
        int numControlPoints = 3,
        double scale = 0.05,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (numControlPoints < 2)
            throw new ArgumentOutOfRangeException(nameof(numControlPoints), "Must be at least 2.");
        if (scale < 0)
            throw new ArgumentOutOfRangeException(nameof(scale), "Must be non-negative.");

        NumControlPoints = numControlPoints;
        Scale = scale;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies TPS transformation using grid-based approximation for efficiency.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;
        int nPts = NumControlPoints * NumControlPoints;

        // Generate control points on regular grid with random displacements
        var srcX = new double[nPts];
        var srcY = new double[nPts];
        var dstX = new double[nPts];
        var dstY = new double[nPts];

        int idx = 0;
        for (int gy = 0; gy < NumControlPoints; gy++)
        {
            for (int gx = 0; gx < NumControlPoints; gx++)
            {
                double px = (gx + 0.5) / NumControlPoints;
                double py = (gy + 0.5) / NumControlPoints;
                srcX[idx] = px;
                srcY[idx] = py;
                dstX[idx] = px + context.GetRandomDouble(-Scale, Scale);
                dstY[idx] = py + context.GetRandomDouble(-Scale, Scale);
                idx++;
            }
        }

        // Compute TPS weights for X and Y displacements
        var weightsX = ComputeTpsWeights(srcX, srcY, dstX, nPts);
        var weightsY = ComputeTpsWeights(srcX, srcY, dstY, nPts);

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
                double nx = (double)x / w;
                double ny = (double)y / h;

                // Evaluate TPS at this point
                double mappedX = EvaluateTps(nx, ny, srcX, srcY, weightsX, nPts) * w;
                double mappedY = EvaluateTps(nx, ny, srcX, srcY, weightsY, nPts) * h;

                for (int c = 0; c < data.Channels; c++)
                {
                    if (mappedX >= 0 && mappedX < w - 1 && mappedY >= 0 && mappedY < h - 1)
                    {
                        result.SetPixel(y, x, c, BilinearSample(data, mappedX, mappedY, c));
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

    private static double TpsKernel(double r)
    {
        if (r < 1e-10) return 0;
        return r * r * Math.Log(r);
    }

    private static double[] ComputeTpsWeights(double[] srcX, double[] srcY, double[] dst, int n)
    {
        // Build the TPS system: [K P; P' 0] * [w; a] = [dst; 0]
        int size = n + 3;
        var mat = new double[size, size];
        var rhs = new double[size];

        // K matrix (kernel evaluations)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double dx = srcX[i] - srcX[j];
                double dy = srcY[i] - srcY[j];
                double r = Math.Sqrt(dx * dx + dy * dy);
                mat[i, j] = TpsKernel(r);
            }
            // Regularization for numerical stability
            mat[i, i] += 1e-6;
        }

        // P matrix (affine part)
        for (int i = 0; i < n; i++)
        {
            mat[i, n] = 1;
            mat[i, n + 1] = srcX[i];
            mat[i, n + 2] = srcY[i];
            mat[n, i] = 1;
            mat[n + 1, i] = srcX[i];
            mat[n + 2, i] = srcY[i];
        }

        // RHS
        for (int i = 0; i < n; i++)
            rhs[i] = dst[i];

        return SolveSystem(mat, rhs, size);
    }

    private static double EvaluateTps(double x, double y, double[] srcX, double[] srcY, double[] weights, int n)
    {
        double result = weights[n] + weights[n + 1] * x + weights[n + 2] * y;

        for (int i = 0; i < n; i++)
        {
            double dx = x - srcX[i];
            double dy = y - srcY[i];
            double r = Math.Sqrt(dx * dx + dy * dy);
            result += weights[i] * TpsKernel(r);
        }

        return result;
    }

    private static double[] SolveSystem(double[,] a, double[] b, int n)
    {
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) augmented[i, j] = a[i, j];
            augmented[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            double maxVal = Math.Abs(augmented[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > maxVal)
                {
                    maxVal = Math.Abs(augmented[row, col]);
                    maxRow = row;
                }
            }

            for (int j = 0; j <= n; j++)
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);

            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-12) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = augmented[row, col] / pivot;
                for (int j = col; j <= n; j++)
                    augmented[row, j] -= factor * augmented[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= augmented[i, j] * x[j];
            if (Math.Abs(augmented[i, i]) > 1e-12)
                x[i] /= augmented[i, i];
        }

        return x;
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
        parameters["num_control_points"] = NumControlPoints;
        parameters["scale"] = Scale;
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
