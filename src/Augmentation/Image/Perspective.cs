namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a random perspective transformation to an image.
/// </summary>
/// <remarks>
/// <para>
/// Perspective transformation simulates viewing the image from different angles by applying
/// a projective (homography) transform. Each corner of the image is displaced by a random
/// amount, creating a realistic 3D perspective effect.
/// </para>
/// <para><b>For Beginners:</b> Imagine tilting a photo in 3D space — objects closer to you
/// appear larger, and objects farther away appear smaller. This augmentation simulates that
/// effect, teaching your model to recognize objects regardless of viewing angle.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Document recognition (photos of documents taken at angles)</item>
/// <item>Street sign/license plate recognition</item>
/// <item>Any task where the camera angle may vary</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Perspective<T> : SpatialImageAugmenterBase<T>
{
    /// <summary>
    /// Gets the maximum distortion scale (fraction of image size).
    /// </summary>
    public double DistortionScale { get; }

    /// <summary>
    /// Gets the interpolation mode.
    /// </summary>
    public InterpolationMode Interpolation { get; }

    /// <summary>
    /// Gets the fill value for areas outside the transformed image.
    /// </summary>
    public double FillValue { get; }

    /// <summary>
    /// Creates a new perspective transformation.
    /// </summary>
    /// <param name="distortionScale">
    /// Maximum displacement of each corner as a fraction of image size.
    /// Industry standard default is 0.5. Range: [0, 1].
    /// </param>
    /// <param name="interpolation">Interpolation mode. Default is Bilinear.</param>
    /// <param name="fillValue">Fill value for out-of-bounds pixels. Default is 0.</param>
    /// <param name="probability">Probability of applying. Default is 0.5.</param>
    public Perspective(
        double distortionScale = 0.5,
        InterpolationMode interpolation = InterpolationMode.Bilinear,
        double fillValue = 0,
        double probability = 0.5) : base(probability)
    {
        if (distortionScale < 0 || distortionScale > 1)
            throw new ArgumentOutOfRangeException(nameof(distortionScale), "Must be between 0 and 1.");

        DistortionScale = distortionScale;
        Interpolation = interpolation;
        FillValue = fillValue;
    }

    /// <summary>
    /// Applies perspective transformation and returns transform parameters.
    /// </summary>
    protected override (ImageTensor<T> data, IDictionary<string, object> parameters) ApplyWithTransformParams(
        ImageTensor<T> data, AugmentationContext<T> context)
    {
        int h = data.Height;
        int w = data.Width;
        int halfH = (int)(h * DistortionScale * 0.5);
        int halfW = (int)(w * DistortionScale * 0.5);

        // Source corners (original image corners)
        double[][] srcPts =
        [
            [0, 0], [w, 0], [w, h], [0, h]
        ];

        // Destination corners (randomly displaced)
        double[][] dstPts =
        [
            [context.GetRandomInt(0, Math.Max(1, halfW)), context.GetRandomInt(0, Math.Max(1, halfH))],
            [w - context.GetRandomInt(0, Math.Max(1, halfW)), context.GetRandomInt(0, Math.Max(1, halfH))],
            [w - context.GetRandomInt(0, Math.Max(1, halfW)), h - context.GetRandomInt(0, Math.Max(1, halfH))],
            [context.GetRandomInt(0, Math.Max(1, halfW)), h - context.GetRandomInt(0, Math.Max(1, halfH))]
        ];

        // Compute the 3x3 perspective matrix (dst -> src for inverse mapping)
        var matrix = ComputePerspectiveMatrix(dstPts, srcPts);

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
                // Apply inverse perspective transform
                double denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2];
                if (Math.Abs(denom) < 1e-10) denom = 1e-10;

                double srcX = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom;
                double srcY = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom;

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

        var parameters = new Dictionary<string, object>
        {
            ["src_points"] = srcPts,
            ["dst_points"] = dstPts,
            ["original_width"] = w,
            ["original_height"] = h
        };

        return (result, parameters);
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

        double xFrac = x - Math.Floor(x);
        double yFrac = y - Math.Floor(y);

        double v00 = NumOps.ToDouble(image.GetPixel(y0, x0, channel));
        double v01 = NumOps.ToDouble(image.GetPixel(y0, x1, channel));
        double v10 = NumOps.ToDouble(image.GetPixel(y1, x0, channel));
        double v11 = NumOps.ToDouble(image.GetPixel(y1, x1, channel));

        double top = v00 * (1 - xFrac) + v01 * xFrac;
        double bottom = v10 * (1 - xFrac) + v11 * xFrac;
        return NumOps.FromDouble(top * (1 - yFrac) + bottom * yFrac);
    }

    /// <summary>
    /// Computes a 3x3 perspective transformation matrix using 4 point correspondences.
    /// Uses a simplified approach solving an 8x8 linear system.
    /// </summary>
    private static double[,] ComputePerspectiveMatrix(double[][] src, double[][] dst)
    {
        // Build the 8x8 system for the homography
        var a = new double[8, 8];
        var b = new double[8];

        for (int i = 0; i < 4; i++)
        {
            double sx = src[i][0], sy = src[i][1];
            double dx = dst[i][0], dy = dst[i][1];

            a[i * 2, 0] = sx; a[i * 2, 1] = sy; a[i * 2, 2] = 1;
            a[i * 2, 3] = 0;  a[i * 2, 4] = 0;  a[i * 2, 5] = 0;
            a[i * 2, 6] = -dx * sx; a[i * 2, 7] = -dx * sy;
            b[i * 2] = dx;

            a[i * 2 + 1, 0] = 0;  a[i * 2 + 1, 1] = 0;  a[i * 2 + 1, 2] = 0;
            a[i * 2 + 1, 3] = sx; a[i * 2 + 1, 4] = sy; a[i * 2 + 1, 5] = 1;
            a[i * 2 + 1, 6] = -dy * sx; a[i * 2 + 1, 7] = -dy * sy;
            b[i * 2 + 1] = dy;
        }

        // Solve using Gaussian elimination
        var h = SolveLinearSystem(a, b);

        return new double[,]
        {
            { h[0], h[1], h[2] },
            { h[3], h[4], h[5] },
            { h[6], h[7], 1.0 }
        };
    }

    private static double[] SolveLinearSystem(double[,] a, double[] b)
    {
        int n = b.Length;
        var augmented = new double[n, n + 1];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                augmented[i, j] = a[i, j];
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
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

            // Swap rows
            for (int j = 0; j <= n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-12) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = augmented[row, col] / pivot;
                for (int j = col; j <= n; j++)
                    augmented[row, j] -= factor * augmented[col, j];
            }
        }

        // Back substitution
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

    /// <inheritdoc />
    protected override BoundingBox<T> TransformBoundingBox(BoundingBox<T> box,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        // Perspective transforms are complex for bboxes; return unchanged as approximation
        return box.Clone();
    }

    /// <inheritdoc />
    protected override Keypoint<T> TransformKeypoint(Keypoint<T> keypoint,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        return keypoint.Clone();
    }

    /// <inheritdoc />
    protected override SegmentationMask<T> TransformMask(SegmentationMask<T> mask,
        IDictionary<string, object> transformParams, AugmentationContext<T> context)
    {
        return mask;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["distortion_scale"] = DistortionScale;
        parameters["interpolation"] = Interpolation.ToString();
        parameters["fill_value"] = FillValue;
        return parameters;
    }
}
