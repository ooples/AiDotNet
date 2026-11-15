namespace AiDotNet.Diffusion.ThreeD;

/// <summary>The interpolation kernel of <see cref="AntialiasedResize{T}"/>.</summary>
internal enum ResizeFilter
{
    /// <summary>
    /// The triangle (bilinear) kernel, support 1: torch <c>F.interpolate(mode="bilinear", antialias=True)</c> (ATen's
    /// <c>upsample_bilinear2d_aa</c>) and PIL's <c>BILINEAR</c>.
    /// </summary>
    Triangle,

    /// <summary>
    /// The cubic convolution kernel with a = -0.5, support 2: PIL's <c>BICUBIC</c> (<c>bicubic_filter</c> in
    /// <c>libImaging/Resample.c</c>). torch's antialiased bicubic uses the same a = -0.5.
    /// </summary>
    Bicubic,
}

/// <summary>
/// A separable antialiased image resize, as PIL's <c>Image.resize</c> and torch's
/// <c>F.interpolate(..., align_corners=False, antialias=True)</c> compute it (PIL <c>ImagingResample</c>, ATen
/// <c>_upsample_*_aa</c>). Along each axis, output i takes the kernel centred at <c>scale (i + 0.5)</c> in input
/// pixels, <c>scale = in / out</c>; when shrinking the kernel is stretched by <c>scale</c> (so it covers every input
/// pixel it averages: the antialiasing), when enlarging it is the plain kernel. Each output reads the inputs within
/// the stretched support, truncated as both libraries truncate and clamped to the image, with weights normalized to sum
/// to one. Each axis is one matrix product, width first then height, as PIL resizes horizontally first.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal static class AntialiasedResize<T>
{
    private const double CubicA = -0.5;   // PIL bicubic_filter: #define a -0.5

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>The resampling matrix [outSize, inSize], row-major: output i is the row-i weighted sum of the inputs.</summary>
    public static double[] Weights(int inSize, int outSize, ResizeFilter filter)
    {
        if (inSize <= 0) throw new ArgumentOutOfRangeException(nameof(inSize), "The input size must be positive.");
        if (outSize <= 0) throw new ArgumentOutOfRangeException(nameof(outSize), "The output size must be positive.");
        double kernelSupport = filter switch
        {
            ResizeFilter.Triangle => 1.0,
            ResizeFilter.Bicubic => 2.0,
            _ => throw new ArgumentOutOfRangeException(nameof(filter), filter, "Unknown resize filter."),
        };
        double scale = (double)inSize / outSize;
        double stretch = scale >= 1.0 ? scale : 1.0;
        double support = kernelSupport * stretch;
        double invStretch = 1.0 / stretch;
        var weights = new double[outSize * inSize];
        for (int i = 0; i < outSize; i++)
        {
            double center = scale * (i + 0.5);
            // Both libraries truncate toward zero, then clamp to the input.
            int start = Math.Max((int)(center - support + 0.5), 0);
            int stop = Math.Min((int)(center + support + 0.5), inSize);
            double total = 0.0;
            for (int j = start; j < stop; j++)
            {
                double w = Kernel((j - center + 0.5) * invStretch, filter);
                weights[(i * inSize) + j] = w;
                total += w;
            }

            if (total != 0.0)
                for (int j = start; j < stop; j++) weights[(i * inSize) + j] /= total;
        }

        return weights;
    }

    /// <summary>Resizes images [B, C, H, W] to [B, C, height, width].</summary>
    public static Tensor<T> Resize(Tensor<T> images, int height, int width, ResizeFilter filter)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        if (images.Rank != 4) throw new ArgumentException("Images must be [B, C, H, W].", nameof(images));
        int batch = images.Shape[0], channels = images.Shape[1], inHeight = images.Shape[2], inWidth = images.Shape[3];
        var alongWidth = Matrix(inWidth, width, filter);
        var alongHeight = Matrix(inHeight, height, filter);

        // Width: rows [B C H, W] times the [width, W] weights transposed -> [B C H, width].
        var x = Engine.TensorMatMulTransposed(Engine.Reshape(images.Contiguous(), new[] { batch * channels * inHeight, inWidth }), alongWidth);
        // Height: bring H last, [B C width, H] times [height, H] transposed -> [B C width, height], and back.
        x = Engine.TensorPermute(Engine.Reshape(x, new[] { batch * channels, inHeight, width }), new[] { 0, 2, 1 }).Contiguous();
        x = Engine.TensorMatMulTransposed(Engine.Reshape(x, new[] { batch * channels * width, inHeight }), alongHeight);
        x = Engine.TensorPermute(Engine.Reshape(x, new[] { batch * channels, width, height }), new[] { 0, 2, 1 }).Contiguous();
        return Engine.Reshape(x, new[] { batch, channels, height, width });
    }

    // PIL bilinear_filter / bicubic_filter (and ATen's aa_filter equivalents).
    private static double Kernel(double x, ResizeFilter filter)
    {
        x = Math.Abs(x);
        if (filter == ResizeFilter.Triangle) return x < 1.0 ? 1.0 - x : 0.0;
        if (x < 1.0) return (((CubicA + 2.0) * x) - (CubicA + 3.0)) * x * x + 1.0;
        if (x < 2.0) return ((((x - 5.0) * x) + 8.0) * x - 4.0) * CubicA;
        return 0.0;
    }

    private static Tensor<T> Matrix(int inSize, int outSize, ResizeFilter filter)
        => new Tensor<double>(new[] { outSize, inSize }, new Vector<double>(Weights(inSize, outSize, filter))).Cast<T>();
}
