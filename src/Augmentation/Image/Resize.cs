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

        // Every interpolation mode is separable: build per-axis weight matrices and apply them
        var engine = Engine;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var rowWeights = AxisWeights(engine, TargetHeight, data.Height, scaleY, Interpolation);
        var colWeights = AxisWeights(engine, TargetWidth, data.Width, scaleX, Interpolation);
        var resized = ImageTensor<T>.ResampleSeparable(data.ToPlanarTensor(), rowWeights, colWeights);

        // Clamp to valid range (bicubic and Lanczos can overshoot)
        result.SetFromPlanarTensor(engine.TensorClamp(resized, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <summary>
    /// [target, source] interpolation weights along one axis for the given mode.
    /// </summary>
    private static Tensor<T> AxisWeights(IEngine engine, int target, int source, double scale, InterpolationMode mode)
    {
        var dest = engine.TensorArange<T>(0, target, 1);
        T last = NumOps.FromDouble(source - 1.0);
        T half = NumOps.FromDouble(0.5);

        // Map target pixel to source pixel centers for accurate interpolation
        var src = engine.TensorSubtractScalar(
            engine.TensorMultiplyScalar(engine.TensorAddScalar(dest, half), NumOps.FromDouble(scale)), half);
        var floor = engine.TensorFloor(src);
        var srcColumn = engine.Reshape(src, [target, 1]);
        var floorColumn = engine.Reshape(floor, [target, 1]);

        switch (mode)
        {
            case InterpolationMode.Nearest:
                return Tap(engine, engine.TensorClamp(engine.TensorRound(srcColumn), NumOps.Zero, last),
                    ImageTensor<T>.Constant(srcColumn, 1.0), source);

            case InterpolationMode.Bicubic:
            {
                // Taps at floor - 1 .. floor + 2 with Keys weights of (frac - d)
                var offsets = TapOffsets(engine, target, -1, 2);
                int[] shape = offsets._shape;
                var index = engine.TensorClamp(engine.TensorAdd(engine.TensorBroadcastTo(floorColumn, shape), offsets), NumOps.Zero, last);
                var frac = engine.TensorSubtract(srcColumn, floorColumn);
                var weights = CubicWeights(engine, engine.TensorSubtract(engine.TensorBroadcastTo(frac, shape), offsets));
                return Tap(engine, index, weights, source);
            }

            case InterpolationMode.Area:
            {
                // Average over source pixels floor(d * scale) .. ceil((d + 1) * scale) - 1
                T scaleT = NumOps.FromDouble(scale);
                var lo = engine.TensorClamp(engine.TensorFloor(engine.TensorMultiplyScalar(dest, scaleT)), NumOps.Zero, last);
                var hi = engine.TensorClamp(
                    engine.TensorSubtractScalar(
                        engine.TensorCeiling(engine.TensorMultiplyScalar(engine.TensorAddScalar(dest, NumOps.One), scaleT)), NumOps.One),
                    NumOps.Zero, last);
                int[] shape = [target, source];
                var positions = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, source, 1), [1, source]), shape);
                var loB = engine.TensorBroadcastTo(engine.Reshape(lo, [target, 1]), shape);
                var hiB = engine.TensorBroadcastTo(engine.Reshape(hi, [target, 1]), shape);
                var member = engine.TensorMultiply(
                    engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(positions, loB)),
                    engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(positions, hiB)));
                var count = engine.Reshape(engine.TensorAddScalar(engine.TensorSubtract(hi, lo), NumOps.One), [target, 1]);
                return engine.TensorDivide(member, engine.TensorBroadcastTo(count, shape));
            }

            case InterpolationMode.Lanczos:
            {
                // Lanczos-3 taps at floor - 2 .. floor + 3, normalized to sum to 1 (0 when they sum to 0)
                const int a = 3;
                var offsets = TapOffsets(engine, target, -(a - 1), a);
                int[] shape = offsets._shape;
                var offset = engine.TensorAdd(engine.TensorBroadcastTo(floorColumn, shape), offsets);
                var index = engine.TensorClamp(offset, NumOps.Zero, last);
                var raw = Tap(engine, index,
                    LanczosWeights(engine, engine.TensorSubtract(engine.TensorBroadcastTo(srcColumn, shape), offset), a), source);
                var rowSum = engine.ReduceSum(raw, [1], true);
                var positive = engine.TensorBroadcastTo(engine.TensorGreaterThan(rowSum, NumOps.Zero), [target, source]);
                return engine.TensorWhere(positive,
                    engine.TensorDivide(raw, engine.TensorBroadcastTo(rowSum, [target, source])),
                    new Tensor<T>([target, source]));
            }

            default:
            {
                // Bilinear
                var i0 = engine.TensorClamp(floorColumn, NumOps.Zero, last);
                var i1 = engine.TensorClamp(engine.TensorAddScalar(i0, NumOps.One), NumOps.Zero, last);
                var frac = engine.TensorSubtract(srcColumn, floorColumn);
                return Tap(engine,
                    engine.TensorConcatenate([i0, i1], 1),
                    engine.TensorConcatenate([engine.ScalarMinusTensor(NumOps.One, frac), frac], 1),
                    source);
            }
        }
    }

    /// <summary>
    /// Tap offsets first .. last as a [target, taps] tensor (the same row for every target).
    /// </summary>
    private static Tensor<T> TapOffsets(IEngine engine, int target, int first, int last)
    {
        int taps = last - first + 1;
        var row = engine.Reshape(engine.TensorArange<T>(first, last + 1, 1), [1, taps]);
        return engine.TensorBroadcastTo(row, [target, taps]).Contiguous();
    }

    /// <summary>
    /// [target, source] matrix summing weight[t, k] at column index[t, k] over the taps k of every row t.
    /// </summary>
    private static Tensor<T> Tap(IEngine engine, Tensor<T> index, Tensor<T> weight, int source)
    {
        int target = index.Shape[0], taps = index.Shape[1];
        int[] shape = [target, taps, source];
        var positions = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<T>(0, source, 1), [1, 1, source]), shape);
        var hit = engine.TensorEquals(positions, engine.TensorBroadcastTo(engine.Reshape(index, [target, taps, 1]), shape));
        var weighted = engine.TensorMultiply(hit, engine.TensorBroadcastTo(engine.Reshape(weight, [target, taps, 1]), shape));
        return engine.ReduceSum(weighted, [1], false);
    }

    /// <summary>
    /// Keys cubic kernel (a = -0.5): 1.5|t|^3 - 2.5|t|^2 + 1 for |t| &lt;= 1,
    /// -0.5|t|^3 + 2.5|t|^2 - 4|t| + 2 for |t| &lt;= 2, else 0.
    /// </summary>
    private static Tensor<T> CubicWeights(IEngine engine, Tensor<T> t)
    {
        var a = engine.TensorAbs(t);
        var a2 = engine.TensorMultiply(a, a);
        var a3 = engine.TensorMultiply(a2, a);
        T c15 = NumOps.FromDouble(1.5), c25 = NumOps.FromDouble(2.5);
        var near = engine.TensorAddScalar(
            engine.TensorSubtract(engine.TensorMultiplyScalar(a3, c15), engine.TensorMultiplyScalar(a2, c25)), NumOps.One);
        var far = engine.TensorAddScalar(
            engine.TensorSubtract(
                engine.TensorAdd(engine.TensorMultiplyScalar(a3, NumOps.FromDouble(-0.5)), engine.TensorMultiplyScalar(a2, c25)),
                engine.TensorMultiplyScalar(a, NumOps.FromDouble(4.0))),
            NumOps.FromDouble(2.0));
        var zero = ImageTensor<T>.Constant(a, 0.0);
        return engine.TensorWhere(engine.TensorGreaterThan(a, NumOps.FromDouble(2.0)), zero,
            engine.TensorWhere(engine.TensorGreaterThan(a, NumOps.One), far, near));
    }

    /// <summary>
    /// Lanczos kernel: 1 for |x| &lt; 1e-8, 0 for |x| &gt;= a, else a sin(pi x) sin(pi x / a) / (pi x)^2.
    /// </summary>
    private static Tensor<T> LanczosWeights(IEngine engine, Tensor<T> x, int a)
    {
        T aT = NumOps.FromDouble(a);
        var absX = engine.TensorAbs(x);
        var pix = engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.PI));
        var value = engine.TensorDivide(
            engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorSin(pix), engine.TensorSin(engine.TensorDivideScalar(pix, aT))), aT),
            engine.TensorMultiply(pix, pix));
        var outside = ImageTensor<T>.Constant(x, 0.0);
        var center = ImageTensor<T>.Constant(x, 1.0);
        return engine.TensorWhere(engine.TensorLessThan(absX, NumOps.FromDouble(1e-8)), center,
            engine.TensorWhere(engine.TensorLessThan(absX, aT), value, outside));
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
