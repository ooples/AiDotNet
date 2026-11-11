namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates spatter effects (mud, rain drops) on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Spatter<T> : ImageAugmenterBase<T>
{
    public double Intensity { get; }
    public double MinSpotSize { get; }
    public double MaxSpotSize { get; }
    public SpatterMode Mode { get; }

    public Spatter(double intensity = 0.3, double minSpotSize = 0.01, double maxSpotSize = 0.05,
        SpatterMode mode = SpatterMode.Mud, double probability = 0.5) : base(probability)
    {
        Intensity = intensity; MinSpotSize = minSpotSize; MaxSpotSize = maxSpotSize; Mode = mode;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numSpots = (int)(Intensity * 50);

        var engine = Engine;
        var image = data.ToPlanarTensor();
        int channels = data.Channels, height = data.Height, width = data.Width;
        var spotColor = SpotColor(engine, channels, maxVal);
        var rows = engine.Reshape(engine.TensorArange<int>(0, height, 1), [height, 1]);
        var cols = engine.Reshape(engine.TensorArange<int>(0, width, 1), [1, width]);

        for (int s = 0; s < numSpots; s++)
        {
            double spotRadius = context.GetRandomDouble(MinSpotSize, MaxSpotSize) *
                                Math.Min(data.Height, data.Width);
            int centerY = context.GetRandomInt(0, data.Height);
            int centerX = context.GetRandomInt(0, data.Width);

            image = ApplySpot(engine, image, spotColor, rows, cols, spotRadius, centerY, centerX, maxVal, context);
        }

        result.SetFromPlanarTensor(image);

        return result;
    }

    /// <summary>
    /// Blends one spot of radius <paramref name="spotRadius"/> centred at (centerY, centerX) into a
    /// [channels, height, width] image, drawing one U(0.5, 1) weight per inside pixel in y, x order.
    /// </summary>
    private static Tensor<T> ApplySpot(IEngine engine, Tensor<T> image, Tensor<T> spotColor, Tensor<int> rows, Tensor<int> cols,
        double spotRadius, int centerY, int centerX, double maxVal, AugmentationContext<T> context)
    {
        int height = image.Shape[1], width = image.Shape[2];
        T one = NumOps.One;

        // A pixel is inside when sqrt(dy^2 + dx^2) <= spotRadius; with integer offsets that is
        // dy^2 + dx^2 <= limit for the largest integer limit whose square root is within the radius
        int limit = (int)Math.Floor(spotRadius * spotRadius);
        while (Math.Sqrt(limit + 1.0) <= spotRadius) limit++;
        while (limit >= 0 && Math.Sqrt(limit) > spotRadius) limit--;

        var dy = engine.TensorSubtractScalar(rows, centerY);
        var dx = engine.TensorSubtractScalar(cols, centerX);
        var distSq = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.TensorMultiply(dy, dy), [height, width]),
            engine.TensorBroadcastTo(engine.TensorMultiply(dx, dx), [height, width]));
        var insideFlat = engine.Reshape(engine.TensorLessThan(distSq, limit + 1), [height * width]);
        int insideCount = engine.TensorSum(insideFlat);
        if (insideCount == 0)
        {
            return image;
        }

        // One U(0.5, 1) draw per inside pixel in y, x order, scattered back to its pixel
        var draws = AugmentationRandom<T>.Uniform(context, [insideCount], 0.5, 1.0);
        var rank = engine.TensorClamp(engine.TensorSubtractScalar(engine.TensorCumSum(insideFlat, 0), 1), 0, insideCount - 1);
        var inside = engine.Reshape(insideFlat.Cast<T>(), [1, height, width]);
        var pixelDraw = engine.Reshape(engine.TensorTake(draws, rank), [1, height, width]);

        // alpha = (1 - dist / spotRadius) * draw inside the spot, 0 outside
        var dist = engine.TensorSqrt(engine.Reshape(distSq.Cast<T>(), [1, height, width]));
        var falloff = engine.ScalarMinusTensor(one, engine.TensorDivideScalar(dist, NumOps.FromDouble(spotRadius)));
        var a3 = engine.TensorMultiply(engine.TensorMultiply(falloff, pixelDraw), inside);

        // Blend the spot color into the pixels inside the spot
        var blended = engine.TensorClamp(
            engine.TensorAdd(
                engine.TensorMultiply(image, engine.ScalarMinusTensor(one, a3)),
                engine.TensorMultiply(spotColor, a3)),
            NumOps.Zero, NumOps.FromDouble(maxVal));
        var mask = engine.TensorBroadcastTo(inside, image._shape);
        return engine.TensorWhere(mask, blended, image);
    }

    /// <summary>
    /// Spot color per channel as a [channels, 1, 1] tensor: mud is brownish (0.4, 0.3, then 0.15 of the
    /// maximum), rain is bright (0.85), anything else 0.4.
    /// </summary>
    private Tensor<T> SpotColor(IEngine engine, int channels, double maxVal)
    {
        var shape = new Tensor<T>([channels, 1, 1]);
        if (Mode != SpatterMode.Mud)
        {
            return ImageTensor<T>.Constant(shape, (Mode == SpatterMode.Rain ? 0.85 : 0.4) * maxVal);
        }

        var channel = engine.Reshape(engine.TensorArange<T>(0, channels, 1), [channels, 1, 1]);
        return engine.TensorWhere(engine.TensorEquals(channel, NumOps.Zero), ImageTensor<T>.Constant(shape, 0.4 * maxVal),
            engine.TensorWhere(engine.TensorEquals(channel, NumOps.One), ImageTensor<T>.Constant(shape, 0.3 * maxVal),
                ImageTensor<T>.Constant(shape, 0.15 * maxVal)));
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["intensity"] = Intensity; p["mode"] = Mode.ToString();
        p["min_spot_size"] = MinSpotSize; p["max_spot_size"] = MaxSpotSize;
        return p;
    }
}

/// <summary>
/// Spatter effect type.
/// </summary>
public enum SpatterMode
{
    /// <summary>Mud-like brown spatter.</summary>
    Mud,
    /// <summary>Rain drop-like transparent spatter.</summary>
    Rain
}
