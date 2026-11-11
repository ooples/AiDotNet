namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates sun flare / lens flare effect on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SunFlare<T> : ImageAugmenterBase<T>
{
    public double MinRadius { get; }
    public double MaxRadius { get; }
    public int NumFlareCircles { get; }

    public SunFlare(double minRadius = 0.1, double maxRadius = 0.4,
        int numFlareCircles = 5, double probability = 0.5) : base(probability)
    {
        MinRadius = minRadius; MaxRadius = maxRadius;
        NumFlareCircles = numFlareCircles;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Sun position (typically upper half)
        int sunY = context.GetRandomInt(0, data.Height / 2);
        int sunX = context.GetRandomInt(0, data.Width);
        double mainRadius = context.GetRandomDouble(MinRadius, MaxRadius) * Math.Min(data.Height, data.Width);

        // Draw main sun glow
        DrawGlow(result, sunY, sunX, mainRadius, maxVal, 0.8);

        // Draw secondary flare circles along a line from center
        int centerY = data.Height / 2;
        int centerX = data.Width / 2;

        for (int i = 0; i < NumFlareCircles; i++)
        {
            double t = context.GetRandomDouble(0.3, 1.5);
            int flareY = (int)(sunY + (centerY - sunY) * t);
            int flareX = (int)(sunX + (centerX - sunX) * t);
            double flareRadius = mainRadius * context.GetRandomDouble(0.1, 0.5);

            if (flareY >= 0 && flareY < data.Height && flareX >= 0 && flareX < data.Width)
                DrawGlow(result, flareY, flareX, flareRadius, maxVal, 0.3);
        }

        return result;
    }

    private static void DrawGlow(ImageTensor<T> image, int centerY, int centerX,
        double radius, double maxVal, double intensity)
    {
        // Pixels within the radius brighten by (1 - dist / radius) * intensity * maxVal
        var engine = AiDotNetEngine.Current;
        T r = NumOps.FromDouble(radius);
        var dy = engine.Reshape(engine.TensorSubtractScalar(engine.TensorArange<T>(0, image.Height, 1), NumOps.FromDouble(centerY)), [image.Height, 1]);
        var dx = engine.Reshape(engine.TensorSubtractScalar(engine.TensorArange<T>(0, image.Width, 1), NumOps.FromDouble(centerX)), [1, image.Width]);
        var dist = engine.TensorSqrt(engine.TensorAdd(engine.TensorMultiply(dy, dy), engine.TensorMultiply(dx, dx)));
        var inside = engine.ScalarMinusTensor(NumOps.One, engine.TensorGreaterThan(dist, r));
        var alpha = engine.TensorMultiplyScalar(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorDivideScalar(dist, r)), NumOps.FromDouble(intensity));

        var planar = image.ToPlanarTensor();
        int[] shape = planar._shape;
        var boost = engine.Reshape(engine.TensorMultiplyScalar(alpha, NumOps.FromDouble(maxVal)), [1, image.Height, image.Width]);
        var glowed = engine.TensorClamp(engine.TensorAdd(planar, boost), NumOps.Zero, NumOps.FromDouble(maxVal));
        var mask = engine.TensorBroadcastTo(engine.Reshape(inside, [1, image.Height, image.Width]), shape);
        image.SetFromPlanarTensor(engine.TensorWhere(mask, glowed, planar));
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_radius"] = MinRadius; p["max_radius"] = MaxRadius;
        p["num_flare_circles"] = NumFlareCircles;
        return p;
    }
}
