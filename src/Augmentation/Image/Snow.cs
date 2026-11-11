namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates snow effect on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Snow<T> : ImageAugmenterBase<T>
{
    public double MinSnowPoint { get; }
    public double MaxSnowPoint { get; }
    public double MinBrightness { get; }
    public double MaxBrightness { get; }

    public Snow(double minSnowPoint = 0.1, double maxSnowPoint = 0.3,
        double minBrightness = 1.5, double maxBrightness = 2.5,
        double probability = 0.5) : base(probability)
    {
        MinSnowPoint = minSnowPoint; MaxSnowPoint = maxSnowPoint;
        MinBrightness = minBrightness; MaxBrightness = maxBrightness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double snowPoint = context.GetRandomDouble(MinSnowPoint, MaxSnowPoint);
        double brightness = context.GetRandomDouble(MinBrightness, MaxBrightness);

        // Add snow particles
        // Mark snow particles (RNG draws in the original order); they become fully white
        int numParticles = (int)(snowPoint * data.Height * data.Width * 0.1);
        var snowMask = new int[data.Height * data.Width];
        for (int i = 0; i < numParticles; i++)
        {
            int y = context.GetRandomInt(0, data.Height);
            int x = context.GetRandomInt(0, data.Width);
            int radius = context.GetRandomInt(1, 3);

            for (int dy = -radius; dy <= radius; dy++)
                for (int dx = -radius; dx <= radius; dx++)
                {
                    int ny = y + dy; int nx = x + dx;
                    if (ny < 0 || ny >= data.Height || nx < 0 || nx >= data.Width) continue;
                    if (dy * dy + dx * dx > radius * radius) continue;
                    snowMask[ny * data.Width + nx] = 1;
                }
        }

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var mask = engine.TensorBroadcastTo(
            new Tensor<int>(snowMask, [1, data.Height, data.Width]).Cast<T>(), image._shape);
        var snowed = engine.TensorWhere(mask, ImageTensor<T>.Constant(image, maxVal), image);

        // Brighten pixels whose mean luminance (normalized) exceeds 1 - snowPoint
        var normLum = engine.TensorDivideScalar(engine.ReduceMean(snowed, [0], true), NumOps.FromDouble(maxVal));
        var bright = engine.TensorBroadcastTo(engine.TensorGreaterThan(normLum, NumOps.FromDouble(1 - snowPoint)), image._shape);
        var brightened = engine.TensorClamp(
            engine.TensorMultiplyScalar(snowed, NumOps.FromDouble(brightness)), NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(engine.TensorWhere(bright, brightened, snowed));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_snow_point"] = MinSnowPoint; p["max_snow_point"] = MaxSnowPoint;
        p["min_brightness"] = MinBrightness; p["max_brightness"] = MaxBrightness;
        return p;
    }
}
