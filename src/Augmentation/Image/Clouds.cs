namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates cloud overlay on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Clouds<T> : ImageAugmenterBase<T>
{
    public double MinOpacity { get; }
    public double MaxOpacity { get; }
    public double MinScale { get; }
    public double MaxScale { get; }

    public Clouds(double minOpacity = 0.2, double maxOpacity = 0.6,
        double minScale = 0.05, double maxScale = 0.2,
        double probability = 0.5) : base(probability)
    {
        MinOpacity = minOpacity; MaxOpacity = maxOpacity;
        MinScale = minScale; MaxScale = maxScale;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double opacity = context.GetRandomDouble(MinOpacity, MaxOpacity);
        double scale = context.GetRandomDouble(MinScale, MaxScale);
        int gridSize = Math.Max(2, (int)(Math.Min(data.Height, data.Width) * scale));

        // Generate cloud noise map with multiple octaves (octave k: cells of gridSize / 2^k, weight 1 / 2^k)
        Tensor<T>? map = null;
        T[] octaveWeights = ImageTensor<T>.OctaveWeights(4);
        for (int octave = 0; octave < 4; octave++)
        {
            int gs = Math.Max(2, gridSize / (1 << octave));
            var layer = Engine.TensorMultiplyScalar(
                ImageTensor<T>.GridNoise(context, data.Height, data.Width, gs, 0, 1, smoothstep: false),
                octaveWeights[octave]);
            map = map is null ? layer : Engine.TensorAdd(map, layer);
        }

        // Normalize and threshold
        var engine = Engine;
        map ??= new Tensor<T>([data.Height, data.Width]);
        T minC = engine.TensorMinValue(map);
        T maxC = engine.TensorMaxValue(map);

        T rangeC = NumOps.Subtract(maxC, minC);
        if (NumOps.LessThan(rangeC, NumOps.FromDouble(1e-10))) rangeC = NumOps.One;

        // cloud = max(0, normalized - 0.3) / 0.7 (threshold), weighted by opacity
        var normalized = engine.TensorDivideScalar(engine.TensorSubtractScalar(map, minC), rangeC);
        var cloud = engine.TensorDivideScalar(
            engine.TensorMax(engine.TensorSubtractScalar(normalized, NumOps.FromDouble(0.3)), NumOps.Zero), NumOps.FromDouble(0.7));
        var alpha = engine.TensorMultiplyScalar(cloud, NumOps.FromDouble(opacity));
        result.SetFromPlanarTensor(ImageTensor<T>.BlendTowardsMax(data.ToPlanarTensor(), alpha, maxVal));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_opacity"] = MinOpacity; p["max_opacity"] = MaxOpacity;
        p["min_scale"] = MinScale; p["max_scale"] = MaxScale;
        return p;
    }
}
