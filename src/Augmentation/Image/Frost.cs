namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates frost/ice crystal patterns on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Frost<T> : ImageAugmenterBase<T>
{
    public double MinSeverity { get; }
    public double MaxSeverity { get; }

    public Frost(double minSeverity = 0.2, double maxSeverity = 0.7,
        double probability = 0.5) : base(probability)
    {
        MinSeverity = minSeverity; MaxSeverity = maxSeverity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double severity = context.GetRandomDouble(MinSeverity, MaxSeverity);

        // Generate frost pattern using diamond-square-like noise
        int gridSize = Math.Max(4, Math.Min(data.Height, data.Width) / 16);

        // Multiple octaves for realistic frost texture (octave k: cells of gridSize * 2^k, weight 1 / 2^k)
        Tensor<T>? pattern = null;
        T[] octaveWeights = ImageTensor<T>.OctaveWeights(3);
        for (int octave = 0; octave < 3; octave++)
        {
            int currentGrid = Math.Max(2, gridSize * (1 << octave));
            var layer = Engine.TensorMultiplyScalar(
                ImageTensor<T>.GridNoise(context, data.Height, data.Width, currentGrid, 0, 1, smoothstep: false),
                octaveWeights[octave]);
            pattern = pattern is null ? layer : Engine.TensorAdd(pattern, layer);
        }

        // Normalize frost pattern to [0, 1]
        var engine = Engine;
        pattern ??= new Tensor<T>([data.Height, data.Width]);
        T minP = engine.TensorMinValue(pattern);
        T maxP = engine.TensorMaxValue(pattern);

        T range = NumOps.Subtract(maxP, minP);
        if (NumOps.LessThan(range, NumOps.FromDouble(1e-10))) range = NumOps.One;

        // Sharpen the normalized pattern (square it) and scale by severity
        var frost = engine.TensorDivideScalar(engine.TensorSubtractScalar(pattern, minP), range);
        var alpha = engine.TensorMultiplyScalar(engine.TensorMultiply(frost, frost), NumOps.FromDouble(severity));
        result.SetFromPlanarTensor(ImageTensor<T>.BlendTowardsMax(data.ToPlanarTensor(), alpha, maxVal));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_severity"] = MinSeverity; p["max_severity"] = MaxSeverity;
        return p;
    }
}
