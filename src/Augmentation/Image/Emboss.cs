namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies an emboss filter to create a 3D shadow effect.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Emboss<T> : ImageAugmenterBase<T>
{
    public double MinAlpha { get; }
    public double MaxAlpha { get; }
    public double MinStrength { get; }
    public double MaxStrength { get; }

    public Emboss(double minAlpha = 0.2, double maxAlpha = 0.5,
        double minStrength = 0.5, double maxStrength = 1.0,
        double probability = 0.5) : base(probability)
    {
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
        MinStrength = minStrength; MaxStrength = maxStrength;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);
        double strength = context.GetRandomDouble(MinStrength, MaxStrength);

        // Emboss kernel:
        // [-1*s, -s, 0]
        // [-s,    1, s]
        // [0,     s, 1*s]
        double[,] kernel = {
            { -strength, -strength, 0 },
            { -strength,  1,        strength },
            { 0,          strength,  strength }
        };

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var embossed = ImageTensor<T>.Correlate(image, ImageTensor<T>.KernelTensor(kernel), mirror: false);
        var blended = engine.TensorAdd(
            engine.TensorMultiplyScalar(image, NumOps.FromDouble(1 - alpha)),
            engine.TensorMultiplyScalar(embossed, NumOps.FromDouble(alpha)));
        result.SetFromPlanarTensor(engine.TensorClamp(blended, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        p["min_strength"] = MinStrength; p["max_strength"] = MaxStrength;
        return p;
    }
}
