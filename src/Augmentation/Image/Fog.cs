namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates fog effect by blending the image with a fog layer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Fog<T> : ImageAugmenterBase<T>
{
    public double MinFogCoeff { get; }
    public double MaxFogCoeff { get; }
    public double MinAlpha { get; }
    public double MaxAlpha { get; }

    public Fog(double minFogCoeff = 0.3, double maxFogCoeff = 1.0,
        double minAlpha = 0.08, double maxAlpha = 0.5,
        double probability = 0.5) : base(probability)
    {
        MinFogCoeff = minFogCoeff; MaxFogCoeff = maxFogCoeff;
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double fogCoeff = context.GetRandomDouble(MinFogCoeff, MaxFogCoeff);
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);

        // Generate Perlin-like fog: a coarse random grid bilinearly interpolated over the image
        int gridSize = Math.Max(4, Math.Min(data.Height, data.Width) / 8);
        var fogMap = ImageTensor<T>.GridNoise(context, data.Height, data.Width, gridSize, 0, 1, smoothstep: false);

        var engine = Engine;
        var fogStrength = engine.TensorMultiplyScalar(
            engine.TensorMultiplyScalar(fogMap, NumOps.FromDouble(alpha)), NumOps.FromDouble(fogCoeff));
        result.SetFromPlanarTensor(ImageTensor<T>.BlendTowardsMax(data.ToPlanarTensor(), fogStrength, maxVal));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_fog_coeff"] = MinFogCoeff; p["max_fog_coeff"] = MaxFogCoeff;
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        return p;
    }
}
