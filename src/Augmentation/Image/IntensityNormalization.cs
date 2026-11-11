namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Medical image intensity normalization using z-score or min-max normalization.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class IntensityNormalization<T> : ImageAugmenterBase<T>
{
    public IntensityNormMethod Method { get; }
    public double ClipMin { get; }
    public double ClipMax { get; }

    public IntensityNormalization(IntensityNormMethod method = IntensityNormMethod.ZScore,
        double clipMin = -3.0, double clipMax = 3.0,
        double probability = 1.0) : base(probability)
    {
        Method = method; ClipMin = clipMin; ClipMax = clipMax;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Per-channel statistics over all pixels
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var mean = engine.ReduceMean(image, [1, 2], true);

        if (Method == IntensityNormMethod.ZScore)
        {
            var centered = engine.TensorSubtract(image, mean);
            var variance = engine.ReduceMean(engine.TensorMultiply(centered, centered), [1, 2], true);
            var std = engine.TensorSqrt(engine.TensorAddScalar(variance, NumOps.FromDouble(1e-8)));

            var z = engine.TensorClamp(engine.TensorDivide(centered, std), NumOps.FromDouble(ClipMin), NumOps.FromDouble(ClipMax));
            // Map back to [0, maxVal]
            var mapped = engine.TensorMultiplyScalar(
                engine.TensorDivideScalar(engine.TensorSubtractScalar(z, NumOps.FromDouble(ClipMin)), NumOps.FromDouble(ClipMax - ClipMin)),
                NumOps.FromDouble(maxVal));
            result.SetFromPlanarTensor(mapped);
        }
        else // MinMax
        {
            var min = engine.TensorAmin(image, [1, 2], true);
            var max = engine.ReduceMax(image, [1, 2], true);
            var range = engine.TensorSubtract(max, min);
            var flat = engine.TensorLessThan(range, NumOps.FromDouble(1e-10));
            range = engine.TensorWhere(flat, ImageTensor<T>.Constant(range, 1.0), range);
            var mapped = engine.TensorMultiplyScalar(
                engine.TensorDivide(engine.TensorSubtract(image, min), range), NumOps.FromDouble(maxVal));
            result.SetFromPlanarTensor(mapped);
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["method"] = Method.ToString();
        p["clip_min"] = ClipMin; p["clip_max"] = ClipMax;
        return p;
    }
}

/// <summary>Intensity normalization method.</summary>
public enum IntensityNormMethod
{
    /// <summary>Z-score normalization (zero mean, unit variance).</summary>
    ZScore,
    /// <summary>Min-max normalization to [0, 1] range.</summary>
    MinMax
}
