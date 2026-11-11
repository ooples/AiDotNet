namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates WebP compression artifacts (similar to JPEG but with different characteristics).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WebPCompression<T> : ImageAugmenterBase<T>
{
    public int MinQuality { get; }
    public int MaxQuality { get; }

    public WebPCompression(int minQuality = 40, int maxQuality = 100, double probability = 0.5)
        : base(probability)
    {
        if (minQuality < 1 || minQuality > 100) throw new ArgumentOutOfRangeException(nameof(minQuality));
        if (maxQuality < minQuality || maxQuality > 100) throw new ArgumentOutOfRangeException(nameof(maxQuality));
        MinQuality = minQuality; MaxQuality = maxQuality;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int quality = context.GetRandomInt(MinQuality, MaxQuality + 1);
        double quantScale = (100 - quality) / 60.0; // WebP is slightly less aggressive than JPEG
        if (quantScale < 0.01) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // WebP uses 4x4 transform blocks (vs JPEG's 8x8); the quantization step is the same for every block
        double step = quantScale * maxVal * 0.08;
        if (step > 0.001)
        {
            var engine = Engine;
            T stepT = NumOps.FromDouble(step);
            var quantized = engine.TensorMultiplyScalar(engine.TensorRound(engine.TensorDivideScalar(data.ToPlanarTensor(), stepT)), stepT);
            result.SetFromPlanarTensor(engine.TensorClamp(quantized, NumOps.Zero, NumOps.FromDouble(maxVal)));
        }

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_quality"] = MinQuality; p["max_quality"] = MaxQuality; return p;
    }
}
