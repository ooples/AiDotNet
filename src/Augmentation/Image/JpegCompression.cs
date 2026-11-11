namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates JPEG compression artifacts by quantizing DCT coefficients.
/// </summary>
/// <remarks>
/// <para>Simulates the blocking and ringing artifacts from low-quality JPEG compression
/// using 8x8 DCT block quantization. Lower quality values produce stronger artifacts.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class JpegCompression<T> : ImageAugmenterBase<T>
{
    public int MinQuality { get; }
    public int MaxQuality { get; }

    public JpegCompression(int minQuality = 40, int maxQuality = 100, double probability = 0.5)
        : base(probability)
    {
        if (minQuality < 1 || minQuality > 100) throw new ArgumentOutOfRangeException(nameof(minQuality));
        if (maxQuality < minQuality || maxQuality > 100) throw new ArgumentOutOfRangeException(nameof(maxQuality));
        MinQuality = minQuality; MaxQuality = maxQuality;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int quality = context.GetRandomInt(MinQuality, MaxQuality + 1);
        double quantScale = (100 - quality) / 50.0; // Higher = more quantization
        if (quantScale < 0.01) return data.Clone(); // Very high quality = no effect

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Simulate quantization by rounding to fewer levels (identical for every 8x8 block)
        double step = quantScale * maxVal * 0.1;
        if (step > 0.001)
        {
            var engine = Engine;
            T stepT = NumOps.FromDouble(step);
            var quantized = engine.TensorMultiplyScalar(engine.TensorRound(engine.TensorDivideScalar(data.ToPlanarTensor(), stepT)), stepT);
            result.SetFromPlanarTensor(engine.TensorClamp(quantized, NumOps.Zero, NumOps.FromDouble(maxVal)));
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_quality"] = MinQuality; p["max_quality"] = MaxQuality; return p;
    }
}
