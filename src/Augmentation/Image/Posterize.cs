namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Reduces the number of bits per color channel, creating a poster-like effect.
/// </summary>
/// <remarks>
/// <para>Posterization quantizes pixel values to fewer discrete levels, creating flat areas
/// of color. This simulates low-bit-depth imaging and teaches robustness to quantization.</para>
/// <para><b>For Beginners:</b> Reducing from 8 bits (256 levels) to 4 bits (16 levels) per
/// channel makes the image look like a poster with fewer, more distinct colors.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Posterize<T> : ImageAugmenterBase<T>
{
    public int MinBitsPerChannel { get; }
    public int MaxBitsPerChannel { get; }

    public Posterize(int minBitsPerChannel = 4, int maxBitsPerChannel = 8, double probability = 0.5)
        : base(probability)
    {
        if (minBitsPerChannel < 1 || minBitsPerChannel > 8)
            throw new ArgumentOutOfRangeException(nameof(minBitsPerChannel), "Must be 1-8.");
        if (maxBitsPerChannel < minBitsPerChannel || maxBitsPerChannel > 8)
            throw new ArgumentOutOfRangeException(nameof(maxBitsPerChannel));
        MinBitsPerChannel = minBitsPerChannel;
        MaxBitsPerChannel = maxBitsPerChannel;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int bits = context.GetRandomInt(MinBitsPerChannel, MaxBitsPerChannel + 1);
        int levels = 1 << bits;
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        var engine = Engine;
        var normalized = engine.TensorDivideScalar(data.ToPlanarTensor(), NumOps.FromDouble(maxVal));
        var quantized = engine.TensorFloor(engine.TensorMultiplyScalar(normalized, NumOps.FromDouble(levels - 1)));
        var posterized = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(quantized, NumOps.FromDouble(levels - 1)), NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(posterized);

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_bits"] = MinBitsPerChannel;
        p["max_bits"] = MaxBitsPerChannel;
        return p;
    }
}
