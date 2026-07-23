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

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double normalized = val / maxVal;
                    double posterized = Math.Floor(normalized * (levels - 1)) / (levels - 1) * maxVal;
                    result.SetPixel(y, x, c, NumOps.FromDouble(posterized));
                }

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
