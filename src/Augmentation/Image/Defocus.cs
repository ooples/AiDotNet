namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates defocus blur using a circular (disc) kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Defocus<T> : ImageAugmenterBase<T>
{
    public int MinRadius { get; }
    public int MaxRadius { get; }

    public Defocus(int minRadius = 1, int maxRadius = 3, double probability = 0.5)
        : base(probability)
    {
        if (minRadius < 1) throw new ArgumentOutOfRangeException(nameof(minRadius));
        MinRadius = minRadius; MaxRadius = maxRadius;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int radius = context.GetRandomInt(MinRadius, MaxRadius + 1);
        int kSize = radius * 2 + 1;
        var engine = Engine;

        // Create circular kernel: taps within the radius, normalized to sum to 1
        var disk = engine.TensorLessThan(ImageTensor<T>.KernelRadiusSquared(kSize), radius * radius + 1).Cast<T>();
        var kernel = ImageTensor<T>.NormalizeKernel(disk);

        var result = data.Clone();
        result.SetFromPlanarTensor(
            ImageTensor<T>.Correlate(data.ToPlanarTensor(), kernel, mirror: false));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_radius"] = MinRadius; p["max_radius"] = MaxRadius; return p;
    }
}
