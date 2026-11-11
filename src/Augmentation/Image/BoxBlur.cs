namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies simple box (mean) blur using a uniform kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BoxBlur<T> : ImageAugmenterBase<T>
{
    public int MinKernelSize { get; }
    public int MaxKernelSize { get; }

    public BoxBlur(int minKernelSize = 3, int maxKernelSize = 5, double probability = 0.5)
        : base(probability)
    {
        if (minKernelSize < 1) throw new ArgumentOutOfRangeException(nameof(minKernelSize));
        if (maxKernelSize < minKernelSize) throw new ArgumentOutOfRangeException(nameof(maxKernelSize), "maxKernelSize must be >= minKernelSize.");
        MinKernelSize = minKernelSize; MaxKernelSize = maxKernelSize;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int kSize = context.GetRandomInt(MinKernelSize, MaxKernelSize + 1);
        if (kSize % 2 == 0) kSize++;
        int half = kSize / 2;
        double weight = 1.0 / ((double)kSize * kSize);
        var result = data.Clone();

        // Sum the k x k edge-replicated neighbourhood, then scale by 1 / k^2
        var engine = Engine;
        var ones = new Tensor<T>([kSize, kSize]);
        engine.TensorFill(ones, NumOps.One);
        var sums = ImageTensor<T>.Correlate(data.ToPlanarTensor(), ones, mirror: false);
        result.SetFromPlanarTensor(engine.TensorMultiplyScalar(sums, NumOps.FromDouble(weight)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_kernel_size"] = MinKernelSize; p["max_kernel_size"] = MaxKernelSize; return p;
    }
}
