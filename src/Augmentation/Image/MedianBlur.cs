namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies median filter blur, effective at removing salt-and-pepper noise while preserving edges.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MedianBlur<T> : ImageAugmenterBase<T>
{
    public int MinKernelSize { get; }
    public int MaxKernelSize { get; }

    public MedianBlur(int minKernelSize = 3, int maxKernelSize = 5, double probability = 0.5)
        : base(probability)
    {
        if (minKernelSize < 3 || minKernelSize % 2 == 0) throw new ArgumentException("Kernel size must be odd >= 3.");
        if (maxKernelSize < minKernelSize || maxKernelSize % 2 == 0) throw new ArgumentException("Max kernel must be odd >= min.");
        MinKernelSize = minKernelSize;
        MaxKernelSize = maxKernelSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Pick random odd kernel size
        int range = (MaxKernelSize - MinKernelSize) / 2 + 1;
        int kSize = MinKernelSize + context.GetRandomInt(0, range) * 2;
        int half = kSize / 2;
        var result = data.Clone();

        // Median of the edge-replicated neighbourhood: sort each window and take the middle element
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var windows = ImageTensor<T>.Neighbourhoods(image, kSize);
        var sorted = engine.TensorSort(windows, 3, false).Values;
        int count = kSize * kSize;
        var median = engine.TensorSliceAxis(sorted, 3, count / 2);
        result.SetFromPlanarTensor(median);

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_kernel_size"] = MinKernelSize; p["max_kernel_size"] = MaxKernelSize;
        return p;
    }
}
