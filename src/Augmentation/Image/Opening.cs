namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological opening (erosion followed by dilation) - removes small bright spots.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Opening<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }

    public Opening(int kernelSize = 3, double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var erode = new Erode<T>(KernelSize, 1, probability: 1.0);
        var eroded = erode.Apply(data, context);

        var dilate = new Dilate<T>(KernelSize, 1, probability: 1.0);
        return dilate.Apply(eroded, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize;
        return p;
    }
}
