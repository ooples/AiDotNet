namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological closing (dilation followed by erosion) - removes small dark spots.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Closing<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }

    public Closing(int kernelSize = 3, double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var dilate = new Dilate<T>(KernelSize, 1, probability: 1.0);
        var dilated = dilate.Apply(data, context);

        var erode = new Erode<T>(KernelSize, 1, probability: 1.0);
        return erode.Apply(dilated, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize;
        return p;
    }
}
