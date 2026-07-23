namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Equalizes the image histogram per channel (same as HistogramEqualization with per-channel mode).
/// </summary>
/// <remarks>
/// <para>This is a convenience wrapper matching the torchvision/AutoAugment Equalize operation.
/// Each color channel's histogram is equalized independently.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Equalize<T> : ImageAugmenterBase<T>
{
    private readonly HistogramEqualization<T> _equalization;

    public Equalize(double probability = 0.5) : base(probability)
    {
        _equalization = new HistogramEqualization<T>(perChannel: true, probability: 1.0);
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return _equalization.Apply(data, context);
    }
}
