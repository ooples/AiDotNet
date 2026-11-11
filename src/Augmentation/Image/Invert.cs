namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Inverts all pixel values in the image (creates a negative).
/// </summary>
/// <remarks>
/// <para>Each pixel value is replaced with (max - value), where max is 255 for uint8 images
/// or 1.0 for normalized images. This creates a photographic negative effect.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Invert<T> : ImageAugmenterBase<T>
{
    public Invert(double probability = 0.5) : base(probability) { }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        result.SetFromPlanarTensor(Engine.ScalarMinusTensor(NumOps.FromDouble(maxVal), data.ToPlanarTensor()));

        return result;
    }
}
