namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a sepia tone filter to the image.
/// </summary>
/// <remarks>
/// <para>Sepia creates a warm brownish tone reminiscent of vintage photographs. The transform
/// uses the standard sepia matrix commonly used in image processing.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ToSepia<T> : ImageAugmenterBase<T>
{
    public ToSepia(double probability = 0.5) : base(probability) { }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Standard sepia matrix
        var engine = Engine;
        var image = data.ToPlanarTensor();
        T max = NumOps.FromDouble(maxVal);
        var newR = engine.TensorClampMax(ImageTensor<T>.MixChannels(image, 0.393, 0.769, 0.189), max);
        var newG = engine.TensorClampMax(ImageTensor<T>.MixChannels(image, 0.349, 0.686, 0.168), max);
        var newB = engine.TensorClampMax(ImageTensor<T>.MixChannels(image, 0.272, 0.534, 0.131), max);
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image, newR, newG, newB));

        return result;
    }
}
