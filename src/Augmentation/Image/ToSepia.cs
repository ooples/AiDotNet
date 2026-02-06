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
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double r = NumOps.ToDouble(data.GetPixel(y, x, 0));
                double g = NumOps.ToDouble(data.GetPixel(y, x, 1));
                double b = NumOps.ToDouble(data.GetPixel(y, x, 2));

                double newR = r * 0.393 + g * 0.769 + b * 0.189;
                double newG = r * 0.349 + g * 0.686 + b * 0.168;
                double newB = r * 0.272 + g * 0.534 + b * 0.131;

                result.SetPixel(y, x, 0, NumOps.FromDouble(Math.Min(maxVal, newR)));
                result.SetPixel(y, x, 1, NumOps.FromDouble(Math.Min(maxVal, newG)));
                result.SetPixel(y, x, 2, NumOps.FromDouble(Math.Min(maxVal, newB)));
            }

        return result;
    }
}
