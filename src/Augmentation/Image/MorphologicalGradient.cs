namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological gradient - the difference between dilation and erosion, highlighting edges.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MorphologicalGradient<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }
    public double BlendAlpha { get; }

    public MorphologicalGradient(int kernelSize = 3, double blendAlpha = 0.5,
        double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize; BlendAlpha = blendAlpha;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var dilated = new Dilate<T>(KernelSize, 1, probability: 1.0).Apply(data, context);
        var eroded = new Erode<T>(KernelSize, 1, probability: 1.0).Apply(data, context);

        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double d = NumOps.ToDouble(dilated.GetPixel(y, x, c));
                    double e = NumOps.ToDouble(eroded.GetPixel(y, x, c));
                    double gradient = d - e;
                    double original = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double blended = original * (1 - BlendAlpha) + gradient * BlendAlpha;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, blended))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["blend_alpha"] = BlendAlpha;
        return p;
    }
}
