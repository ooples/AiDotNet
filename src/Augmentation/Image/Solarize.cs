namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Inverts all pixel values above a threshold, creating a solarization effect.
/// </summary>
/// <remarks>
/// <para>Solarization was originally a photographic darkroom effect. Pixels above the threshold
/// are inverted (value = max - value), while pixels below remain unchanged.</para>
/// <para><b>For Beginners:</b> Bright parts of the image get inverted (made dark) while dark
/// parts stay the same, creating a surreal effect. Used as augmentation in AutoAugment.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Solarize<T> : ImageAugmenterBase<T>
{
    public double Threshold { get; }

    public Solarize(double threshold = 0.5, double probability = 0.5) : base(probability)
    {
        Threshold = threshold;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double thresh = Threshold * maxVal;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    if (val > thresh)
                        result.SetPixel(y, x, c, NumOps.FromDouble(maxVal - val));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["threshold"] = Threshold; return p;
    }
}
