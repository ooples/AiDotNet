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

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double thresh = Threshold * maxVal;

        // Pixels above the threshold are inverted; the rest are kept
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var above = engine.TensorGreaterThan(image, NumOps.FromDouble(thresh));
        var inverted = engine.ScalarMinusTensor(NumOps.FromDouble(maxVal), image);
        result.SetFromPlanarTensor(engine.TensorWhere(above, inverted, image));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["threshold"] = Threshold; return p;
    }
}
