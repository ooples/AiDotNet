namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies gamma correction to adjust image brightness non-linearly.
/// </summary>
/// <remarks>
/// <para>Gamma correction raises each pixel value to the power of (1/gamma). Gamma &gt; 1
/// brightens the image (especially dark areas), while gamma &lt; 1 darkens it.</para>
/// <para><b>For Beginners:</b> Unlike brightness which adds/multiplies uniformly, gamma
/// correction affects dark and bright areas differently, giving more natural-looking
/// brightness adjustments.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GammaCorrection<T> : ImageAugmenterBase<T>
{
    public double MinGamma { get; }
    public double MaxGamma { get; }

    public GammaCorrection(double minGamma = 0.8, double maxGamma = 1.2, double probability = 0.5)
        : base(probability)
    {
        if (minGamma <= 0) throw new ArgumentOutOfRangeException(nameof(minGamma));
        if (minGamma > maxGamma) throw new ArgumentException("minGamma must be <= maxGamma.");
        MinGamma = minGamma;
        MaxGamma = maxGamma;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double gamma = context.GetRandomDouble(MinGamma, MaxGamma);
        double invGamma = 1.0 / gamma;
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        var engine = Engine;
        var normalized = engine.TensorDivideScalar(data.ToPlanarTensor(), NumOps.FromDouble(maxVal));
        var corrected = engine.TensorMultiplyScalar(
            engine.TensorPower(normalized, NumOps.FromDouble(invGamma)), NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(corrected);

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_gamma"] = MinGamma; p["max_gamma"] = MaxGamma;
        return p;
    }
}
