namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies random gamma correction with a random gamma value within a range.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomGamma<T> : ImageAugmenterBase<T>
{
    public double MinGamma { get; }
    public double MaxGamma { get; }

    public RandomGamma(double minGamma = 0.5, double maxGamma = 2.0,
        double probability = 0.5) : base(probability)
    {
        if (minGamma <= 0) throw new ArgumentOutOfRangeException(nameof(minGamma));
        MinGamma = minGamma; MaxGamma = maxGamma;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double gamma = context.GetRandomDouble(MinGamma, MaxGamma);
        var correction = new GammaCorrection<T>(gamma, gamma, probability: 1.0);
        return correction.Apply(data, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_gamma"] = MinGamma; p["max_gamma"] = MaxGamma;
        return p;
    }
}
