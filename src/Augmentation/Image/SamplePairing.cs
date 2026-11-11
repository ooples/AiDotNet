using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Sample Pairing augmentation (Inoue, 2018) - averages two images together.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SamplePairing<T> : ImageMixingAugmenterBase<T>
{
    public double MinWeight { get; }
    public double MaxWeight { get; }

    public SamplePairing(double minWeight = 0.0, double maxWeight = 0.5,
        double probability = 0.5) : base(probability)
    {
        if (minWeight < 0 || minWeight > 1) throw new ArgumentOutOfRangeException(nameof(minWeight), "MinWeight must be in [0, 1].");
        if (maxWeight < minWeight || maxWeight > 1) throw new ArgumentOutOfRangeException(nameof(maxWeight), "MaxWeight must be >= MinWeight and <= 1.");
        MinWeight = minWeight; MaxWeight = maxWeight;
    }

    /// <summary>
    /// Pairs two images by weighted averaging.
    /// </summary>
    public ImageTensor<T> ApplyPairing(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
        {
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);
        }

        var result = image1.Clone();
        double weight = context.GetRandomDouble(MinWeight, MaxWeight);
        double lambda = 1 - weight;
        LastMixingLambda = NumOps.FromDouble(lambda);

        // result = image1 * (1 - weight) + image2 * weight for the channels both images have
        var engine = Engine;
        int channels = Math.Min(image1.Channels, image2.Channels);
        var planar1 = image1.ToPlanarTensor();
        var paired = engine.TensorAdd(
            engine.TensorMultiplyScalar(ImageTensor<T>.LeadingChannels(planar1, channels), NumOps.FromDouble(1 - weight)),
            engine.TensorMultiplyScalar(ImageTensor<T>.LeadingChannels(image2.ToPlanarTensor(), channels), NumOps.FromDouble(weight)));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingBlock(planar1, paired));

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Mixup);
            RaiseLabelMixing(args);
        }

        return result;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_weight"] = MinWeight; p["max_weight"] = MaxWeight;
        return p;
    }
}
