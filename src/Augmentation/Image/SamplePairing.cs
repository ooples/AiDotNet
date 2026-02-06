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

        for (int y = 0; y < image1.Height; y++)
            for (int x = 0; x < image1.Width; x++)
                for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                {
                    double v1 = NumOps.ToDouble(image1.GetPixel(y, x, c));
                    double v2 = NumOps.ToDouble(image2.GetPixel(y, x, c));
                    double paired = v1 * (1 - weight) + v2 * weight;
                    result.SetPixel(y, x, c, NumOps.FromDouble(paired));
                }

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Mixup);
            RaiseLabelMixing(args);
        }

        return result;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_weight"] = MinWeight; p["max_weight"] = MaxWeight;
        return p;
    }
}
