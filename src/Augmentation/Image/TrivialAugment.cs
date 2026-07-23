namespace AiDotNet.Augmentation.Image;

/// <summary>
/// TrivialAugmentWide (Muller &amp; Hutter, 2021) - applies a single random augmentation
/// with uniformly sampled magnitude.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TrivialAugment<T> : ImageAugmenterBase<T>
{
    public int NumBins { get; }

    public TrivialAugment(int numBins = 31, double probability = 1.0) : base(probability)
    {
        NumBins = numBins;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var ops = (AugmentOp[])Enum.GetValues(typeof(AugmentOp));
        var op = ops[context.GetRandomInt(0, ops.Length)];
        double magnitude = (double)context.GetRandomInt(0, NumBins) / NumBins;

        return ApplyOp(data, op, magnitude, context);
    }

    private static ImageTensor<T> ApplyOp(ImageTensor<T> data, AugmentOp op, double magnitude,
        AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        switch (op)
        {
            case AugmentOp.ShearX:
            case AugmentOp.ShearY:
                double shear = magnitude * 0.99;
                if (context.GetRandomBool()) shear = -shear;
                double shearDeg = shear * 180 / Math.PI;
                return new Affine<T>(
                    shearRange: (shearDeg, shearDeg),
                    probability: 1.0).Apply(data, context);

            case AugmentOp.TranslateX:
            case AugmentOp.TranslateY:
                double trans = magnitude * 0.33;
                if (context.GetRandomBool()) trans = -trans;
                return new Affine<T>(
                    translationRange: (Math.Abs(trans), Math.Abs(trans)),
                    probability: 1.0).Apply(data, context);

            case AugmentOp.Rotate:
                double angle = magnitude * 135;
                if (context.GetRandomBool()) angle = -angle;
                return new Rotation<T>(angle, angle, probability: 1.0).Apply(data, context);

            case AugmentOp.Brightness:
                double bf = 1.0 + magnitude * 0.99;
                if (context.GetRandomBool()) bf = 1.0 / bf;
                return new Brightness<T>(bf, bf, probability: 1.0).Apply(data, context);

            case AugmentOp.Contrast:
                double cf = 1.0 + magnitude * 0.99;
                if (context.GetRandomBool()) cf = 1.0 / cf;
                return new Contrast<T>(cf, cf, probability: 1.0).Apply(data, context);

            case AugmentOp.Sharpness:
                return new Sharpen<T>(magnitude, magnitude, 1.0, 1.0, probability: 1.0).Apply(data, context);

            case AugmentOp.Posterize:
                int bits = Math.Max(1, 8 - (int)(magnitude * 7));
                return new Posterize<T>(bits, bits, probability: 1.0).Apply(data, context);

            case AugmentOp.Solarize:
                return new Solarize<T>(maxVal * (1 - magnitude), probability: 1.0).Apply(data, context);

            case AugmentOp.AutoContrast:
                return new AutoContrast<T>(probability: 1.0).Apply(data, context);

            case AugmentOp.Equalize:
                return new Equalize<T>(probability: 1.0).Apply(data, context);

            case AugmentOp.Invert:
                return new Invert<T>(probability: 1.0).Apply(data, context);

            case AugmentOp.Color:
                double sf = 1.0 + magnitude * 0.99;
                if (context.GetRandomBool()) sf = 1.0 / sf;
                return new Saturation<T>(sf, sf, probability: 1.0).Apply(data, context);

            default:
                return data.Clone();
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["num_bins"] = NumBins;
        return p;
    }
}
