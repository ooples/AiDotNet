namespace AiDotNet.Augmentation.Image;

/// <summary>
/// RandAugment (Cubuk et al., 2019) - applies N random augmentations at magnitude M.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandAugment<T> : ImageAugmenterBase<T>
{
    public int N { get; }
    public double M { get; }
    public double MStd { get; }

    public RandAugment(int n = 2, double m = 9, double mStd = 0.5,
        double probability = 1.0) : base(probability)
    {
        N = n; M = m; MStd = mStd;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var ops = (AugmentOp[])Enum.GetValues(typeof(AugmentOp));
        var result = data;

        for (int i = 0; i < N; i++)
        {
            var op = ops[context.GetRandomInt(0, ops.Length)];
            double magnitude = M / 10.0;
            if (MStd > 0)
            {
                magnitude = (M + context.SampleGaussian(0, MStd)) / 10.0;
                magnitude = Math.Max(0, Math.Min(1, magnitude));
            }

            result = ApplyOp(result, op, magnitude, context);
        }

        return result;
    }

    private static ImageTensor<T> ApplyOp(ImageTensor<T> data, AugmentOp op, double magnitude,
        AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        switch (op)
        {
            case AugmentOp.ShearX:
            case AugmentOp.ShearY:
                double shear = magnitude * 0.3;
                if (context.GetRandomBool()) shear = -shear;
                double shearDeg = shear * 180 / Math.PI;
                return new Affine<T>(
                    shearRange: (shearDeg, shearDeg),
                    probability: 1.0).Apply(data, context);

            case AugmentOp.TranslateX:
            case AugmentOp.TranslateY:
                double trans = magnitude * 0.45;
                if (context.GetRandomBool()) trans = -trans;
                return new Affine<T>(
                    translationRange: (Math.Abs(trans), Math.Abs(trans)),
                    probability: 1.0).Apply(data, context);

            case AugmentOp.Rotate:
                double angle = magnitude * 30;
                if (context.GetRandomBool()) angle = -angle;
                return new Rotation<T>(angle, angle, probability: 1.0).Apply(data, context);

            case AugmentOp.Brightness:
                double bf = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) bf = 1.0 / bf;
                return new Brightness<T>(bf, bf, probability: 1.0).Apply(data, context);

            case AugmentOp.Contrast:
                double cf = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) cf = 1.0 / cf;
                return new Contrast<T>(cf, cf, probability: 1.0).Apply(data, context);

            case AugmentOp.Sharpness:
                return new Sharpen<T>(magnitude, magnitude, 1.0, 1.0, probability: 1.0).Apply(data, context);

            case AugmentOp.Posterize:
                int bits = Math.Max(1, 8 - (int)(magnitude * 4));
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
                double sf = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) sf = 1.0 / sf;
                return new Saturation<T>(sf, sf, probability: 1.0).Apply(data, context);

            default:
                return data.Clone();
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["n"] = N; p["m"] = M; p["m_std"] = MStd;
        return p;
    }
}
