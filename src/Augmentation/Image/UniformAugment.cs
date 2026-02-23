namespace AiDotNet.Augmentation.Image;

/// <summary>
/// UniformAugment - applies a random number of randomly selected augmentations
/// with uniformly sampled magnitudes.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class UniformAugment<T> : ImageAugmenterBase<T>
{
    public int MaxOps { get; }

    public UniformAugment(int maxOps = 3, double probability = 1.0) : base(probability)
    {
        MaxOps = maxOps;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var ops = (AugmentOp[])Enum.GetValues(typeof(AugmentOp));
        int numOps = context.GetRandomInt(0, MaxOps + 1);
        var result = data;

        for (int i = 0; i < numOps; i++)
        {
            var op = ops[context.GetRandomInt(0, ops.Length)];
            double magnitude = context.GetRandomDouble(0, 1);
            result = ApplyOp(result, op, magnitude, context);
        }

        return result;
    }

    private static ImageTensor<T> ApplyOp(ImageTensor<T> data, AugmentOp op, double magnitude,
        AugmentationContext<T> context)
    {
        switch (op)
        {
            case AugmentOp.Rotate:
                double angle = magnitude * 30;
                if (context.GetRandomBool()) angle = -angle;
                return new Rotation<T>(angle, angle, probability: 1.0).Apply(data, context);
            case AugmentOp.Brightness:
                double bf = 1.0 + magnitude * 0.5;
                if (context.GetRandomBool()) bf = 1.0 / bf;
                return new Brightness<T>(bf, bf, probability: 1.0).Apply(data, context);
            case AugmentOp.Contrast:
                double cf = 1.0 + magnitude * 0.5;
                if (context.GetRandomBool()) cf = 1.0 / cf;
                return new Contrast<T>(cf, cf, probability: 1.0).Apply(data, context);
            case AugmentOp.Sharpness:
                return new Sharpen<T>(magnitude, magnitude, 1.0, 1.0, probability: 1.0).Apply(data, context);
            case AugmentOp.Posterize:
                int bits = Math.Max(1, 8 - (int)(magnitude * 4));
                return new Posterize<T>(bits, bits, probability: 1.0).Apply(data, context);
            case AugmentOp.Solarize:
                double maxVal = data.IsNormalized ? 1.0 : 255.0;
                return new Solarize<T>(maxVal * (1 - magnitude), probability: 1.0).Apply(data, context);
            case AugmentOp.AutoContrast:
                return new AutoContrast<T>(probability: 1.0).Apply(data, context);
            case AugmentOp.Equalize:
                return new Equalize<T>(probability: 1.0).Apply(data, context);
            case AugmentOp.Invert:
                return new Invert<T>(probability: 1.0).Apply(data, context);
            default:
                return data.Clone();
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["max_ops"] = MaxOps;
        return p;
    }
}
