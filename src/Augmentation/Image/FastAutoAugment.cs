namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Fast AutoAugment (Lim et al., 2019) - efficient augmentation policy search
/// using density matching.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FastAutoAugment<T> : ImageAugmenterBase<T>
{
    public int NumPolicies { get; }
    public int NumOpsPerPolicy { get; }

    public FastAutoAugment(int numPolicies = 5, int numOpsPerPolicy = 2,
        double probability = 1.0) : base(probability)
    {
        NumPolicies = numPolicies; NumOpsPerPolicy = numOpsPerPolicy;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Select a random policy
        var ops = (AugmentOp[])Enum.GetValues(typeof(AugmentOp));
        var result = data;

        for (int i = 0; i < NumOpsPerPolicy; i++)
        {
            var op = ops[context.GetRandomInt(0, ops.Length)];
            double magnitude = context.GetRandomDouble(0, 1);
            double prob = context.GetRandomDouble(0.2, 1.0);

            if (context.GetRandomDouble(0, 1) < prob)
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
            case AugmentOp.AutoContrast:
                return new AutoContrast<T>(probability: 1.0).Apply(data, context);
            case AugmentOp.Equalize:
                return new Equalize<T>(probability: 1.0).Apply(data, context);
            case AugmentOp.Posterize:
                int bits = Math.Max(1, 8 - (int)(magnitude * 4));
                return new Posterize<T>(bits, bits, probability: 1.0).Apply(data, context);
            case AugmentOp.Solarize:
                double maxVal = data.IsNormalized ? 1.0 : 255.0;
                return new Solarize<T>(maxVal * (1 - magnitude), probability: 1.0).Apply(data, context);
            default:
                return data.Clone();
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["num_policies"] = NumPolicies; p["num_ops_per_policy"] = NumOpsPerPolicy;
        return p;
    }
}
