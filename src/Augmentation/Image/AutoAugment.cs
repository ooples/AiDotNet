namespace AiDotNet.Augmentation.Image;

/// <summary>
/// AutoAugment (Cubuk et al., 2018) - applies learned augmentation policies.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AutoAugment<T> : ImageAugmenterBase<T>
{
    public AutoAugmentPolicy Policy { get; }

    public AutoAugment(AutoAugmentPolicy policy = AutoAugmentPolicy.ImageNet,
        double probability = 1.0) : base(probability)
    {
        Policy = policy;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var policies = GetPolicies();
        int policyIdx = context.GetRandomInt(0, policies.Length);
        var (op1, mag1, prob1, op2, mag2, prob2) = policies[policyIdx];

        var result = data;
        if (context.GetRandomDouble(0, 1) < prob1)
            result = ApplyOperation(result, op1, mag1, context);
        if (context.GetRandomDouble(0, 1) < prob2)
            result = ApplyOperation(result, op2, mag2, context);

        return result;
    }

    private ImageTensor<T> ApplyOperation(ImageTensor<T> data, AugmentOp op, double magnitude,
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
                var affine = new Affine<T>(
                    shearRange: op == AugmentOp.ShearX ? (shearDeg, shearDeg) : (shearDeg, shearDeg),
                    probability: 1.0);
                return affine.Apply(data, context);

            case AugmentOp.TranslateX:
            case AugmentOp.TranslateY:
                double trans = magnitude * 0.45;
                if (context.GetRandomBool()) trans = -trans;
                var translate = new Affine<T>(
                    translationRange: (Math.Abs(trans), Math.Abs(trans)),
                    probability: 1.0);
                return translate.Apply(data, context);

            case AugmentOp.Rotate:
                double angle = magnitude * 30;
                if (context.GetRandomBool()) angle = -angle;
                var rotate = new Rotation<T>(angle, angle, probability: 1.0);
                return rotate.Apply(data, context);

            case AugmentOp.Brightness:
                double bFactor = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) bFactor = 1.0 / bFactor;
                var bright = new Brightness<T>(bFactor, bFactor, probability: 1.0);
                return bright.Apply(data, context);

            case AugmentOp.Contrast:
                double cFactor = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) cFactor = 1.0 / cFactor;
                var contrast = new Contrast<T>(cFactor, cFactor, probability: 1.0);
                return contrast.Apply(data, context);

            case AugmentOp.Sharpness:
                double alpha = magnitude * 0.9;
                var sharpen = new Sharpen<T>(alpha, alpha, 1.0, 1.0, probability: 1.0);
                return sharpen.Apply(data, context);

            case AugmentOp.Posterize:
                int bits = Math.Max(1, 8 - (int)(magnitude * 4));
                var posterize = new Posterize<T>(bits, bits, probability: 1.0);
                return posterize.Apply(data, context);

            case AugmentOp.Solarize:
                double threshold = maxVal * (1 - magnitude);
                var solarize = new Solarize<T>(threshold, probability: 1.0);
                return solarize.Apply(data, context);

            case AugmentOp.AutoContrast:
                var autoContrast = new AutoContrast<T>(probability: 1.0);
                return autoContrast.Apply(data, context);

            case AugmentOp.Equalize:
                var equalize = new Equalize<T>(probability: 1.0);
                return equalize.Apply(data, context);

            case AugmentOp.Invert:
                var invert = new Invert<T>(probability: 1.0);
                return invert.Apply(data, context);

            case AugmentOp.Color:
                double sFactor = 1.0 + magnitude * 0.9;
                if (context.GetRandomBool()) sFactor = 1.0 / sFactor;
                var saturation = new Saturation<T>(sFactor, sFactor, probability: 1.0);
                return saturation.Apply(data, context);

            default:
                return data.Clone();
        }
    }

    private (AugmentOp, double, double, AugmentOp, double, double)[] GetPolicies()
    {
        return Policy switch
        {
            AutoAugmentPolicy.ImageNet => new[]
            {
                (AugmentOp.Posterize, 0.8, 0.4, AugmentOp.Rotate, 0.9, 0.6),
                (AugmentOp.Solarize, 0.5, 0.6, AugmentOp.AutoContrast, 0.5, 0.6),
                (AugmentOp.Equalize, 0.8, 0.8, AugmentOp.Equalize, 0.6, 0.6),
                (AugmentOp.Posterize, 0.6, 0.6, AugmentOp.Posterize, 0.6, 0.6),
                (AugmentOp.Equalize, 0.4, 0.8, AugmentOp.Solarize, 0.4, 0.4),
                (AugmentOp.Equalize, 0.4, 0.2, AugmentOp.Rotate, 0.8, 0.8),
                (AugmentOp.Solarize, 0.6, 0.6, AugmentOp.Equalize, 0.6, 0.6),
                (AugmentOp.Posterize, 0.8, 0.6, AugmentOp.Equalize, 1.0, 0.2),
                (AugmentOp.Rotate, 0.2, 0.8, AugmentOp.Color, 0.4, 0.0),
                (AugmentOp.Equalize, 0.6, 0.8, AugmentOp.Equalize, 0.0, 0.8),
                (AugmentOp.Invert, 0.0, 0.6, AugmentOp.Equalize, 1.0, 0.6),
                (AugmentOp.Color, 0.4, 0.6, AugmentOp.Equalize, 0.8, 1.0),
                (AugmentOp.Rotate, 0.8, 0.8, AugmentOp.Color, 0.2, 0.8),
                (AugmentOp.Color, 0.8, 0.8, AugmentOp.Solarize, 0.8, 0.8),
                (AugmentOp.Sharpness, 0.3, 0.6, AugmentOp.Brightness, 0.6, 0.6),
            },

            AutoAugmentPolicy.CIFAR10 => new[]
            {
                (AugmentOp.Invert, 0.1, 0.7, AugmentOp.Contrast, 0.2, 0.6),
                (AugmentOp.Rotate, 0.7, 0.2, AugmentOp.TranslateX, 0.3, 0.9),
                (AugmentOp.Sharpness, 0.8, 0.1, AugmentOp.Sharpness, 0.9, 0.3),
                (AugmentOp.ShearY, 0.5, 0.7, AugmentOp.TranslateY, 0.7, 0.9),
                (AugmentOp.AutoContrast, 0.5, 0.8, AugmentOp.Equalize, 0.9, 0.2),
                (AugmentOp.ShearY, 0.2, 0.7, AugmentOp.Posterize, 0.3, 0.7),
                (AugmentOp.Color, 0.4, 0.3, AugmentOp.Brightness, 0.6, 0.7),
                (AugmentOp.Sharpness, 0.3, 0.9, AugmentOp.Brightness, 0.7, 0.9),
                (AugmentOp.Equalize, 0.6, 0.5, AugmentOp.Equalize, 0.3, 0.3),
                (AugmentOp.Contrast, 0.6, 0.7, AugmentOp.Sharpness, 0.6, 0.3),
            },

            AutoAugmentPolicy.SVHN => new[]
            {
                (AugmentOp.ShearX, 0.9, 0.9, AugmentOp.Equalize, 0.6, 0.5),
                (AugmentOp.ShearY, 0.9, 0.8, AugmentOp.Invert, 0.4, 0.5),
                (AugmentOp.Equalize, 0.6, 0.3, AugmentOp.Rotate, 0.9, 0.3),
                (AugmentOp.ShearX, 0.9, 0.1, AugmentOp.Equalize, 0.3, 0.6),
                (AugmentOp.Invert, 0.6, 0.5, AugmentOp.Equalize, 0.6, 0.3),
                (AugmentOp.Equalize, 0.9, 0.3, AugmentOp.Rotate, 0.3, 0.8),
            },

            _ => new[]
            {
                (AugmentOp.Equalize, 0.5, 0.5, AugmentOp.AutoContrast, 0.5, 0.5),
            }
        };
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["policy"] = Policy.ToString();
        return p;
    }
}

/// <summary>AutoAugment policy preset.</summary>
public enum AutoAugmentPolicy
{
    /// <summary>ImageNet policy.</summary>
    ImageNet,
    /// <summary>CIFAR-10 policy.</summary>
    CIFAR10,
    /// <summary>SVHN policy.</summary>
    SVHN,
    /// <summary>Custom policy.</summary>
    Custom
}

/// <summary>Augmentation operation type for AutoAugment policies.</summary>
public enum AugmentOp
{
    ShearX, ShearY, TranslateX, TranslateY, Rotate,
    Brightness, Contrast, Sharpness, Posterize, Solarize,
    AutoContrast, Equalize, Invert, Color
}
