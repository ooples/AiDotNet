namespace AiDotNet.Augmentation.Image;

/// <summary>
/// DADA - Differentiable Automatic Data Augmentation (Li et al., 2020).
/// Applies augmentations with learned probabilities via Gumbel-Softmax relaxation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DADA<T> : ImageAugmenterBase<T>
{
    public double Temperature { get; }
    public int OperationCount { get; }

    public DADA(double temperature = 0.5, int operationCount = 2,
        double probability = 1.0) : base(probability)
    {
        Temperature = temperature; OperationCount = operationCount;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var ops = (AugmentOp[])Enum.GetValues(typeof(AugmentOp));

        // Simulate Gumbel-Softmax sampling for operation selection
        var logits = new double[ops.Length];
        for (int i = 0; i < ops.Length; i++)
        {
            double u = context.GetRandomDouble(1e-10, 1.0 - 1e-10);
            double gumbel = -Math.Log(-Math.Log(u));
            logits[i] = gumbel / Temperature;
        }

        // Softmax to get probabilities
        double maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > maxLogit) maxLogit = logits[i];

        double sumExp = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = Math.Exp(logits[i] - maxLogit);
            sumExp += logits[i];
        }

        var result = data;
        for (int n = 0; n < OperationCount; n++)
        {
            // Select operation with highest Gumbel-Softmax score
            int bestIdx = 0;
            double bestScore = logits[0] / sumExp;
            for (int i = 1; i < logits.Length; i++)
            {
                double score = logits[i] / sumExp;
                if (score > bestScore) { bestScore = score; bestIdx = i; }
            }

            double magnitude = context.GetRandomDouble(0, 1);
            result = ApplyOp(result, ops[bestIdx], magnitude, context);

            // Re-sample for next operation
            logits[bestIdx] = 0;
            sumExp -= bestScore * sumExp;
        }

        return result;
    }

    private static ImageTensor<T> ApplyOp(ImageTensor<T> data, AugmentOp op, double magnitude,
        AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

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
            default:
                return data.Clone();
        }
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["temperature"] = Temperature; p["operation_count"] = OperationCount;
        return p;
    }
}
