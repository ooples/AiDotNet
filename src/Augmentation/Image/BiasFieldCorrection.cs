namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates or corrects MRI bias field (intensity non-uniformity).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BiasFieldCorrection<T> : ImageAugmenterBase<T>
{
    public double MinCoeff { get; }
    public double MaxCoeff { get; }
    public int Order { get; }

    public BiasFieldCorrection(double minCoeff = 0.0, double maxCoeff = 0.3,
        int order = 3, double probability = 0.5) : base(probability)
    {
        MinCoeff = minCoeff; MaxCoeff = maxCoeff; Order = order;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Generate polynomial bias field
        var coeffs = new double[Order + 1, Order + 1];
        for (int i = 0; i <= Order; i++)
            for (int j = 0; j <= Order; j++)
                if (i + j <= Order && i + j > 0)
                    coeffs[i, j] = context.GetRandomDouble(MinCoeff, MaxCoeff) *
                                   (context.GetRandomBool() ? 1 : -1);

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double ny = 2.0 * y / data.Height - 1; // [-1, 1]
                double nx = 2.0 * x / data.Width - 1;

                double bias = 1.0;
                for (int i = 0; i <= Order; i++)
                    for (int j = 0; j <= Order; j++)
                        if (i + j <= Order && i + j > 0)
                            bias += coeffs[i, j] * Math.Pow(ny, i) * Math.Pow(nx, j);

                bias = Math.Max(0.5, Math.Min(1.5, bias)); // Clamp bias field

                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c)) * bias;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_coeff"] = MinCoeff; p["max_coeff"] = MaxCoeff; p["order"] = Order;
        return p;
    }
}
