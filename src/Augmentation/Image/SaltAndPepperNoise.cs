namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds salt-and-pepper (impulse) noise by randomly setting pixels to black or white.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SaltAndPepperNoise<T> : ImageAugmenterBase<T>
{
    public double Amount { get; }
    public double SaltVsPepper { get; }

    public SaltAndPepperNoise(double amount = 0.02, double saltVsPepper = 0.5, double probability = 0.5)
        : base(probability)
    {
        if (amount < 0 || amount > 1) throw new ArgumentOutOfRangeException(nameof(amount));
        Amount = amount; SaltVsPepper = saltVsPepper;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                if (context.GetRandomDouble(0, 1) < Amount)
                {
                    double val = context.GetRandomDouble(0, 1) < SaltVsPepper ? maxVal : 0;
                    for (int c = 0; c < data.Channels; c++)
                        result.SetPixel(y, x, c, NumOps.FromDouble(val));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["amount"] = Amount; p["salt_vs_pepper"] = SaltVsPepper; return p;
    }
}
