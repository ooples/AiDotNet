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

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Draw, per pixel in y, x order, whether it is hit and (only when hit) salt or pepper
        int h = data.Height, w = data.Width;
        var hit = new int[h * w];
        var salt = new int[h * w];
        for (int i = 0; i < hit.Length; i++)
        {
            if (context.GetRandomDouble(0, 1) < Amount)
            {
                hit[i] = 1;
                salt[i] = context.GetRandomDouble(0, 1) < SaltVsPepper ? 1 : 0;
            }
        }

        // Hit pixels take the salt/pepper value in every channel
        var engine = Engine;
        var image = data.ToPlanarTensor();
        int[] shape = [data.Channels, h, w];
        var hitMask = engine.TensorBroadcastTo(new Tensor<int>(hit, [1, h, w]).Cast<T>(), shape);
        var saltMask = engine.TensorBroadcastTo(new Tensor<int>(salt, [1, h, w]).Cast<T>(), shape);
        var values = engine.TensorWhere(saltMask, ImageTensor<T>.Constant(image, maxVal), ImageTensor<T>.Constant(image, 0.0));
        result.SetFromPlanarTensor(engine.TensorWhere(hitMask, values, image));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["amount"] = Amount; p["salt_vs_pepper"] = SaltVsPepper; return p;
    }
}
