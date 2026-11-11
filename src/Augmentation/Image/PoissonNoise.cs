namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Adds Poisson (shot) noise that scales with pixel intensity, simulating photon counting noise.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PoissonNoise<T> : ImageAugmenterBase<T>
{
    public double Scale { get; }

    public PoissonNoise(double scale = 1.0, double probability = 0.5) : base(probability)
    {
        Scale = scale;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var result = data.Clone();

        // Poisson noise approximated by Gaussian with variance = value:
        // noise = N(0, sqrt(max(val / maxVal, 1e-10)) * Scale * 0.1 * maxVal), drawn in y, x, c order
        int h = data.Height, w = data.Width, channels = data.Channels;
        var standardNormal = AugmentationRandom<T>.Gaussian(context, [h, w, channels], 0, 1);

        var engine = Engine;
        var image = data.ToPlanarTensor();
        var normalizedVal = engine.TensorDivideScalar(image, NumOps.FromDouble(maxVal));
        var stdDev = engine.TensorMultiplyScalar(
            engine.TensorMultiplyScalar(
                engine.TensorSqrt(engine.TensorMax(normalizedVal, NumOps.FromDouble(1e-10))), NumOps.FromDouble(Scale)),
            NumOps.FromDouble(0.1));
        var noise = engine.TensorMultiply(
            engine.TensorMultiplyScalar(stdDev, NumOps.FromDouble(maxVal)),
            ImageTensor<T>.PlanarFromInterleaved(standardNormal, h, w, channels));
        var noisy = engine.TensorAdd(image, noise);
        result.SetFromPlanarTensor(engine.TensorClamp(noisy, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["scale"] = Scale; return p;
    }
}
