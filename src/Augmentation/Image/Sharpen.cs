namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a sharpening filter using a 3x3 kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Sharpen<T> : ImageAugmenterBase<T>
{
    public double MinAlpha { get; }
    public double MaxAlpha { get; }
    public double MinLightness { get; }
    public double MaxLightness { get; }

    public Sharpen(double minAlpha = 0.2, double maxAlpha = 0.5,
        double minLightness = 0.5, double maxLightness = 1.0,
        double probability = 0.5) : base(probability)
    {
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
        MinLightness = minLightness; MaxLightness = maxLightness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);
        double lightness = context.GetRandomDouble(MinLightness, MaxLightness);

        // Sharpening kernel blended with identity
        // identity = [0,0,0; 0,1,0; 0,0,0]
        // sharpen  = [-1,-1,-1; -1,8+L,-1; -1,-1,-1] / (L)
        // Effective kernel = (1-alpha)*identity + alpha*sharpen

        // Laplacian: (center * (8 + L) - sum of the 8 neighbours) / (8 + L), edges replicated
        var engine = Engine;
        var image = data.ToPlanarTensor();
        double[,] laplacianKernel =
        {
            { -1, -1, -1 },
            { -1, 8 + lightness, -1 },
            { -1, -1, -1 }
        };
        var laplacian = engine.TensorDivideScalar(
            ImageTensor<T>.Correlate(image, ImageTensor<T>.KernelTensor(laplacianKernel), mirror: false),
            NumOps.FromDouble(8 + lightness));

        var sharpened = engine.TensorAdd(
            engine.TensorMultiplyScalar(image, NumOps.FromDouble(1 - alpha)),
            engine.TensorMultiplyScalar(laplacian, NumOps.FromDouble(alpha)));
        result.SetFromPlanarTensor(engine.TensorClamp(sharpened, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        p["min_lightness"] = MinLightness; p["max_lightness"] = MaxLightness;
        return p;
    }
}
