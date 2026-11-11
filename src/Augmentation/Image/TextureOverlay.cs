namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Overlays a procedurally generated random texture on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TextureOverlay<T> : ImageAugmenterBase<T>
{
    public double MinAlpha { get; }
    public double MaxAlpha { get; }
    public double MinScale { get; }
    public double MaxScale { get; }

    public TextureOverlay(double minAlpha = 0.05, double maxAlpha = 0.3,
        double minScale = 0.02, double maxScale = 0.1,
        double probability = 0.5) : base(probability)
    {
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
        MinScale = minScale; MaxScale = maxScale;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);
        double scale = context.GetRandomDouble(MinScale, MaxScale);
        int gridSize = Math.Max(2, (int)(Math.Min(data.Height, data.Width) * scale));

        // Generate procedural texture
        var texture = ImageTensor<T>.GridNoise(context, data.Height, data.Width, gridSize, 0, 1, smoothstep: false);

        var engine = Engine;
        var tex = engine.TensorMultiplyScalar(texture, NumOps.FromDouble(maxVal));
        var tex3 = engine.Reshape(tex, [1, data.Height, data.Width]);
        var blended = engine.TensorAdd(
            engine.TensorMultiplyScalar(data.ToPlanarTensor(), NumOps.FromDouble(1 - alpha)),
            engine.TensorMultiplyScalar(tex3, NumOps.FromDouble(alpha)));
        result.SetFromPlanarTensor(engine.TensorClamp(blended, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        p["min_scale"] = MinScale; p["max_scale"] = MaxScale;
        return p;
    }
}
