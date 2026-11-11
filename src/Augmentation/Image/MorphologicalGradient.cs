namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological gradient - the difference between dilation and erosion, highlighting edges.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MorphologicalGradient<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }
    public double BlendAlpha { get; }

    public MorphologicalGradient(int kernelSize = 3, double blendAlpha = 0.5,
        double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize; BlendAlpha = blendAlpha;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var dilated = new Dilate<T>(KernelSize, 1, probability: 1.0).Apply(data, context);
        var eroded = new Erode<T>(KernelSize, 1, probability: 1.0).Apply(data, context);

        var result = data.Clone();

        var engine = Engine;
        var gradient = engine.TensorSubtract(dilated.ToPlanarTensor(), eroded.ToPlanarTensor());
        var blended = engine.TensorAdd(
            engine.TensorMultiplyScalar(data.ToPlanarTensor(), NumOps.FromDouble(1 - BlendAlpha)),
            engine.TensorMultiplyScalar(gradient, NumOps.FromDouble(BlendAlpha)));
        result.SetFromPlanarTensor(engine.TensorClamp(blended, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["blend_alpha"] = BlendAlpha;
        return p;
    }
}
