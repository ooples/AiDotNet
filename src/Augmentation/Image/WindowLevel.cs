namespace AiDotNet.Augmentation.Image;

/// <summary>
/// CT/MRI window/level adjustment for medical image augmentation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WindowLevel<T> : ImageAugmenterBase<T>
{
    public double WindowWidth { get; }
    public double WindowCenter { get; }
    public double WindowWidthVar { get; }
    public double WindowCenterVar { get; }

    public WindowLevel(double windowWidth = 0.5, double windowCenter = 0.5,
        double windowWidthVar = 0.1, double windowCenterVar = 0.1,
        double probability = 0.5) : base(probability)
    {
        WindowWidth = windowWidth; WindowCenter = windowCenter;
        WindowWidthVar = windowWidthVar; WindowCenterVar = windowCenterVar;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        double width = WindowWidth + context.GetRandomDouble(-WindowWidthVar, WindowWidthVar);
        double center = WindowCenter + context.GetRandomDouble(-WindowCenterVar, WindowCenterVar);
        width = Math.Max(0.01, width);

        double lower = (center - width / 2) * maxVal;
        double upper = (center + width / 2) * maxVal;
        double range = upper - lower;

        var engine = Engine;
        var image = data.ToPlanarTensor();
        Tensor<T> windowed;
        if (range > 1e-10)
        {
            windowed = engine.TensorMultiplyScalar(
                engine.TensorDivideScalar(engine.TensorSubtractScalar(image, NumOps.FromDouble(lower)), NumOps.FromDouble(range)),
                NumOps.FromDouble(maxVal));
        }
        else
        {
            engine.TensorFill(image, NumOps.FromDouble(maxVal / 2));
            windowed = image;
        }

        result.SetFromPlanarTensor(engine.TensorClamp(windowed, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["window_width"] = WindowWidth; p["window_center"] = WindowCenter;
        p["window_width_var"] = WindowWidthVar; p["window_center_var"] = WindowCenterVar;
        return p;
    }
}
