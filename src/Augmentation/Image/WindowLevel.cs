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

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double windowed = range > 1e-10 ? (val - lower) / range * maxVal : maxVal / 2;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, windowed))));
                }

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
