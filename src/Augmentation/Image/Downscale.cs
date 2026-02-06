namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Downscales then upscales the image to simulate resolution degradation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Downscale<T> : ImageAugmenterBase<T>
{
    public double MinScale { get; }
    public double MaxScale { get; }
    public InterpolationMode DownInterpolation { get; }
    public InterpolationMode UpInterpolation { get; }

    public Downscale(double minScale = 0.25, double maxScale = 0.5,
        InterpolationMode downInterpolation = InterpolationMode.Nearest,
        InterpolationMode upInterpolation = InterpolationMode.Nearest,
        double probability = 0.5) : base(probability)
    {
        if (minScale <= 0 || minScale > 1) throw new ArgumentOutOfRangeException(nameof(minScale));
        if (maxScale < minScale || maxScale > 1) throw new ArgumentOutOfRangeException(nameof(maxScale));
        MinScale = minScale; MaxScale = maxScale;
        DownInterpolation = downInterpolation; UpInterpolation = upInterpolation;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double scale = context.GetRandomDouble(MinScale, MaxScale);
        int smallH = Math.Max(1, (int)(data.Height * scale));
        int smallW = Math.Max(1, (int)(data.Width * scale));

        var downResize = new Resize<T>(smallH, smallW, DownInterpolation, probability: 1.0);
        var small = downResize.Apply(data, context);

        var upResize = new Resize<T>(data.Height, data.Width, UpInterpolation, probability: 1.0);
        return upResize.Apply(small, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_scale"] = MinScale; p["max_scale"] = MaxScale;
        return p;
    }
}
