namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies radial/zoom blur emanating from the image center.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ZoomBlur<T> : ImageAugmenterBase<T>
{
    public double MinZoomFactor { get; }
    public double MaxZoomFactor { get; }

    public ZoomBlur(double minZoomFactor = 0.01, double maxZoomFactor = 0.05, double probability = 0.5)
        : base(probability)
    {
        MinZoomFactor = minZoomFactor; MaxZoomFactor = maxZoomFactor;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        double factor = context.GetRandomDouble(MinZoomFactor, MaxZoomFactor);
        int numSteps = Math.Max(3, (int)(factor * 100));
        double cx = data.Width / 2.0, cy = data.Height / 2.0;
        var result = data.Clone();

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double sum = 0;
                    for (int s = 0; s < numSteps; s++)
                    {
                        double t = s * factor / numSteps;
                        double sx = x + (cx - x) * t;
                        double sy = y + (cy - y) * t;
                        int ix = Math.Max(0, Math.Min(data.Width - 1, (int)Math.Round(sx)));
                        int iy = Math.Max(0, Math.Min(data.Height - 1, (int)Math.Round(sy)));
                        sum += NumOps.ToDouble(data.GetPixel(iy, ix, c));
                    }
                    result.SetPixel(y, x, c, NumOps.FromDouble(sum / numSteps));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_zoom_factor"] = MinZoomFactor; p["max_zoom_factor"] = MaxZoomFactor;
        return p;
    }
}
