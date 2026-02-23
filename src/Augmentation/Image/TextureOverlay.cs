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
        var texture = new double[data.Height, data.Width];
        int gH = data.Height / gridSize + 2;
        int gW = data.Width / gridSize + 2;
        var grid = new double[gH, gW];
        for (int y = 0; y < gH; y++)
            for (int x = 0; x < gW; x++)
                grid[y, x] = context.GetRandomDouble(0, 1);

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double fy = (double)y / gridSize;
                double fx = (double)x / gridSize;
                int y0 = (int)fy; int x0 = (int)fx;
                int y1 = Math.Min(y0 + 1, gH - 1);
                int x1 = Math.Min(x0 + 1, gW - 1);
                double ty = fy - y0; double tx = fx - x0;

                texture[y, x] = grid[y0, x0] * (1 - ty) * (1 - tx) +
                                grid[y1, x0] * ty * (1 - tx) +
                                grid[y0, x1] * (1 - ty) * tx +
                                grid[y1, x1] * ty * tx;
            }

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double tex = texture[y, x] * maxVal;
                    double blended = val * (1 - alpha) + tex * alpha;
                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, blended))));
                }

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
