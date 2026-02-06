namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates cloud overlay on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Clouds<T> : ImageAugmenterBase<T>
{
    public double MinOpacity { get; }
    public double MaxOpacity { get; }
    public double MinScale { get; }
    public double MaxScale { get; }

    public Clouds(double minOpacity = 0.2, double maxOpacity = 0.6,
        double minScale = 0.05, double maxScale = 0.2,
        double probability = 0.5) : base(probability)
    {
        MinOpacity = minOpacity; MaxOpacity = maxOpacity;
        MinScale = minScale; MaxScale = maxScale;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double opacity = context.GetRandomDouble(MinOpacity, MaxOpacity);
        double scale = context.GetRandomDouble(MinScale, MaxScale);
        int gridSize = Math.Max(2, (int)(Math.Min(data.Height, data.Width) * scale));

        // Generate cloud noise map with multiple octaves
        var cloudMap = new double[data.Height, data.Width];

        for (int octave = 0; octave < 4; octave++)
        {
            int gs = Math.Max(2, gridSize / (1 << octave));
            double weight = 1.0 / (1 << octave);
            int gH = data.Height / gs + 2;
            int gW = data.Width / gs + 2;

            var grid = new double[gH, gW];
            for (int gy = 0; gy < gH; gy++)
                for (int gx = 0; gx < gW; gx++)
                    grid[gy, gx] = context.GetRandomDouble(0, 1);

            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double fy = (double)y / gs;
                    double fx = (double)x / gs;
                    int y0 = (int)fy; int x0 = (int)fx;
                    int y1 = Math.Min(y0 + 1, gH - 1);
                    int x1 = Math.Min(x0 + 1, gW - 1);
                    double ty = fy - y0; double tx = fx - x0;

                    double val = grid[y0, x0] * (1 - ty) * (1 - tx) +
                                 grid[y1, x0] * ty * (1 - tx) +
                                 grid[y0, x1] * (1 - ty) * tx +
                                 grid[y1, x1] * ty * tx;

                    cloudMap[y, x] += val * weight;
                }
        }

        // Normalize and threshold
        double minC = double.MaxValue, maxC = double.MinValue;
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                if (cloudMap[y, x] < minC) minC = cloudMap[y, x];
                if (cloudMap[y, x] > maxC) maxC = cloudMap[y, x];
            }

        double rangeC = maxC - minC;
        if (rangeC < 1e-10) rangeC = 1;

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double cloud = (cloudMap[y, x] - minC) / rangeC;
                cloud = Math.Max(0, cloud - 0.3) / 0.7; // threshold
                double alpha = cloud * opacity;

                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double clouded = val * (1 - alpha) + maxVal * alpha;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, clouded))));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_opacity"] = MinOpacity; p["max_opacity"] = MaxOpacity;
        p["min_scale"] = MinScale; p["max_scale"] = MaxScale;
        return p;
    }
}
