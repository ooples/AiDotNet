namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates fog effect by blending the image with a fog layer.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Fog<T> : ImageAugmenterBase<T>
{
    public double MinFogCoeff { get; }
    public double MaxFogCoeff { get; }
    public double MinAlpha { get; }
    public double MaxAlpha { get; }

    public Fog(double minFogCoeff = 0.3, double maxFogCoeff = 1.0,
        double minAlpha = 0.08, double maxAlpha = 0.5,
        double probability = 0.5) : base(probability)
    {
        MinFogCoeff = minFogCoeff; MaxFogCoeff = maxFogCoeff;
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double fogCoeff = context.GetRandomDouble(MinFogCoeff, MaxFogCoeff);
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);

        // Generate Perlin-like fog using multiple octaves of random noise
        var fogMap = new double[data.Height, data.Width];
        int gridSize = Math.Max(4, Math.Min(data.Height, data.Width) / 8);

        // Create coarse random grid
        int gridH = data.Height / gridSize + 2;
        int gridW = data.Width / gridSize + 2;
        var grid = new double[gridH, gridW];
        for (int gy = 0; gy < gridH; gy++)
            for (int gx = 0; gx < gridW; gx++)
                grid[gy, gx] = context.GetRandomDouble(0, 1);

        // Bilinear interpolation of grid
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double gy = (double)y / gridSize;
                double gx = (double)x / gridSize;
                int gy0 = (int)gy; int gx0 = (int)gx;
                int gy1 = Math.Min(gy0 + 1, gridH - 1);
                int gx1 = Math.Min(gx0 + 1, gridW - 1);
                double fy = gy - gy0; double fx = gx - gx0;

                fogMap[y, x] = grid[gy0, gx0] * (1 - fy) * (1 - fx) +
                               grid[gy1, gx0] * fy * (1 - fx) +
                               grid[gy0, gx1] * (1 - fy) * fx +
                               grid[gy1, gx1] * fy * fx;
            }

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                double fogStrength = fogMap[y, x] * alpha * fogCoeff;
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double fogged = val * (1 - fogStrength) + maxVal * fogStrength;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, fogged))));
                }
            }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_fog_coeff"] = MinFogCoeff; p["max_fog_coeff"] = MaxFogCoeff;
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        return p;
    }
}
