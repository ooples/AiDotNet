namespace AiDotNet.Augmentation.Image;

/// <summary>
/// GridMask augmentation (Chen et al. 2020) that masks grid-shaped regions.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GridMask<T> : ImageAugmenterBase<T>
{
    public double MinRatio { get; }
    public double MaxRatio { get; }
    public int GridDMin { get; }
    public int GridDMax { get; }
    public bool Rotate { get; }
    public double FillValue { get; }

    public GridMask(double minRatio = 0.3, double maxRatio = 0.7,
        int gridDMin = 64, int gridDMax = 128, bool rotate = true,
        double fillValue = 0.0, double probability = 0.5) : base(probability)
    {
        MinRatio = minRatio; MaxRatio = maxRatio;
        GridDMin = gridDMin; GridDMax = gridDMax;
        Rotate = rotate; FillValue = fillValue;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int d = context.GetRandomInt(GridDMin, GridDMax + 1);
        double ratio = context.GetRandomDouble(MinRatio, MaxRatio);
        int maskLen = (int)(d * ratio);

        double angle = Rotate ? context.GetRandomDouble(0, Math.PI * 2) : 0;
        double cosA = Math.Cos(angle);
        double sinA = Math.Sin(angle);

        double centerY = data.Height / 2.0;
        double centerX = data.Width / 2.0;

        int offsetY = context.GetRandomInt(0, d);
        int offsetX = context.GetRandomInt(0, d);

        for (int y = 0; y < data.Height; y++)
        {
            for (int x = 0; x < data.Width; x++)
            {
                // Rotate coordinates around center
                double dy = y - centerY;
                double dx = x - centerX;
                double ry = dy * cosA - dx * sinA + centerY;
                double rx = dy * sinA + dx * cosA + centerX;

                int gridY = (int)(((ry + offsetY) % d + d) % d);
                int gridX = (int)(((rx + offsetX) % d + d) % d);

                if (gridY < maskLen && gridX < maskLen)
                {
                    for (int c = 0; c < data.Channels; c++)
                        result.SetPixel(y, x, c, NumOps.FromDouble(FillValue));
                }
            }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_ratio"] = MinRatio; p["max_ratio"] = MaxRatio;
        p["grid_d_min"] = GridDMin; p["grid_d_max"] = GridDMax;
        p["rotate"] = Rotate; p["fill_value"] = FillValue;
        return p;
    }
}
