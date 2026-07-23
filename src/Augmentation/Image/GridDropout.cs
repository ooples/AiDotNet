namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Drops cells from a regular grid overlay on the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GridDropout<T> : ImageAugmenterBase<T>
{
    public double Ratio { get; }
    public int GridSize { get; }
    public double FillValue { get; }
    public bool RandomOffset { get; }

    public GridDropout(double ratio = 0.5, int gridSize = 4,
        double fillValue = 0.0, bool randomOffset = true,
        double probability = 0.5) : base(probability)
    {
        if (ratio < 0 || ratio > 1) throw new ArgumentOutOfRangeException(nameof(ratio));
        if (gridSize < 1) throw new ArgumentOutOfRangeException(nameof(gridSize));
        Ratio = ratio; GridSize = gridSize;
        FillValue = fillValue; RandomOffset = randomOffset;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        if (Ratio <= 0) return result;

        int cellH = Math.Max(1, data.Height / GridSize);
        int cellW = Math.Max(1, data.Width / GridSize);
        int dropH = Math.Max(1, (int)(cellH * Ratio));
        int dropW = Math.Max(1, (int)(cellW * Ratio));

        int offsetY = RandomOffset ? context.GetRandomInt(0, cellH) : 0;
        int offsetX = RandomOffset ? context.GetRandomInt(0, cellW) : 0;

        for (int gy = 0; gy < GridSize + 1; gy++)
        {
            for (int gx = 0; gx < GridSize + 1; gx++)
            {
                int startY = gy * cellH + offsetY;
                int startX = gx * cellW + offsetX;

                for (int y = startY; y < startY + dropH && y < data.Height; y++)
                    for (int x = startX; x < startX + dropW && x < data.Width; x++)
                    {
                        if (y < 0) continue;
                        if (x < 0) continue;
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
        p["ratio"] = Ratio; p["grid_size"] = GridSize;
        p["fill_value"] = FillValue; p["random_offset"] = RandomOffset;
        return p;
    }
}
