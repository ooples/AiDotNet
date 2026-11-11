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

    /// <inheritdoc />
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

        var image = data.ToPlanarTensor();
        T fill = NumOps.FromDouble(FillValue);
        for (int gy = 0; gy < GridSize + 1; gy++)
        {
            for (int gx = 0; gx < GridSize + 1; gx++)
            {
                int startY = gy * cellH + offsetY;
                int startX = gx * cellW + offsetX;
                int endY = Math.Min(startY + dropH, data.Height);
                int endX = Math.Min(startX + dropW, data.Width);
                if (endY > startY && endX > startX)
                {
                    image = ImageTensor<T>.FillRect(image, startY, startX, endY - startY, endX - startX, fill);
                }
            }
        }

        result.SetFromPlanarTensor(image);

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["ratio"] = Ratio; p["grid_size"] = GridSize;
        p["fill_value"] = FillValue; p["random_offset"] = RandomOffset;
        return p;
    }
}
