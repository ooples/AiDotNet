namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Hide-and-Seek augmentation (Singh &amp; Lee 2017) that randomly hides grid patches.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HideAndSeek<T> : ImageAugmenterBase<T>
{
    public int GridSize { get; }
    public double HideProbability { get; }
    public double FillValue { get; }

    public HideAndSeek(int gridSize = 4, double hideProbability = 0.5,
        double fillValue = 0.0, double probability = 0.5) : base(probability)
    {
        if (gridSize < 1) throw new ArgumentOutOfRangeException(nameof(gridSize));
        GridSize = gridSize; HideProbability = hideProbability; FillValue = fillValue;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int patchH = Math.Max(1, data.Height / GridSize);
        int patchW = Math.Max(1, data.Width / GridSize);

        for (int gy = 0; gy < GridSize; gy++)
        {
            for (int gx = 0; gx < GridSize; gx++)
            {
                if (context.GetRandomDouble(0, 1) >= HideProbability) continue;

                int startY = gy * patchH;
                int startX = gx * patchW;
                int endY = Math.Min(startY + patchH, data.Height);
                int endX = Math.Min(startX + patchW, data.Width);

                for (int y = startY; y < endY; y++)
                    for (int x = startX; x < endX; x++)
                        for (int c = 0; c < data.Channels; c++)
                            result.SetPixel(y, x, c, NumOps.FromDouble(FillValue));
            }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["grid_size"] = GridSize; p["hide_probability"] = HideProbability;
        p["fill_value"] = FillValue;
        return p;
    }
}
