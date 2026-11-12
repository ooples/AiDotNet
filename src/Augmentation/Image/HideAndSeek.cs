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

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        int patchH = Math.Max(1, data.Height / GridSize);
        int patchW = Math.Max(1, data.Width / GridSize);

        var image = data.ToPlanarTensor();
        T fill = NumOps.FromDouble(FillValue);
        for (int gy = 0; gy < GridSize; gy++)
        {
            for (int gx = 0; gx < GridSize; gx++)
            {
                if (context.GetRandomDouble(0, 1) >= HideProbability) continue;

                int startY = gy * patchH;
                int startX = gx * patchW;
                int endY = Math.Min(startY + patchH, data.Height);
                int endX = Math.Min(startX + patchW, data.Width);

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
        p["grid_size"] = GridSize; p["hide_probability"] = HideProbability;
        p["fill_value"] = FillValue;
        return p;
    }
}
