namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Replaces random regions with their superpixel (average color) representation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Superpixels<T> : ImageAugmenterBase<T>
{
    public int GridSize { get; }
    public double MinReplaceRate { get; }
    public double MaxReplaceRate { get; }

    public Superpixels(int gridSize = 16, double minReplaceRate = 0.1, double maxReplaceRate = 0.5,
        double probability = 0.5) : base(probability)
    {
        if (gridSize < 2) throw new ArgumentOutOfRangeException(nameof(gridSize));
        GridSize = gridSize; MinReplaceRate = minReplaceRate; MaxReplaceRate = maxReplaceRate;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double replaceRate = context.GetRandomDouble(MinReplaceRate, MaxReplaceRate);

        int cellH = Math.Max(1, data.Height / GridSize);
        int cellW = Math.Max(1, data.Width / GridSize);
        int gridH = (data.Height + cellH - 1) / cellH;
        int gridW = (data.Width + cellW - 1) / cellW;

        var engine = Engine;
        var original = data.ToPlanarTensor();
        var image = original;
        int channels = data.Channels;
        for (int gy = 0; gy < gridH; gy++)
        {
            for (int gx = 0; gx < gridW; gx++)
            {
                if (context.GetRandomDouble(0, 1) > replaceRate) continue;

                int startY = gy * cellH;
                int startX = gx * cellW;
                int endY = Math.Min(startY + cellH, data.Height);
                int endX = Math.Min(startX + cellW, data.Width);
                int count = (endY - startY) * (endX - startX);

                // Replace the cell's pixels with its average color
                var cell = engine.TensorSlice(original, [0, startY, startX], [channels, endY - startY, endX - startX]);
                var avg = engine.ReduceMean(cell, [1, 2], true);
                image = engine.TensorSetSlice(image, engine.TensorBroadcastTo(avg, cell._shape), [0, startY, startX]);
            }
        }

        result.SetFromPlanarTensor(image);

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["grid_size"] = GridSize;
        p["min_replace_rate"] = MinReplaceRate; p["max_replace_rate"] = MaxReplaceRate;
        return p;
    }
}
