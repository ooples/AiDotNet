namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Drops rectangular regions from the image (similar to Cutout/Random Erasing).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CoarseDropout<T> : ImageAugmenterBase<T>
{
    public int MinHoles { get; }
    public int MaxHoles { get; }
    public double MinHoleHeight { get; }
    public double MaxHoleHeight { get; }
    public double MinHoleWidth { get; }
    public double MaxHoleWidth { get; }
    public double FillValue { get; }
    public bool FillWithNoise { get; }

    public CoarseDropout(int minHoles = 1, int maxHoles = 8,
        double minHoleHeight = 0.02, double maxHoleHeight = 0.1,
        double minHoleWidth = 0.02, double maxHoleWidth = 0.1,
        double fillValue = 0.0, bool fillWithNoise = false,
        double probability = 0.5) : base(probability)
    {
        MinHoles = minHoles; MaxHoles = maxHoles;
        MinHoleHeight = minHoleHeight; MaxHoleHeight = maxHoleHeight;
        MinHoleWidth = minHoleWidth; MaxHoleWidth = maxHoleWidth;
        FillValue = fillValue; FillWithNoise = fillWithNoise;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numHoles = context.GetRandomInt(MinHoles, MaxHoles + 1);

        for (int h = 0; h < numHoles; h++)
        {
            int holeH = Math.Max(1, Math.Min(data.Height, (int)(data.Height * context.GetRandomDouble(MinHoleHeight, MaxHoleHeight))));
            int holeW = Math.Max(1, Math.Min(data.Width, (int)(data.Width * context.GetRandomDouble(MinHoleWidth, MaxHoleWidth))));
            int y1 = context.GetRandomInt(0, Math.Max(1, data.Height - holeH + 1));
            int x1 = context.GetRandomInt(0, Math.Max(1, data.Width - holeW + 1));

            for (int y = y1; y < y1 + holeH; y++)
                for (int x = x1; x < x1 + holeW; x++)
                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = FillWithNoise
                            ? context.GetRandomDouble(0, maxVal)
                            : FillValue;
                        result.SetPixel(y, x, c, NumOps.FromDouble(val));
                    }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_holes"] = MinHoles; p["max_holes"] = MaxHoles;
        p["min_hole_height"] = MinHoleHeight; p["max_hole_height"] = MaxHoleHeight;
        p["min_hole_width"] = MinHoleWidth; p["max_hole_width"] = MaxHoleWidth;
        p["fill_value"] = FillValue; p["fill_with_noise"] = FillWithNoise;
        return p;
    }
}
