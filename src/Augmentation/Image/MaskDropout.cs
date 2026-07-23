namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Drops random object masks from the segmentation mask.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MaskDropout<T> : ImageAugmenterBase<T>
{
    public double DropRate { get; }
    public double FillValue { get; }

    public MaskDropout(double dropRate = 0.3, double fillValue = 0.0,
        double probability = 0.5) : base(probability)
    {
        DropRate = dropRate; FillValue = fillValue;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();

        // Find unique non-zero values (object IDs) in first channel
        var objectIds = new HashSet<int>();
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
            {
                int id = (int)NumOps.ToDouble(data.GetPixel(y, x, 0));
                if (id != 0) objectIds.Add(id);
            }

        // Randomly select objects to drop
        var dropIds = new HashSet<int>();
        foreach (int id in objectIds)
            if (context.GetRandomDouble(0, 1) < DropRate)
                dropIds.Add(id);

        // Fill dropped object regions
        if (dropIds.Count > 0)
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    int id = (int)NumOps.ToDouble(data.GetPixel(y, x, 0));
                    if (dropIds.Contains(id))
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
        p["drop_rate"] = DropRate; p["fill_value"] = FillValue;
        return p;
    }
}
