namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly sets individual pixels to a fill value.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PixelDropout<T> : ImageAugmenterBase<T>
{
    public double DropoutRate { get; }
    public double FillValue { get; }
    public bool PerChannel { get; }

    public PixelDropout(double dropoutRate = 0.05, double fillValue = 0.0,
        bool perChannel = false, double probability = 0.5) : base(probability)
    {
        if (dropoutRate < 0 || dropoutRate > 1) throw new ArgumentOutOfRangeException(nameof(dropoutRate));
        DropoutRate = dropoutRate; FillValue = fillValue; PerChannel = perChannel;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();

        if (PerChannel)
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    for (int c = 0; c < data.Channels; c++)
                    {
                        if (context.GetRandomDouble(0, 1) < DropoutRate)
                            result.SetPixel(y, x, c, NumOps.FromDouble(FillValue));
                    }
        }
        else
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    if (context.GetRandomDouble(0, 1) < DropoutRate)
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
        p["dropout_rate"] = DropoutRate; p["fill_value"] = FillValue;
        p["per_channel"] = PerChannel;
        return p;
    }
}
