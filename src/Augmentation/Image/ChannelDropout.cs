namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly zeros out one or more color channels.
/// </summary>
/// <remarks>
/// <para>ChannelDropout sets entire color channels to a fill value (default 0), forcing the
/// model to learn from the remaining channels. This prevents over-reliance on any single
/// color channel.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ChannelDropout<T> : ImageAugmenterBase<T>
{
    public int MinChannelsToDrop { get; }
    public int MaxChannelsToDrop { get; }
    public double FillValue { get; }

    public ChannelDropout(int minChannelsToDrop = 1, int maxChannelsToDrop = 1,
        double fillValue = 0, double probability = 0.5) : base(probability)
    {
        if (minChannelsToDrop < 1) throw new ArgumentOutOfRangeException(nameof(minChannelsToDrop));
        if (maxChannelsToDrop < minChannelsToDrop) throw new ArgumentException("max must be >= min.");
        MinChannelsToDrop = minChannelsToDrop;
        MaxChannelsToDrop = maxChannelsToDrop;
        FillValue = fillValue;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int clampedMin = Math.Min(MinChannelsToDrop, data.Channels - 1);
        int clampedMax = Math.Min(MaxChannelsToDrop, data.Channels - 1);
        clampedMin = Math.Max(1, clampedMin);
        clampedMax = Math.Max(clampedMin, clampedMax);
        int numToDrop = context.GetRandomInt(clampedMin, clampedMax + 1);

        // Select channels to drop
        var candidates = new List<int>();
        for (int i = 0; i < data.Channels; i++) candidates.Add(i);
        var toDrop = new HashSet<int>();

        for (int i = 0; i < numToDrop && candidates.Count > 0; i++)
        {
            int idx = context.GetRandomInt(0, candidates.Count);
            toDrop.Add(candidates[idx]);
            candidates.RemoveAt(idx);
        }

        var result = data.Clone();
        T fill = NumOps.FromDouble(FillValue);

        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                foreach (int c in toDrop)
                    result.SetPixel(y, x, c, fill);

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_channels_to_drop"] = MinChannelsToDrop;
        p["max_channels_to_drop"] = MaxChannelsToDrop;
        p["fill_value"] = FillValue;
        return p;
    }
}
