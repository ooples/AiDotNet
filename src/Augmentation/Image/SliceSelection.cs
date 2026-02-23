namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Randomly selects a slice from a multi-channel volume (simulates 3D volume slice selection).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SliceSelection<T> : ImageAugmenterBase<T>
{
    public int OutputChannels { get; }

    public SliceSelection(int outputChannels = 1, double probability = 0.5) : base(probability)
    {
        OutputChannels = outputChannels;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels <= OutputChannels) return data.Clone();

        int startChannel = context.GetRandomInt(0, data.Channels - OutputChannels + 1);
        var result = new ImageTensor<T>(height: data.Height, width: data.Width, channels: OutputChannels);

        for (int c = 0; c < OutputChannels; c++)
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                    result.SetPixel(y, x, c, data.GetPixel(y, x, startChannel + c));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["output_channels"] = OutputChannels;
        return p;
    }
}
