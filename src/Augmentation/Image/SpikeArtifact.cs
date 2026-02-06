namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates MRI spike (Herringbone) artifacts caused by electrical interference in k-space.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpikeArtifact<T> : ImageAugmenterBase<T>
{
    public int MinSpikes { get; }
    public int MaxSpikes { get; }
    public double MinIntensity { get; }
    public double MaxIntensity { get; }

    public SpikeArtifact(int minSpikes = 1, int maxSpikes = 3,
        double minIntensity = 0.1, double maxIntensity = 0.5,
        double probability = 0.5) : base(probability)
    {
        MinSpikes = minSpikes; MaxSpikes = maxSpikes;
        MinIntensity = minIntensity; MaxIntensity = maxIntensity;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int numSpikes = context.GetRandomInt(MinSpikes, MaxSpikes + 1);

        for (int s = 0; s < numSpikes; s++)
        {
            double intensity = context.GetRandomDouble(MinIntensity, MaxIntensity);
            double freqX = context.GetRandomDouble(0.1, 0.5) * Math.PI * 2;
            double freqY = context.GetRandomDouble(0.1, 0.5) * Math.PI * 2;
            double phase = context.GetRandomDouble(0, Math.PI * 2);

            // Herringbone pattern: sinusoidal stripes
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double pattern = Math.Sin(freqX * x + freqY * y + phase) * intensity * maxVal;

                    for (int c = 0; c < data.Channels; c++)
                    {
                        double val = NumOps.ToDouble(result.GetPixel(y, x, c)) + pattern;
                        result.SetPixel(y, x, c,
                            NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                    }
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_spikes"] = MinSpikes; p["max_spikes"] = MaxSpikes;
        p["min_intensity"] = MinIntensity; p["max_intensity"] = MaxIntensity;
        return p;
    }
}
