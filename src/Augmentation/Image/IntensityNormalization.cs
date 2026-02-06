namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Medical image intensity normalization using z-score or min-max normalization.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class IntensityNormalization<T> : ImageAugmenterBase<T>
{
    public IntensityNormMethod Method { get; }
    public double ClipMin { get; }
    public double ClipMax { get; }

    public IntensityNormalization(IntensityNormMethod method = IntensityNormMethod.ZScore,
        double clipMin = -3.0, double clipMax = 3.0,
        double probability = 1.0) : base(probability)
    {
        Method = method; ClipMin = clipMin; ClipMax = clipMax;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        for (int c = 0; c < data.Channels; c++)
        {
            double mean = 0, min = double.MaxValue, max = double.MinValue;
            int count = data.Height * data.Width;

            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    mean += val;
                    if (val < min) min = val;
                    if (val > max) max = val;
                }
            mean /= count;

            if (Method == IntensityNormMethod.ZScore)
            {
                double variance = 0;
                for (int y = 0; y < data.Height; y++)
                    for (int x = 0; x < data.Width; x++)
                    {
                        double d = NumOps.ToDouble(data.GetPixel(y, x, c)) - mean;
                        variance += d * d;
                    }
                double std = Math.Sqrt(variance / count + 1e-8);

                for (int y = 0; y < data.Height; y++)
                    for (int x = 0; x < data.Width; x++)
                    {
                        double val = (NumOps.ToDouble(data.GetPixel(y, x, c)) - mean) / std;
                        val = Math.Max(ClipMin, Math.Min(ClipMax, val));
                        // Map back to [0, maxVal]
                        val = (val - ClipMin) / (ClipMax - ClipMin) * maxVal;
                        result.SetPixel(y, x, c, NumOps.FromDouble(val));
                    }
            }
            else // MinMax
            {
                double range = max - min;
                if (range < 1e-10) range = 1;
                for (int y = 0; y < data.Height; y++)
                    for (int x = 0; x < data.Width; x++)
                    {
                        double val = (NumOps.ToDouble(data.GetPixel(y, x, c)) - min) / range * maxVal;
                        result.SetPixel(y, x, c, NumOps.FromDouble(val));
                    }
            }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["method"] = Method.ToString();
        p["clip_min"] = ClipMin; p["clip_max"] = ClipMax;
        return p;
    }
}

/// <summary>Intensity normalization method.</summary>
public enum IntensityNormMethod
{
    /// <summary>Z-score normalization (zero mean, unit variance).</summary>
    ZScore,
    /// <summary>Min-max normalization to [0, 1] range.</summary>
    MinMax
}
