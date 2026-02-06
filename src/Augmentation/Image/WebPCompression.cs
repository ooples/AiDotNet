namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates WebP compression artifacts (similar to JPEG but with different characteristics).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WebPCompression<T> : ImageAugmenterBase<T>
{
    public int MinQuality { get; }
    public int MaxQuality { get; }

    public WebPCompression(int minQuality = 40, int maxQuality = 100, double probability = 0.5)
        : base(probability)
    {
        if (minQuality < 1 || minQuality > 100) throw new ArgumentOutOfRangeException(nameof(minQuality));
        MinQuality = minQuality; MaxQuality = maxQuality;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int quality = context.GetRandomInt(MinQuality, MaxQuality + 1);
        double quantScale = (100 - quality) / 60.0; // WebP is slightly less aggressive than JPEG
        if (quantScale < 0.01) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // WebP uses 4x4 transform blocks (vs JPEG's 8x8)
        for (int c = 0; c < data.Channels; c++)
        {
            for (int by = 0; by < data.Height; by += 4)
            {
                for (int bx = 0; bx < data.Width; bx += 4)
                {
                    int blockH = Math.Min(4, data.Height - by);
                    int blockW = Math.Min(4, data.Width - bx);

                    double step = quantScale * maxVal * 0.08;
                    if (step > 0.001)
                    {
                        for (int y = 0; y < blockH; y++)
                            for (int x = 0; x < blockW; x++)
                            {
                                double val = NumOps.ToDouble(data.GetPixel(by + y, bx + x, c));
                                val = Math.Round(val / step) * step;
                                result.SetPixel(by + y, bx + x, c,
                                    NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, val))));
                            }
                    }
                }
            }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_quality"] = MinQuality; p["max_quality"] = MaxQuality; return p;
    }
}
