namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies glass-like blur by randomly displacing pixels then smoothing.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GlassBlur<T> : ImageAugmenterBase<T>
{
    public double Sigma { get; }
    public int MaxDelta { get; }
    public int Iterations { get; }

    public GlassBlur(double sigma = 0.7, int maxDelta = 2, int iterations = 2, double probability = 0.5)
        : base(probability)
    {
        Sigma = sigma; MaxDelta = maxDelta; Iterations = iterations;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();

        for (int iter = 0; iter < Iterations; iter++)
        {
            var temp = result.Clone();
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    int dy = context.GetRandomInt(-MaxDelta, MaxDelta + 1);
                    int dx = context.GetRandomInt(-MaxDelta, MaxDelta + 1);
                    int sy = Math.Max(0, Math.Min(data.Height - 1, y + dy));
                    int sx = Math.Max(0, Math.Min(data.Width - 1, x + dx));
                    for (int c = 0; c < data.Channels; c++)
                        result.SetPixel(y, x, c, temp.GetPixel(sy, sx, c));
                }
        }

        // Apply light Gaussian blur to smooth
        if (Sigma > 0)
        {
            var blur = new GaussianBlur<T>(Sigma, Sigma, probability: 1.0);
            result = blur.Apply(result, context);
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["sigma"] = Sigma; p["max_delta"] = MaxDelta; p["iterations"] = Iterations;
        return p;
    }
}
