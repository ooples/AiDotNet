namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates defocus blur using a circular (disc) kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Defocus<T> : ImageAugmenterBase<T>
{
    public int MinRadius { get; }
    public int MaxRadius { get; }

    public Defocus(int minRadius = 1, int maxRadius = 3, double probability = 0.5)
        : base(probability)
    {
        if (minRadius < 1) throw new ArgumentOutOfRangeException(nameof(minRadius));
        MinRadius = minRadius; MaxRadius = maxRadius;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int radius = context.GetRandomInt(MinRadius, MaxRadius + 1);
        int kSize = radius * 2 + 1;

        // Create circular kernel
        var kernel = new double[kSize, kSize];
        double sum = 0;
        for (int ky = 0; ky < kSize; ky++)
            for (int kx = 0; kx < kSize; kx++)
            {
                double dx = kx - radius, dy = ky - radius;
                if (dx * dx + dy * dy <= (double)radius * radius)
                {
                    kernel[ky, kx] = 1.0;
                    sum += 1.0;
                }
            }

        if (sum > 0)
            for (int ky = 0; ky < kSize; ky++)
                for (int kx = 0; kx < kSize; kx++)
                    kernel[ky, kx] /= sum;

        var result = data.Clone();
        for (int c = 0; c < data.Channels; c++)
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double val = 0;
                    for (int ky = 0; ky < kSize; ky++)
                        for (int kx = 0; kx < kSize; kx++)
                        {
                            int sy = Math.Max(0, Math.Min(data.Height - 1, y + ky - radius));
                            int sx = Math.Max(0, Math.Min(data.Width - 1, x + kx - radius));
                            val += NumOps.ToDouble(data.GetPixel(sy, sx, c)) * kernel[ky, kx];
                        }
                    result.SetPixel(y, x, c, NumOps.FromDouble(val));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_radius"] = MinRadius; p["max_radius"] = MaxRadius; return p;
    }
}
