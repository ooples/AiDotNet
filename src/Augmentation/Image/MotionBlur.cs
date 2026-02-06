namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies directional motion blur simulating camera or object motion.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MotionBlur<T> : ImageAugmenterBase<T>
{
    public int MinKernelSize { get; }
    public int MaxKernelSize { get; }

    public MotionBlur(int minKernelSize = 3, int maxKernelSize = 7, double probability = 0.5)
        : base(probability)
    {
        if (minKernelSize < 3) throw new ArgumentOutOfRangeException(nameof(minKernelSize));
        if (maxKernelSize < minKernelSize) throw new ArgumentOutOfRangeException(nameof(maxKernelSize), "maxKernelSize must be >= minKernelSize.");
        MinKernelSize = minKernelSize;
        MaxKernelSize = maxKernelSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int kSize = context.GetRandomInt(MinKernelSize, MaxKernelSize + 1);
        if (kSize % 2 == 0) kSize--;
        kSize = Math.Max(3, kSize);
        double angle = context.GetRandomDouble(0, 360) * Math.PI / 180.0;
        int half = kSize / 2;

        // Generate motion blur kernel along the angle
        var kernel = new double[kSize, kSize];
        var visited = new bool[kSize, kSize];
        double cosA = Math.Cos(angle), sinA = Math.Sin(angle);
        double sum = 0;

        for (int i = 0; i < kSize; i++)
        {
            double offset = i - half;
            int ky = (int)Math.Round(half + offset * sinA);
            int kx = (int)Math.Round(half + offset * cosA);
            if (ky >= 0 && ky < kSize && kx >= 0 && kx < kSize)
            {
                if (!visited[ky, kx])
                    sum += 1.0;
                visited[ky, kx] = true;
                kernel[ky, kx] = 1.0;
            }
        }

        if (sum > 0)
            for (int ky = 0; ky < kSize; ky++)
                for (int kx = 0; kx < kSize; kx++)
                    kernel[ky, kx] /= sum;

        // Convolve
        var result = data.Clone();
        for (int c = 0; c < data.Channels; c++)
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double val = 0;
                    for (int ky = 0; ky < kSize; ky++)
                        for (int kx = 0; kx < kSize; kx++)
                        {
                            int sy = Math.Max(0, Math.Min(data.Height - 1, y + ky - half));
                            int sx = Math.Max(0, Math.Min(data.Width - 1, x + kx - half));
                            val += NumOps.ToDouble(data.GetPixel(sy, sx, c)) * kernel[ky, kx];
                        }
                    result.SetPixel(y, x, c, NumOps.FromDouble(val));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_kernel_size"] = MinKernelSize; p["max_kernel_size"] = MaxKernelSize;
        return p;
    }
}
