namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies simple box (mean) blur using a uniform kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BoxBlur<T> : ImageAugmenterBase<T>
{
    public int MinKernelSize { get; }
    public int MaxKernelSize { get; }

    public BoxBlur(int minKernelSize = 3, int maxKernelSize = 5, double probability = 0.5)
        : base(probability)
    {
        if (minKernelSize < 1) throw new ArgumentOutOfRangeException(nameof(minKernelSize));
        MinKernelSize = minKernelSize; MaxKernelSize = maxKernelSize;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        int kSize = context.GetRandomInt(MinKernelSize, MaxKernelSize + 1);
        if (kSize % 2 == 0) kSize++;
        int half = kSize / 2;
        double weight = 1.0 / (kSize * kSize);
        var result = data.Clone();

        for (int c = 0; c < data.Channels; c++)
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double sum = 0;
                    for (int ky = -half; ky <= half; ky++)
                        for (int kx = -half; kx <= half; kx++)
                        {
                            int sy = Math.Max(0, Math.Min(data.Height - 1, y + ky));
                            int sx = Math.Max(0, Math.Min(data.Width - 1, x + kx));
                            sum += NumOps.ToDouble(data.GetPixel(sy, sx, c));
                        }
                    result.SetPixel(y, x, c, NumOps.FromDouble(sum * weight));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters(); p["min_kernel_size"] = MinKernelSize; p["max_kernel_size"] = MaxKernelSize; return p;
    }
}
