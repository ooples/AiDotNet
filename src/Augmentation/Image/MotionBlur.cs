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
        // Taps at round(half + offset * sin/cos) for offset = -half .. half, each set once, normalized
        double cosA = Math.Cos(angle), sinA = Math.Sin(angle);
        var taps = new List<int>(kSize);
        for (int i = 0; i < kSize; i++)
        {
            double offset = i - half;
            int ky = (int)Math.Round(half + offset * sinA);
            int kx = (int)Math.Round(half + offset * cosA);
            if (ky >= 0 && ky < kSize && kx >= 0 && kx < kSize)
            {
                taps.Add(ky * kSize + kx);
            }
        }

        var engine = Engine;
        var line = new Tensor<T>([kSize * kSize]);
        if (taps.Count > 0)
        {
            line = engine.TensorScatter(line, new Tensor<int>(taps.ToArray(), [taps.Count]),
                ImageTensor<T>.Constant(new Tensor<T>([taps.Count]), 1.0), 0);
        }

        var kernel = ImageTensor<T>.NormalizeKernel(engine.Reshape(line, [kSize, kSize]));

        // Convolve
        var result = data.Clone();
        result.SetFromPlanarTensor(
            ImageTensor<T>.Correlate(data.ToPlanarTensor(), kernel, mirror: false));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_kernel_size"] = MinKernelSize; p["max_kernel_size"] = MaxKernelSize;
        return p;
    }
}
