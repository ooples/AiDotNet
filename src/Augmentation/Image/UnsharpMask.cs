namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies unsharp masking to sharpen the image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class UnsharpMask<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }
    public double Sigma { get; }
    public double Amount { get; }
    public double Threshold { get; }

    public UnsharpMask(int kernelSize = 5, double sigma = 1.0,
        double amount = 1.0, double threshold = 0.0,
        double probability = 0.5) : base(probability)
    {
        if (kernelSize < 3 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize; Sigma = sigma;
        Amount = amount; Threshold = threshold;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int half = KernelSize / 2;

        // Build Gaussian kernel
        var kernel = new double[KernelSize, KernelSize];
        double sum = 0;
        for (int ky = -half; ky <= half; ky++)
            for (int kx = -half; kx <= half; kx++)
            {
                double g = Math.Exp(-(ky * ky + kx * kx) / (2 * Sigma * Sigma));
                kernel[ky + half, kx + half] = g;
                sum += g;
            }
        for (int ky = 0; ky < KernelSize; ky++)
            for (int kx = 0; kx < KernelSize; kx++)
                kernel[ky, kx] /= sum;

        for (int c = 0; c < data.Channels; c++)
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double original = NumOps.ToDouble(data.GetPixel(y, x, c));

                    // Compute blurred value
                    double blurred = 0;
                    for (int ky = -half; ky <= half; ky++)
                        for (int kx = -half; kx <= half; kx++)
                        {
                            int ny = Math.Max(0, Math.Min(data.Height - 1, y + ky));
                            int nx = Math.Max(0, Math.Min(data.Width - 1, x + kx));
                            blurred += NumOps.ToDouble(data.GetPixel(ny, nx, c)) * kernel[ky + half, kx + half];
                        }

                    double diff = original - blurred;
                    double sharpened;
                    if (Math.Abs(diff) >= Threshold * maxVal)
                        sharpened = original + Amount * diff;
                    else
                        sharpened = original;

                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, sharpened))));
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["sigma"] = Sigma;
        p["amount"] = Amount; p["threshold"] = Threshold;
        return p;
    }
}
