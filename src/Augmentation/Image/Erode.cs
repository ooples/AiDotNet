namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological erosion - shrinks bright regions using a structuring element.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Erode<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }
    public int Iterations { get; }

    public Erode(int kernelSize = 3, int iterations = 1,
        double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize; Iterations = iterations;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data;
        int half = KernelSize / 2;

        for (int iter = 0; iter < Iterations; iter++)
        {
            var input = result.Clone();
            result = input.Clone();

            for (int c = 0; c < data.Channels; c++)
                for (int y = 0; y < data.Height; y++)
                    for (int x = 0; x < data.Width; x++)
                    {
                        double minVal = double.MaxValue;
                        for (int ky = -half; ky <= half; ky++)
                            for (int kx = -half; kx <= half; kx++)
                            {
                                int ny = Math.Max(0, Math.Min(data.Height - 1, y + ky));
                                int nx = Math.Max(0, Math.Min(data.Width - 1, x + kx));
                                double val = NumOps.ToDouble(input.GetPixel(ny, nx, c));
                                if (val < minVal) minVal = val;
                            }
                        result.SetPixel(y, x, c, NumOps.FromDouble(minVal));
                    }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["iterations"] = Iterations;
        return p;
    }
}
