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
        if (sigma <= 0) throw new ArgumentOutOfRangeException(nameof(sigma), "Sigma must be positive.");
        KernelSize = kernelSize; Sigma = sigma;
        Amount = amount; Threshold = threshold;
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        int half = KernelSize / 2;

        // Build Gaussian kernel
        // (taps -half .. half)
        var kernel = ImageTensor<T>.GaussianKernel(2 * half + 1, Sigma);

        // Compute blurred values (edges replicated)
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var blurred = ImageTensor<T>.Correlate(image, kernel, mirror: false);

        // Sharpen only where |original - blurred| reaches the threshold
        var diff = engine.TensorSubtract(image, blurred);
        var boosted = engine.TensorAdd(image, engine.TensorMultiplyScalar(diff, NumOps.FromDouble(Amount)));
        var belowThreshold = engine.TensorLessThan(engine.TensorAbs(diff), NumOps.FromDouble(Threshold * maxVal));
        var sharpened = engine.TensorWhere(belowThreshold, image, boosted);
        result.SetFromPlanarTensor(engine.TensorClamp(sharpened, NumOps.Zero, NumOps.FromDouble(maxVal)));

        return result;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["sigma"] = Sigma;
        p["amount"] = Amount; p["threshold"] = Threshold;
        return p;
    }
}
