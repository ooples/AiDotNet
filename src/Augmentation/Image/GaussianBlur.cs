
namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies Gaussian blur to an image.
/// </summary>
/// <remarks>
/// <para>
/// Gaussian blur smooths the image by convolving it with a Gaussian kernel. This simulates
/// out-of-focus images or motion blur, helping the model become robust to blurry inputs.
/// </para>
/// <para><b>For Beginners:</b> Think of this like looking at a photo through frosted glass.
/// The image becomes softer and details are less sharp. This teaches your model to recognize
/// objects even when they're not perfectly in focus.
/// </para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When deployed images may be out of focus</item>
/// <item>When training data is too sharp compared to real-world images</item>
/// <item>To reduce high-frequency noise sensitivity</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GaussianBlur<T> : AugmentationBase<T, ImageTensor<T>>
{
    /// <summary>
    /// Gets the minimum sigma (standard deviation) for the Gaussian kernel.
    /// </summary>
    public double MinSigma { get; }

    /// <summary>
    /// Gets the maximum sigma (standard deviation) for the Gaussian kernel.
    /// </summary>
    public double MaxSigma { get; }

    /// <summary>
    /// Gets the kernel size for the Gaussian blur.
    /// </summary>
    public int KernelSize { get; }

    /// <summary>
    /// Creates a new Gaussian blur augmentation.
    /// </summary>
    /// <param name="minSigma">
    /// The minimum sigma for the Gaussian kernel.
    /// Industry standard default is 0.1.
    /// </param>
    /// <param name="maxSigma">
    /// The maximum sigma for the Gaussian kernel.
    /// Industry standard default is 2.0.
    /// </param>
    /// <param name="kernelSize">
    /// The kernel size for the blur. Use 0 for automatic size based on sigma.
    /// Industry standard default is 0 (automatic).
    /// </param>
    /// <param name="probability">
    /// The probability of applying this augmentation (0.0 to 1.0).
    /// Industry standard default is 0.5.
    /// </param>
    public GaussianBlur(
        double minSigma = 0.1,
        double maxSigma = 2.0,
        int kernelSize = 0,
        double probability = 0.5)
        : base(probability)
    {
        if (minSigma <= 0)
            throw new ArgumentOutOfRangeException(nameof(minSigma), "Sigma must be positive");
        if (maxSigma <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSigma), "Sigma must be positive");
        if (minSigma > maxSigma)
            throw new ArgumentException("minSigma must be <= maxSigma");
        if (kernelSize < 0)
            throw new ArgumentOutOfRangeException(nameof(kernelSize), "Kernel size must be non-negative");
        if (kernelSize > 0 && kernelSize % 2 == 0)
            throw new ArgumentException("Kernel size must be odd", nameof(kernelSize));

        MinSigma = minSigma;
        MaxSigma = maxSigma;
        KernelSize = kernelSize;
    }

    /// <summary>
    /// Applies Gaussian blur to the image.
    /// </summary>
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Sample sigma for this augmentation
        double sigma = context.GetRandomDouble(MinSigma, MaxSigma);

        // Determine kernel size
        int kSize = KernelSize;
        if (kSize == 0)
        {
            // Automatic size: typically 3*sigma, rounded to odd integer
            kSize = (int)(sigma * 6) | 1; // Ensure odd
            kSize = Math.Max(3, Math.Min(31, kSize)); // Clamp to reasonable range
        }

        // Generate Gaussian kernel
        var kernel = ImageTensor<T>.GaussianKernel(kSize, sigma);

        // Apply convolution
        return ApplyConvolution(data, kernel);
    }

    /// <summary>
    /// Applies a convolution with the given kernel.
    /// </summary>
    private static ImageTensor<T> ApplyConvolution(ImageTensor<T> image, Tensor<T> kernel)
    {
        var result = image.Clone();

        // Reflect at boundaries
        var blurred = ImageTensor<T>.Correlate(image.ToPlanarTensor(), kernel, mirror: true);
        result.SetFromPlanarTensor(blurred);

        return result;
    }

    /// <summary>
    /// Reflects an index at boundaries.
    /// </summary>
    private static int ReflectIndex(int index, int size)
    {
        if (index < 0)
            return -index - 1;
        if (index >= size)
            return 2 * size - index - 1;
        return index;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["min_sigma"] = MinSigma;
        parameters["max_sigma"] = MaxSigma;
        parameters["kernel_size"] = KernelSize == 0 ? "auto" : KernelSize;
        return parameters;
    }
}
