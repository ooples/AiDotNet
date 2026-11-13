using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// Binarization - Document binarization with multiple algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Binarization converts grayscale document images to binary (black and white),
/// which is essential for many OCR and document analysis tasks.
/// </para>
/// <para>
/// <b>For Beginners:</b> Binarization separates text from background by converting
/// each pixel to either black or white:
///
/// - Otsu: Global threshold, works well for uniform lighting
/// - Sauvola: Local adaptive, handles varying illumination
/// - Niblack: Local adaptive, good for degraded documents
/// - Fixed: Simple fixed threshold value
///
/// Key features:
/// - Multiple binarization algorithms
/// - Handles varying lighting conditions
/// - Works with degraded documents
/// - Configurable parameters
///
/// Example usage:
/// <code>
/// var binarizer = new Binarization&lt;float&gt;();
/// var binary = binarizer.Process(grayscale, BinarizationMethod.Sauvola);
/// </code>
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Binarization<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new Binarization instance.
    /// </summary>
    public Binarization()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Applies binarization to a document image.
    /// </summary>
    /// <param name="image">The input image tensor (grayscale).</param>
    /// <param name="method">The binarization method to use.</param>
    /// <returns>The binarized image.</returns>
    public Tensor<T> Process(Tensor<T> image, BinarizationMethod method = BinarizationMethod.Otsu)
    {
        // Convert to grayscale if needed
        var gray = ToGrayscale(image);

        return method switch
        {
            BinarizationMethod.Otsu => OtsuBinarization(gray),
            BinarizationMethod.Sauvola => SauvolaBinarization(gray),
            BinarizationMethod.Niblack => NiblackBinarization(gray),
            BinarizationMethod.Fixed => FixedThreshold(gray, 0.5),
            _ => OtsuBinarization(gray)
        };
    }

    /// <summary>
    /// Applies Otsu's method for global thresholding.
    /// </summary>
    public Tensor<T> OtsuBinarization(Tensor<T> gray)
    {
        var engine = AiDotNetEngine.Current;

        // Compute histogram over 256 bins: bin = clamp((int)(v * 255), 0, 255), with v * 255 taken in double (a
        // product rounded to T, e.g. Half, can round up across an integer and move the pixel to the next bin)
        var flat = engine.Reshape(gray, new[] { gray.Length }).Cast<double>();
        var bins = engine.TensorClamp(engine.TensorFloor(engine.TensorMultiplyScalar(flat, 255.0)), 0.0, 255.0)
            .Cast<int>();
        // Histogram counted in int (a running count in Half stops at 2048), then turned into bin probabilities
        // count / n without materialising a count in T: count = hi * 2048 + lo, p = hi * (2048 / n) + lo / n, so
        // every value cast to T is below 2048 (exact in Half) as long as n < 2048^2
        var histogram = engine.TensorBinCount(bins, 256);
        int total = gray.Length;
        var counts = histogram.ToArray();
        var high = new Tensor<int>(counts.Select(count => count / 2048).ToArray(), new[] { 256 });
        var low = new Tensor<int>(counts.Select(count => count % 2048).ToArray(), new[] { 256 });
        var probabilities = engine.TensorAdd(
            engine.TensorMultiplyScalar(high.Cast<T>(), _numOps.FromDouble(2048.0 / total)),
            engine.TensorMultiplyScalar(low.Cast<T>(), _numOps.FromDouble(1.0 / total)));

        // Otsu between-class variance for every threshold t in class probabilities: wB wF (mB - mF)^2 (the count
        // form divided by n^2, so the same maximiser), zero where a class is empty; class emptiness is decided on the
        // int counts
        var levels = Tensor<int>.FromVector(new Vector<int>(Enumerable.Range(0, 256))).Cast<T>();
        var wB = engine.TensorCumSum(probabilities, 0);
        var sumB = engine.TensorCumSum(engine.TensorMultiply(levels, probabilities), 0);
        T sum = engine.TensorSum(engine.TensorMultiply(levels, probabilities));
        var wF = engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { 256 }, _numOps.One), wB);

        var countB = engine.TensorCumSum(histogram, 0);
        var valid = engine.TensorMultiply(
            engine.TensorGreaterThan(countB, 0), engine.TensorLessThan(countB, total)).Cast<T>();
        var ones = engine.TensorOnesLike(wB);
        var mB = engine.TensorDivide(sumB, engine.TensorWhere(valid, wB, ones));
        var mF = engine.TensorDivide(
            engine.TensorSubtract(Tensor<T>.CreateDefault(new[] { 256 }, sum), sumB), engine.TensorWhere(valid, wF, ones));
        var variance = engine.TensorMultiply(
            engine.TensorMultiply(engine.TensorMultiply(wB, wF), engine.TensorSquare(engine.TensorSubtract(mB, mF))), valid);

        // First threshold with the largest variance (0 when no split exists); the engine arg max does not promise
        // the first index on ties
        int threshold = AiDotNet.AnomalyDetection.AnomalyTensorOps.FirstArgMax(variance);

        double normalizedThreshold = threshold / 255.0;
        return ApplyThreshold(gray, normalizedThreshold);
    }

    /// <summary>
    /// Applies Sauvola's local adaptive thresholding.
    /// </summary>
    /// <param name="gray">The grayscale image.</param>
    /// <param name="windowSize">Size of the local window (should be odd).</param>
    /// <param name="k">Sauvola's k parameter (typically 0.2-0.5).</param>
    /// <param name="r">Dynamic range parameter (typically 128).</param>
    public Tensor<T> SauvolaBinarization(Tensor<T> gray, int windowSize = 15, double k = 0.2, double r = 128.0)
    {
        var engine = AiDotNetEngine.Current;
        var (pixels, mean, stdDev) = LocalStatistics(gray, windowSize);

        // Sauvola threshold: T = mean * (1 + k * (stdDev / r - 1))
        var factor = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorDivideScalar(stdDev, _numOps.FromDouble(r)), _numOps.One), _numOps.FromDouble(k)),
            _numOps.One);
        var threshold = engine.TensorMultiply(mean, factor);
        return engine.TensorGreaterThan(pixels, threshold);
    }

    /// <summary>
    /// Applies Niblack's local adaptive thresholding.
    /// </summary>
    /// <param name="gray">The grayscale image.</param>
    /// <param name="windowSize">Size of the local window (should be odd).</param>
    /// <param name="k">Niblack's k parameter (typically -0.2).</param>
    public Tensor<T> NiblackBinarization(Tensor<T> gray, int windowSize = 15, double k = -0.2)
    {
        var engine = AiDotNetEngine.Current;
        var (pixels, mean, stdDev) = LocalStatistics(gray, windowSize);

        // Niblack threshold: T = mean + k * stdDev
        var threshold = engine.TensorAdd(mean, engine.TensorMultiplyScalar(stdDev, _numOps.FromDouble(k)));
        return engine.TensorGreaterThan(pixels, threshold);
    }

    /// <summary>
    /// Applies a fixed threshold.
    /// </summary>
    public Tensor<T> FixedThreshold(Tensor<T> gray, double threshold)
    {
        return ApplyThreshold(gray, threshold);
    }

    #endregion

    #region Private Methods

    private Tensor<T> ToGrayscale(Tensor<T> image)
    {
        if (image.Shape[0] == 1)
        {
            return image;
        }

        var engine = AiDotNetEngine.Current;
        int height = image.Shape[1];
        int width = image.Shape[2];
        var r = engine.TensorNarrow(image, 0, 0, 1);
        var g = image.Shape[0] > 1 ? engine.TensorNarrow(image, 0, 1, 1) : r;
        var b = image.Shape[0] > 2 ? engine.TensorNarrow(image, 0, 2, 1) : r;
        var gray = engine.TensorAdd(
            engine.TensorAdd(engine.TensorMultiplyScalar(r, _numOps.FromDouble(0.299)), engine.TensorMultiplyScalar(g, _numOps.FromDouble(0.587))),
            engine.TensorMultiplyScalar(b, _numOps.FromDouble(0.114)));
        return engine.Reshape(gray, new[] { 1, height, width });
    }

    private Tensor<T> ApplyThreshold(Tensor<T> gray, double threshold)
    {
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        var binary = AiDotNetEngine.Current.TensorGreaterThan(gray, _numOps.FromDouble(threshold));
        return AiDotNetEngine.Current.Reshape(binary, new[] { 1, height, width });
    }

    /// <summary>
    /// Pixel values scaled to [0, 255] with the mean and standard deviation (population, floored at zero variance)
    /// of the (2 (windowSize / 2) + 1)^2 window clipped to the image, all shaped [1, height, width]. The window
    /// averages are taken directly over each window (average pooling without padded cells), which is more
    /// accurate than differencing global integral images.
    /// </summary>
    private (Tensor<T> Pixels, Tensor<T> Mean, Tensor<T> StdDev) LocalStatistics(Tensor<T> gray, int windowSize)
    {
        var engine = AiDotNetEngine.Current;
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        int halfWindow = windowSize / 2;
        int pool = 2 * halfWindow + 1;

        var pixels = engine.TensorMultiplyScalar(
            engine.Reshape(engine.TensorNarrow(gray, 0, 0, 1), new[] { 1, 1, height, width }), _numOps.FromDouble(255.0));
        var mean = engine.AvgPool2D(pixels, pool, 1, halfWindow, countIncludePad: false);
        var meanSquare = engine.AvgPool2D(engine.TensorSquare(pixels), pool, 1, halfWindow, countIncludePad: false);
        var variance = engine.TensorClampMin(engine.TensorSubtract(meanSquare, engine.TensorSquare(mean)), _numOps.Zero);

        var shape = new[] { 1, height, width };
        return (engine.Reshape(pixels, shape), engine.Reshape(mean, shape), engine.Reshape(engine.TensorSqrt(variance), shape));
    }

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the binarization utility.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            // No managed resources to dispose
        }
        _disposed = true;
    }

    #endregion
}
