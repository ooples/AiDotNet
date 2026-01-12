using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

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
        int height = gray.Shape[1];
        int width = gray.Shape[2];

        // Compute histogram
        var histogram = new int[256];
        for (int i = 0; i < gray.Length; i++)
        {
            int bin = PreprocessingHelpers.Clamp((int)(_numOps.ToDouble(gray.Data[i]) * 255), 0, 255);
            histogram[bin]++;
        }

        // Find optimal threshold using Otsu's method
        int total = gray.Length;
        double sum = 0;
        for (int i = 0; i < 256; i++)
            sum += i * histogram[i];

        double sumB = 0;
        int wB = 0;
        double maxVariance = 0;
        int threshold = 0;

        for (int t = 0; t < 256; t++)
        {
            wB += histogram[t];
            if (wB == 0) continue;

            int wF = total - wB;
            if (wF == 0) break;

            sumB += t * histogram[t];

            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;

            double variance = wB * wF * (mB - mF) * (mB - mF);

            if (variance > maxVariance)
            {
                maxVariance = variance;
                threshold = t;
            }
        }

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
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        int halfWindow = windowSize / 2;

        var result = new T[height * width];

        // Compute integral images for efficient local mean and variance
        var integralSum = ComputeIntegralImage(gray);
        var integralSqSum = ComputeIntegralSqImage(gray);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int x1 = Math.Max(0, x - halfWindow);
                int y1 = Math.Max(0, y - halfWindow);
                int x2 = Math.Min(width - 1, x + halfWindow);
                int y2 = Math.Min(height - 1, y + halfWindow);

                int area = (x2 - x1 + 1) * (y2 - y1 + 1);

                double sum = GetIntegralSum(integralSum, x1, y1, x2, y2, width);
                double sqSum = GetIntegralSum(integralSqSum, x1, y1, x2, y2, width);

                double mean = sum / area;
                double variance = sqSum / area - mean * mean;
                double stdDev = Math.Sqrt(Math.Max(0, variance));

                // Sauvola threshold: T = mean * (1 + k * (stdDev / r - 1))
                double threshold = mean * (1.0 + k * (stdDev / r - 1.0));

                double pixelValue = _numOps.ToDouble(gray[0, y, x]) * 255.0;
                result[y * width + x] = pixelValue > threshold ? _numOps.FromDouble(1.0) : _numOps.FromDouble(0.0);
            }
        }

        return new Tensor<T>([1, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Applies Niblack's local adaptive thresholding.
    /// </summary>
    /// <param name="gray">The grayscale image.</param>
    /// <param name="windowSize">Size of the local window (should be odd).</param>
    /// <param name="k">Niblack's k parameter (typically -0.2).</param>
    public Tensor<T> NiblackBinarization(Tensor<T> gray, int windowSize = 15, double k = -0.2)
    {
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        int halfWindow = windowSize / 2;

        var result = new T[height * width];

        var integralSum = ComputeIntegralImage(gray);
        var integralSqSum = ComputeIntegralSqImage(gray);

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int x1 = Math.Max(0, x - halfWindow);
                int y1 = Math.Max(0, y - halfWindow);
                int x2 = Math.Min(width - 1, x + halfWindow);
                int y2 = Math.Min(height - 1, y + halfWindow);

                int area = (x2 - x1 + 1) * (y2 - y1 + 1);

                double sum = GetIntegralSum(integralSum, x1, y1, x2, y2, width);
                double sqSum = GetIntegralSum(integralSqSum, x1, y1, x2, y2, width);

                double mean = sum / area;
                double variance = sqSum / area - mean * mean;
                double stdDev = Math.Sqrt(Math.Max(0, variance));

                // Niblack threshold: T = mean + k * stdDev
                double threshold = mean + k * stdDev;

                double pixelValue = _numOps.ToDouble(gray[0, y, x]) * 255.0;
                result[y * width + x] = pixelValue > threshold ? _numOps.FromDouble(1.0) : _numOps.FromDouble(0.0);
            }
        }

        return new Tensor<T>([1, height, width], new Vector<T>(result));
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
            return image;

        int height = image.Shape[1];
        int width = image.Shape[2];
        var result = new T[height * width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double r = _numOps.ToDouble(image[0, y, x]);
                double g = image.Shape[0] > 1 ? _numOps.ToDouble(image[1, y, x]) : r;
                double b = image.Shape[0] > 2 ? _numOps.ToDouble(image[2, y, x]) : r;

                result[y * width + x] = _numOps.FromDouble(0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        return new Tensor<T>([1, height, width], new Vector<T>(result));
    }

    private Tensor<T> ApplyThreshold(Tensor<T> gray, double threshold)
    {
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        var result = new T[height * width];

        for (int i = 0; i < gray.Length; i++)
        {
            double value = _numOps.ToDouble(gray.Data[i]);
            result[i] = value > threshold ? _numOps.FromDouble(1.0) : _numOps.FromDouble(0.0);
        }

        return new Tensor<T>([1, height, width], new Vector<T>(result));
    }

    private double[] ComputeIntegralImage(Tensor<T> gray)
    {
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        var integral = new double[(height + 1) * (width + 1)];

        for (int y = 1; y <= height; y++)
        {
            for (int x = 1; x <= width; x++)
            {
                double value = _numOps.ToDouble(gray[0, y - 1, x - 1]) * 255.0;
                int idx = y * (width + 1) + x;
                integral[idx] = value
                    + integral[(y - 1) * (width + 1) + x]
                    + integral[y * (width + 1) + (x - 1)]
                    - integral[(y - 1) * (width + 1) + (x - 1)];
            }
        }

        return integral;
    }

    private double[] ComputeIntegralSqImage(Tensor<T> gray)
    {
        int height = gray.Shape[1];
        int width = gray.Shape[2];
        var integral = new double[(height + 1) * (width + 1)];

        for (int y = 1; y <= height; y++)
        {
            for (int x = 1; x <= width; x++)
            {
                double value = _numOps.ToDouble(gray[0, y - 1, x - 1]) * 255.0;
                int idx = y * (width + 1) + x;
                integral[idx] = value * value
                    + integral[(y - 1) * (width + 1) + x]
                    + integral[y * (width + 1) + (x - 1)]
                    - integral[(y - 1) * (width + 1) + (x - 1)];
            }
        }

        return integral;
    }

    private double GetIntegralSum(double[] integral, int x1, int y1, int x2, int y2, int width)
    {
        int w = width + 1;
        return integral[(y2 + 1) * w + (x2 + 1)]
            - integral[y1 * w + (x2 + 1)]
            - integral[(y2 + 1) * w + x1]
            + integral[y1 * w + x1];
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
