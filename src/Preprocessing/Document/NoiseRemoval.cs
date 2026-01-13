using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// NoiseRemoval - Document image noise removal with multiple algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// NoiseRemoval provides various filtering techniques to clean up document images
/// by reducing noise while preserving important features like text edges.
/// </para>
/// <para>
/// <b>For Beginners:</b> Scanned documents often have noise (random spots, grain).
/// This tool removes noise while keeping text clear:
///
/// - Median: Best for salt-and-pepper noise
/// - Gaussian: General smoothing
/// - Bilateral: Edge-preserving smoothing
/// - Morphological: Removes small artifacts
///
/// Key features:
/// - Multiple noise removal algorithms
/// - Edge-preserving options
/// - Configurable filter sizes
/// - Works with binary and grayscale images
///
/// Example usage:
/// <code>
/// var noiseRemoval = new NoiseRemoval&lt;float&gt;();
/// var clean = noiseRemoval.Process(noisyImage, NoiseRemovalMethod.Median);
/// </code>
/// </para>
/// </remarks>
public class NoiseRemoval<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new NoiseRemoval instance.
    /// </summary>
    public NoiseRemoval()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Applies noise removal to a document image.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="method">The noise removal method to use.</param>
    /// <returns>The denoised image.</returns>
    public Tensor<T> Process(Tensor<T> image, NoiseRemovalMethod method = NoiseRemovalMethod.Median)
    {
        return method switch
        {
            NoiseRemovalMethod.Median => MedianFilter(image),
            NoiseRemovalMethod.Gaussian => GaussianBlur(image),
            NoiseRemovalMethod.Bilateral => BilateralFilter(image),
            NoiseRemovalMethod.Morphological => MorphologicalOpening(image),
            _ => MedianFilter(image)
        };
    }

    /// <summary>
    /// Applies median filtering for salt-and-pepper noise removal.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the filter kernel (should be odd).</param>
    public Tensor<T> MedianFilter(Tensor<T> image, int kernelSize = 3)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;

        var result = new T[channels * height * width];
        var window = new List<double>(kernelSize * kernelSize);

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    window.Clear();

                    // Collect neighborhood values
                    for (int ky = -halfKernel; ky <= halfKernel; ky++)
                    {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++)
                        {
                            int ny = PreprocessingHelpers.Clamp(y + ky, 0, height - 1);
                            int nx = PreprocessingHelpers.Clamp(x + kx, 0, width - 1);
                            window.Add(_numOps.ToDouble(image[c, ny, nx]));
                        }
                    }

                    // Find median
                    window.Sort();
                    double median = window[window.Count / 2];

                    int idx = c * height * width + y * width + x;
                    result[idx] = _numOps.FromDouble(median);
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Applies Gaussian blur for general noise reduction.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the Gaussian kernel (should be odd).</param>
    /// <param name="sigma">Standard deviation of the Gaussian.</param>
    public Tensor<T> GaussianBlur(Tensor<T> image, int kernelSize = 5, double sigma = 1.0)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;

        // Generate Gaussian kernel
        var kernel = new double[kernelSize, kernelSize];
        double kernelSum = 0;

        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                double dx = i - halfKernel;
                double dy = j - halfKernel;
                kernel[i, j] = Math.Exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                kernelSum += kernel[i, j];
            }
        }

        // Normalize kernel
        for (int i = 0; i < kernelSize; i++)
            for (int j = 0; j < kernelSize; j++)
                kernel[i, j] /= kernelSum;

        // Apply convolution
        var result = new T[channels * height * width];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double sum = 0;

                    for (int ky = -halfKernel; ky <= halfKernel; ky++)
                    {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++)
                        {
                            int ny = PreprocessingHelpers.Clamp(y + ky, 0, height - 1);
                            int nx = PreprocessingHelpers.Clamp(x + kx, 0, width - 1);
                            sum += _numOps.ToDouble(image[c, ny, nx]) * kernel[ky + halfKernel, kx + halfKernel];
                        }
                    }

                    int idx = c * height * width + y * width + x;
                    result[idx] = _numOps.FromDouble(sum);
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Applies bilateral filtering for edge-preserving smoothing.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the filter kernel.</param>
    /// <param name="sigmaSpace">Spatial sigma.</param>
    /// <param name="sigmaColor">Color/intensity sigma.</param>
    public Tensor<T> BilateralFilter(Tensor<T> image, int kernelSize = 5, double sigmaSpace = 75.0, double sigmaColor = 75.0)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;

        var result = new T[channels * height * width];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double centerValue = _numOps.ToDouble(image[c, y, x]);
                    double sum = 0;
                    double weightSum = 0;

                    for (int ky = -halfKernel; ky <= halfKernel; ky++)
                    {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++)
                        {
                            int ny = PreprocessingHelpers.Clamp(y + ky, 0, height - 1);
                            int nx = PreprocessingHelpers.Clamp(x + kx, 0, width - 1);

                            double neighborValue = _numOps.ToDouble(image[c, ny, nx]);

                            // Spatial weight
                            double spatialDist = ky * ky + kx * kx;
                            double spatialWeight = Math.Exp(-spatialDist / (2 * sigmaSpace * sigmaSpace));

                            // Color/intensity weight
                            double colorDist = (centerValue - neighborValue) * (centerValue - neighborValue);
                            double colorWeight = Math.Exp(-colorDist / (2 * sigmaColor * sigmaColor));

                            double weight = spatialWeight * colorWeight;
                            sum += neighborValue * weight;
                            weightSum += weight;
                        }
                    }

                    int idx = c * height * width + y * width + x;
                    result[idx] = _numOps.FromDouble(weightSum > 0 ? sum / weightSum : centerValue);
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Applies morphological opening (erosion followed by dilation) for artifact removal.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the structuring element.</param>
    public Tensor<T> MorphologicalOpening(Tensor<T> image, int kernelSize = 3)
    {
        // Opening = erosion followed by dilation
        var eroded = Erode(image, kernelSize);
        return Dilate(eroded, kernelSize);
    }

    /// <summary>
    /// Applies morphological closing (dilation followed by erosion) for hole filling.
    /// </summary>
    /// <param name="image">The input image.</param>
    /// <param name="kernelSize">Size of the structuring element.</param>
    public Tensor<T> MorphologicalClosing(Tensor<T> image, int kernelSize = 3)
    {
        // Closing = dilation followed by erosion
        var dilated = Dilate(image, kernelSize);
        return Erode(dilated, kernelSize);
    }

    /// <summary>
    /// Applies morphological erosion.
    /// </summary>
    public Tensor<T> Erode(Tensor<T> image, int kernelSize = 3)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;

        var result = new T[channels * height * width];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double minVal = double.MaxValue;

                    for (int ky = -halfKernel; ky <= halfKernel; ky++)
                    {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++)
                        {
                            int ny = PreprocessingHelpers.Clamp(y + ky, 0, height - 1);
                            int nx = PreprocessingHelpers.Clamp(x + kx, 0, width - 1);
                            minVal = Math.Min(minVal, _numOps.ToDouble(image[c, ny, nx]));
                        }
                    }

                    int idx = c * height * width + y * width + x;
                    result[idx] = _numOps.FromDouble(minVal);
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Applies morphological dilation.
    /// </summary>
    public Tensor<T> Dilate(Tensor<T> image, int kernelSize = 3)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];
        int halfKernel = kernelSize / 2;

        var result = new T[channels * height * width];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double maxVal = double.MinValue;

                    for (int ky = -halfKernel; ky <= halfKernel; ky++)
                    {
                        for (int kx = -halfKernel; kx <= halfKernel; kx++)
                        {
                            int ny = PreprocessingHelpers.Clamp(y + ky, 0, height - 1);
                            int nx = PreprocessingHelpers.Clamp(x + kx, 0, width - 1);
                            maxVal = Math.Max(maxVal, _numOps.ToDouble(image[c, ny, nx]));
                        }
                    }

                    int idx = c * height * width + y * width + x;
                    result[idx] = _numOps.FromDouble(maxVal);
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
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
    /// Releases resources used by the noise removal utility.
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
