using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// DocumentPreprocessor - Comprehensive document image preprocessing pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DocumentPreprocessor provides a unified interface for applying multiple
/// preprocessing operations to document images before feeding them to AI models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Document preprocessing improves model accuracy by:
///
/// - Normalizing image characteristics
/// - Removing noise and artifacts
/// - Correcting geometric distortions
/// - Enhancing text contrast
///
/// Key features:
/// - Chained preprocessing pipeline
/// - Configurable operation order
/// - Quality-aware preprocessing
/// - Batch processing support
///
/// Example usage:
/// <code>
/// var preprocessor = new DocumentPreprocessor&lt;float&gt;();
/// var processed = preprocessor.Preprocess(documentImage, options);
/// </code>
/// </para>
/// </remarks>
public class DocumentPreprocessor<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly Deskew<T> _deskew;
    private readonly Binarization<T> _binarization;
    private readonly NoiseRemoval<T> _noiseRemoval;
    private readonly LayoutNormalization<T> _layoutNormalization;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DocumentPreprocessor with default settings.
    /// </summary>
    public DocumentPreprocessor()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _deskew = new Deskew<T>();
        _binarization = new Binarization<T>();
        _noiseRemoval = new NoiseRemoval<T>();
        _layoutNormalization = new LayoutNormalization<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Applies the full preprocessing pipeline to a document image.
    /// </summary>
    public Tensor<T> Preprocess(Tensor<T> image, DocumentPreprocessingOptions? options = null)
    {
        options ??= new DocumentPreprocessingOptions();
        var result = image;

        // Apply operations in order
        if (options.ApplyDeskew)
            result = _deskew.Process(result, options.DeskewMaxAngle);

        if (options.ApplyBinarization)
            result = _binarization.Process(result, options.BinarizationMethod);

        if (options.ApplyNoiseRemoval)
            result = _noiseRemoval.Process(result, options.NoiseRemovalMethod);

        if (options.ApplyLayoutNormalization)
            result = _layoutNormalization.Process(result, options.TargetWidth, options.TargetHeight);

        if (options.NormalizeIntensity)
            result = NormalizeIntensity(result);

        return result;
    }

    /// <summary>
    /// Applies preprocessing to multiple document images.
    /// </summary>
    public IList<Tensor<T>> PreprocessBatch(IEnumerable<Tensor<T>> images, DocumentPreprocessingOptions? options = null)
    {
        return images.Select(img => Preprocess(img, options)).ToList();
    }

    /// <summary>
    /// Converts an RGB image to grayscale.
    /// </summary>
    public Tensor<T> ToGrayscale(Tensor<T> image)
    {
        if (image.Shape.Length < 3 || image.Shape[0] == 1)
            return image; // Already grayscale

        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        var result = new T[height * width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Luminance formula: 0.299*R + 0.587*G + 0.114*B
                double r = _numOps.ToDouble(image[0, y, x]);
                double g = channels > 1 ? _numOps.ToDouble(image[1, y, x]) : r;
                double b = channels > 2 ? _numOps.ToDouble(image[2, y, x]) : r;

                double gray = 0.299 * r + 0.587 * g + 0.114 * b;
                result[y * width + x] = _numOps.FromDouble(gray);
            }
        }

        return new Tensor<T>([1, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Resizes an image to the specified dimensions.
    /// </summary>
    public Tensor<T> Resize(Tensor<T> image, int targetWidth, int targetHeight, InterpolationMethod method = InterpolationMethod.Bilinear)
    {
        int channels = image.Shape[0];
        int srcHeight = image.Shape[1];
        int srcWidth = image.Shape[2];

        var result = new T[channels * targetHeight * targetWidth];

        double scaleX = (double)srcWidth / targetWidth;
        double scaleY = (double)srcHeight / targetHeight;

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    double srcX = x * scaleX;
                    double srcY = y * scaleY;

                    T value = method switch
                    {
                        InterpolationMethod.Nearest => SampleNearest(image, c, srcX, srcY),
                        InterpolationMethod.Bilinear => SampleBilinear(image, c, srcX, srcY),
                        _ => SampleBilinear(image, c, srcX, srcY)
                    };

                    int idx = c * targetHeight * targetWidth + y * targetWidth + x;
                    result[idx] = value;
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    /// <summary>
    /// Pads an image to the specified dimensions.
    /// </summary>
    public Tensor<T> Pad(Tensor<T> image, int targetWidth, int targetHeight, T paddingValue)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        int padLeft = (targetWidth - width) / 2;
        int padTop = (targetHeight - height) / 2;

        var result = new T[channels * targetHeight * targetWidth];
        PreprocessingHelpers.Fill(result, paddingValue);

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int newY = y + padTop;
                    int newX = x + padLeft;

                    if (newY >= 0 && newY < targetHeight && newX >= 0 && newX < targetWidth)
                    {
                        int idx = c * targetHeight * targetWidth + newY * targetWidth + newX;
                        result[idx] = image[c, y, x];
                    }
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    /// <summary>
    /// Center crops an image to the specified dimensions.
    /// </summary>
    public Tensor<T> CenterCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        int cropLeft = (width - targetWidth) / 2;
        int cropTop = (height - targetHeight) / 2;

        var result = new T[channels * targetHeight * targetWidth];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    int srcY = y + cropTop;
                    int srcX = x + cropLeft;

                    if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                    {
                        int idx = c * targetHeight * targetWidth + y * targetWidth + x;
                        result[idx] = image[c, srcY, srcX];
                    }
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    #endregion

    #region Private Methods

    private Tensor<T> NormalizeIntensity(Tensor<T> image)
    {
        // Find min and max
        double min = double.MaxValue;
        double max = double.MinValue;

        for (int i = 0; i < image.Length; i++)
        {
            double val = _numOps.ToDouble(image.Data.Span[i]);
            if (val < min) min = val;
            if (val > max) max = val;
        }

        // Normalize to 0-1 range
        double range = max - min;
        if (range < 1e-10) range = 1.0;

        var result = new T[image.Length];
        for (int i = 0; i < image.Length; i++)
        {
            double val = _numOps.ToDouble(image.Data.Span[i]);
            result[i] = _numOps.FromDouble((val - min) / range);
        }

        return new Tensor<T>(image.Shape, new Vector<T>(result));
    }

    private T SampleNearest(Tensor<T> image, int channel, double x, double y)
    {
        int ix = PreprocessingHelpers.Clamp((int)Math.Round(x), 0, image.Shape[2] - 1);
        int iy = PreprocessingHelpers.Clamp((int)Math.Round(y), 0, image.Shape[1] - 1);
        return image[channel, iy, ix];
    }

    private T SampleBilinear(Tensor<T> image, int channel, double x, double y)
    {
        int x0 = PreprocessingHelpers.Clamp((int)Math.Floor(x), 0, image.Shape[2] - 1);
        int x1 = PreprocessingHelpers.Clamp(x0 + 1, 0, image.Shape[2] - 1);
        int y0 = PreprocessingHelpers.Clamp((int)Math.Floor(y), 0, image.Shape[1] - 1);
        int y1 = PreprocessingHelpers.Clamp(y0 + 1, 0, image.Shape[1] - 1);

        double dx = x - x0;
        double dy = y - y0;

        double v00 = _numOps.ToDouble(image[channel, y0, x0]);
        double v01 = _numOps.ToDouble(image[channel, y0, x1]);
        double v10 = _numOps.ToDouble(image[channel, y1, x0]);
        double v11 = _numOps.ToDouble(image[channel, y1, x1]);

        double top = v00 * (1 - dx) + v01 * dx;
        double bottom = v10 * (1 - dx) + v11 * dx;
        double result = top * (1 - dy) + bottom * dy;

        return _numOps.FromDouble(result);
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
    /// Releases resources used by the preprocessor.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _deskew.Dispose();
            _binarization.Dispose();
            _noiseRemoval.Dispose();
            _layoutNormalization.Dispose();
        }
        _disposed = true;
    }

    #endregion
}
