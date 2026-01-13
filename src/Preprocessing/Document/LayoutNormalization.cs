using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.Document;

/// <summary>
/// LayoutNormalization - Document layout normalization utilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LayoutNormalization provides utilities for normalizing document layouts
/// to standard sizes and aspect ratios for consistent model input.
/// </para>
/// <para>
/// <b>For Beginners:</b> Different documents have different sizes and proportions.
/// This tool standardizes them for AI models:
///
/// - Resize to target dimensions
/// - Preserve aspect ratio options
/// - Handle padding and cropping
/// - Normalize orientation
///
/// Key features:
/// - Multiple normalization strategies
/// - Aspect ratio preservation
/// - Smart padding and cropping
/// - Batch processing support
///
/// Example usage:
/// <code>
/// var normalizer = new LayoutNormalization&lt;float&gt;();
/// var normalized = normalizer.Process(document, targetWidth: 224, targetHeight: 224);
/// </code>
/// </para>
/// </remarks>
public class LayoutNormalization<T> : IDisposable
{
    #region Fields

    private readonly INumericOperations<T> _numOps;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new LayoutNormalization instance.
    /// </summary>
    public LayoutNormalization()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Normalizes a document image to the specified dimensions.
    /// </summary>
    /// <param name="image">The input image tensor.</param>
    /// <param name="targetWidth">Target width.</param>
    /// <param name="targetHeight">Target height.</param>
    /// <param name="strategy">The normalization strategy to use.</param>
    /// <returns>The normalized image.</returns>
    public Tensor<T> Process(Tensor<T> image, int targetWidth = 224, int targetHeight = 224,
        NormalizationStrategy strategy = NormalizationStrategy.ResizeWithPadding)
    {
        return strategy switch
        {
            NormalizationStrategy.Stretch => Stretch(image, targetWidth, targetHeight),
            NormalizationStrategy.ResizeWithPadding => ResizeWithPadding(image, targetWidth, targetHeight),
            NormalizationStrategy.CenterCrop => CenterCrop(image, targetWidth, targetHeight),
            NormalizationStrategy.ResizeAndCrop => ResizeAndCrop(image, targetWidth, targetHeight),
            _ => ResizeWithPadding(image, targetWidth, targetHeight)
        };
    }

    /// <summary>
    /// Stretches the image to target dimensions (may distort aspect ratio).
    /// </summary>
    public Tensor<T> Stretch(Tensor<T> image, int targetWidth, int targetHeight)
    {
        return Resize(image, targetWidth, targetHeight);
    }

    /// <summary>
    /// Resizes the image preserving aspect ratio and adds padding.
    /// </summary>
    public Tensor<T> ResizeWithPadding(Tensor<T> image, int targetWidth, int targetHeight,
        T? paddingValue = default)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // Calculate scale to fit within target while preserving aspect ratio
        double scaleX = (double)targetWidth / width;
        double scaleY = (double)targetHeight / height;
        double scale = Math.Min(scaleX, scaleY);

        int newWidth = (int)(width * scale);
        int newHeight = (int)(height * scale);

        // Resize image
        var resized = Resize(image, newWidth, newHeight);

        // Create padded output
        T padValue = paddingValue ?? _numOps.FromDouble(1.0); // White padding for documents
        var result = new T[channels * targetHeight * targetWidth];
        PreprocessingHelpers.Fill(result, padValue);

        // Calculate padding offsets (center the image)
        int padLeft = (targetWidth - newWidth) / 2;
        int padTop = (targetHeight - newHeight) / 2;

        // Copy resized image to center
        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int dstIdx = c * targetHeight * targetWidth + (y + padTop) * targetWidth + (x + padLeft);
                    result[dstIdx] = resized[c, y, x];
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    /// <summary>
    /// Crops the center of the image to target dimensions.
    /// </summary>
    public Tensor<T> CenterCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // If image is smaller, pad first
        if (width < targetWidth || height < targetHeight)
        {
            int padWidth = Math.Max(width, targetWidth);
            int padHeight = Math.Max(height, targetHeight);
            image = Pad(image, padWidth, padHeight, _numOps.FromDouble(1.0));
            height = padHeight;
            width = padWidth;
        }

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
                    int dstIdx = c * targetHeight * targetWidth + y * targetWidth + x;
                    result[dstIdx] = image[c, srcY, srcX];
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    /// <summary>
    /// Resizes to cover target dimensions then crops to exact size.
    /// </summary>
    public Tensor<T> ResizeAndCrop(Tensor<T> image, int targetWidth, int targetHeight)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // Calculate scale to cover target while preserving aspect ratio
        double scaleX = (double)targetWidth / width;
        double scaleY = (double)targetHeight / height;
        double scale = Math.Max(scaleX, scaleY);

        int newWidth = (int)(width * scale);
        int newHeight = (int)(height * scale);

        // Resize image
        var resized = Resize(image, newWidth, newHeight);

        // Center crop to target size
        return CenterCrop(resized, targetWidth, targetHeight);
    }

    /// <summary>
    /// Detects and corrects document orientation (portrait vs landscape).
    /// </summary>
    public Tensor<T> NormalizeOrientation(Tensor<T> image, bool preferPortrait = true)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];

        bool isPortrait = height > width;

        // Rotate if orientation doesn't match preference
        if (preferPortrait && !isPortrait)
            return Rotate90(image, clockwise: true);
        else if (!preferPortrait && isPortrait)
            return Rotate90(image, clockwise: true);

        return image;
    }

    /// <summary>
    /// Rotates the image 90 degrees.
    /// </summary>
    public Tensor<T> Rotate90(Tensor<T> image, bool clockwise = true)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        // New dimensions after rotation
        int newHeight = width;
        int newWidth = height;

        var result = new T[channels * newHeight * newWidth];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    int srcX, srcY;
                    if (clockwise)
                    {
                        srcX = y;
                        srcY = newWidth - 1 - x;
                    }
                    else
                    {
                        srcX = newHeight - 1 - y;
                        srcY = x;
                    }

                    int dstIdx = c * newHeight * newWidth + y * newWidth + x;
                    result[dstIdx] = image[c, srcY, srcX];
                }
            }
        }

        return new Tensor<T>([channels, newHeight, newWidth], new Vector<T>(result));
    }

    /// <summary>
    /// Flips the image horizontally or vertically.
    /// </summary>
    public Tensor<T> Flip(Tensor<T> image, bool horizontal = true)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        var result = new T[channels * height * width];

        for (int c = 0; c < channels; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcY = horizontal ? y : height - 1 - y;
                    int srcX = horizontal ? width - 1 - x : x;

                    int dstIdx = c * height * width + y * width + x;
                    result[dstIdx] = image[c, srcY, srcX];
                }
            }
        }

        return new Tensor<T>([channels, height, width], new Vector<T>(result));
    }

    /// <summary>
    /// Computes aspect ratio of the image.
    /// </summary>
    public double GetAspectRatio(Tensor<T> image)
    {
        return (double)image.Shape[2] / image.Shape[1]; // width / height
    }

    #endregion

    #region Private Methods

    private Tensor<T> Resize(Tensor<T> image, int targetWidth, int targetHeight)
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

                    // Bilinear interpolation
                    int x0 = PreprocessingHelpers.Clamp((int)Math.Floor(srcX), 0, srcWidth - 1);
                    int x1 = PreprocessingHelpers.Clamp(x0 + 1, 0, srcWidth - 1);
                    int y0 = PreprocessingHelpers.Clamp((int)Math.Floor(srcY), 0, srcHeight - 1);
                    int y1 = PreprocessingHelpers.Clamp(y0 + 1, 0, srcHeight - 1);

                    double dx = srcX - x0;
                    double dy = srcY - y0;

                    double v00 = _numOps.ToDouble(image[c, y0, x0]);
                    double v01 = _numOps.ToDouble(image[c, y0, x1]);
                    double v10 = _numOps.ToDouble(image[c, y1, x0]);
                    double v11 = _numOps.ToDouble(image[c, y1, x1]);

                    double top = v00 * (1 - dx) + v01 * dx;
                    double bottom = v10 * (1 - dx) + v11 * dx;
                    double value = top * (1 - dy) + bottom * dy;

                    int idx = c * targetHeight * targetWidth + y * targetWidth + x;
                    result[idx] = _numOps.FromDouble(value);
                }
            }
        }

        return new Tensor<T>([channels, targetHeight, targetWidth], new Vector<T>(result));
    }

    private Tensor<T> Pad(Tensor<T> image, int targetWidth, int targetHeight, T paddingValue)
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

    #endregion

    #region Disposal

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources used by the layout normalization utility.
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
