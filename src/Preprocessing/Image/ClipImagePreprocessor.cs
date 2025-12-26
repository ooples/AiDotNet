using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Preprocessing.Image;

/// <summary>
/// Preprocesses images for CLIP (Contrastive Language-Image Pre-training) models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// CLIP models expect images to be preprocessed in a specific way:
/// 1. Resize to a square size (typically 224x224 or 336x336)
/// 2. Normalize pixel values using ImageNet mean and standard deviation
/// 3. Convert to tensor format [channels, height, width]
/// </para>
/// <para><b>For Beginners:</b> Before CLIP can "see" an image, it needs to be prepared:
///
/// 1. <b>Resize</b>: Images come in all sizes (1000x2000, 50x50, etc.)
///    CLIP expects a specific size (like 224x224 pixels).
///
/// 2. <b>Normalize</b>: Pixel values (0-255) are scaled and shifted using
///    standard values from ImageNet dataset. This helps the model work consistently.
///
/// 3. <b>Format</b>: The image is arranged as [R, G, B] channels first,
///    then height and width. This is called "channels-first" format.
///
/// Example:
/// - Original: 1920x1080 photo with RGB values 0-255
/// - After preprocessing: 224x224 tensor with normalized values around [-2, 2]
/// </para>
/// </remarks>
public class ClipImagePreprocessor<T>
{
    /// <summary>
    /// The numeric operations helper for type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The computational engine for tensor operations.
    /// </summary>
    private readonly IEngine _engine;

    /// <summary>
    /// The target image size (height and width).
    /// </summary>
    private readonly int _imageSize;

    /// <summary>
    /// The normalization mean values for RGB channels (ImageNet standard).
    /// </summary>
    private readonly T[] _mean;

    /// <summary>
    /// The normalization standard deviation values for RGB channels (ImageNet standard).
    /// </summary>
    private readonly T[] _std;

    /// <summary>
    /// Gets the target image size.
    /// </summary>
    public int ImageSize => _imageSize;

    /// <summary>
    /// Initializes a new instance of the ClipImagePreprocessor class.
    /// </summary>
    /// <param name="imageSize">The target image size (default: 224 for CLIP ViT-B/32).</param>
    /// <param name="mean">The normalization mean values for RGB. If null, uses ImageNet mean.</param>
    /// <param name="std">The normalization std values for RGB. If null, uses ImageNet std.</param>
    /// <remarks>
    /// <para>
    /// The default normalization values are from ImageNet:
    /// - Mean: [0.48145466, 0.4578275, 0.40821073]
    /// - Std: [0.26862954, 0.26130258, 0.27577711]
    /// These are the standard values used by OpenAI's CLIP models.
    /// </para>
    /// <para><b>For Beginners:</b> You usually don't need to change the default values.
    ///
    /// The mean and std values are "magic numbers" that were calculated from
    /// millions of images in the ImageNet dataset. Using them helps CLIP
    /// process images consistently with how it was trained.
    ///
    /// Common image sizes:
    /// - 224: Standard CLIP (ViT-B/32, RN50)
    /// - 336: Higher resolution CLIP (ViT-L/14@336px)
    /// </para>
    /// </remarks>
    public ClipImagePreprocessor(
        int imageSize = 224,
        T[]? mean = null,
        T[]? std = null)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageSize), "Image size must be positive.");

        _numOps = MathHelper.GetNumericOperations<T>();
        _engine = AiDotNetEngine.Current;
        _imageSize = imageSize;

        // ImageNet normalization values (OpenAI CLIP standard)
        _mean = mean ?? new T[]
        {
            _numOps.FromDouble(0.48145466),
            _numOps.FromDouble(0.4578275),
            _numOps.FromDouble(0.40821073)
        };

        _std = std ?? new T[]
        {
            _numOps.FromDouble(0.26862954),
            _numOps.FromDouble(0.26130258),
            _numOps.FromDouble(0.27577711)
        };

        if (_mean.Length != 3 || _std.Length != 3)
            throw new ArgumentException("Mean and std must have exactly 3 values (RGB).");
    }

    /// <summary>
    /// Preprocesses an image for CLIP input.
    /// </summary>
    /// <param name="image">The input image tensor with shape [height, width, channels] or [channels, height, width].</param>
    /// <returns>A normalized tensor with shape [3, ImageSize, ImageSize].</returns>
    /// <remarks>
    /// <para>
    /// The preprocessing pipeline:
    /// 1. Detect input format (HWC or CHW)
    /// 2. Resize to target size using bilinear interpolation
    /// 3. Normalize each channel: (pixel - mean) / std
    /// 4. Convert to channels-first format if needed
    /// </para>
    /// <para><b>For Beginners:</b> This method takes any image and prepares it for CLIP.
    ///
    /// Input can be:
    /// - Any size (will be resized to 224x224 or your chosen size)
    /// - Any format (height-width-channels or channels-height-width)
    /// - Values 0-255 (standard image) or 0-1 (normalized)
    ///
    /// Output is always:
    /// - Size: [3, 224, 224] (or your chosen size)
    /// - Format: Channels-first (RGB)
    /// - Values: Normalized using ImageNet statistics
    /// </para>
    /// </remarks>
    public Tensor<T> Preprocess(Tensor<T> image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));

        // Validate and detect input format
        if (image.Shape.Length < 2 || image.Shape.Length > 4)
            throw new ArgumentException($"Image must have 2-4 dimensions, got {image.Shape.Length}.");

        // Handle different input formats
        Tensor<T> processed;
        if (image.Shape.Length == 2)
        {
            // Grayscale [H, W] -> expand to [1, H, W]
            processed = ExpandGrayscale(image);
        }
        else if (image.Shape.Length == 3)
        {
            // Could be [H, W, C] or [C, H, W]
            processed = NormalizeFormat(image);
        }
        else // 4 dimensions [N, C, H, W] or [N, H, W, C]
        {
            // Take first image from batch
            processed = ExtractFirstImage(image);
        }

        // Resize to target size
        if (processed.Shape[1] != _imageSize || processed.Shape[2] != _imageSize)
        {
            processed = Resize(processed, _imageSize, _imageSize);
        }

        // Normalize pixel values
        processed = NormalizePixels(processed);

        return processed;
    }

    /// <summary>
    /// Preprocesses a batch of images for CLIP input.
    /// </summary>
    /// <param name="images">The input images.</param>
    /// <returns>A batch of normalized tensors.</returns>
    public IEnumerable<Tensor<T>> PreprocessBatch(IEnumerable<Tensor<T>> images)
    {
        if (images == null)
            throw new ArgumentNullException(nameof(images));

        return images.Select(Preprocess);
    }

    /// <summary>
    /// Expands a grayscale image to 3 channels by repeating.
    /// </summary>
    private Tensor<T> ExpandGrayscale(Tensor<T> grayscale)
    {
        int height = grayscale.Shape[0];
        int width = grayscale.Shape[1];
        var result = new Tensor<T>(new[] { 3, height, width });

        // Repeat grayscale values across all 3 channels
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[c, h, w] = grayscale[h, w];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Normalizes the tensor format to channels-first [C, H, W].
    /// </summary>
    private Tensor<T> NormalizeFormat(Tensor<T> image)
    {
        int dim0 = image.Shape[0];
        int dim1 = image.Shape[1];
        int dim2 = image.Shape[2];

        // Heuristic: if last dimension is 3 or 4, it's probably [H, W, C]
        // Otherwise assume [C, H, W]
        if (dim2 == 3 || dim2 == 4)
        {
            // [H, W, C] -> [C, H, W]
            int height = dim0;
            int width = dim1;
            int channels = Math.Min(dim2, 3); // Take only RGB

            var result = new Tensor<T>(new[] { channels, height, width });
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[c, h, w] = image[h, w, c];
                    }
                }
            }
            return result;
        }
        else
        {
            // Already [C, H, W], ensure 3 channels
            if (dim0 == 1)
            {
                // Single channel -> expand to 3
                return ExpandSingleChannel(image);
            }
            else if (dim0 > 3)
            {
                // Too many channels -> take first 3
                return TakeFirstChannels(image, 3);
            }
            return image;
        }
    }

    /// <summary>
    /// Extracts the first image from a batch.
    /// </summary>
    /// <summary>
    /// Extracts the first image from a batch and normalizes to channels-first [C, H, W].
    /// Supports [N, C, H, W] and [N, H, W, C] batch formats.
    /// </summary>
    private Tensor<T> ExtractFirstImage(Tensor<T> batch)
    {
        if (batch.Shape.Length != 4)
        {
            throw new ArgumentException("Batch tensor must have 4 dimensions.", nameof(batch));
        }

        int dim0 = batch.Shape[0];
        int dim1 = batch.Shape[1];
        int dim2 = batch.Shape[2];
        int dim3 = batch.Shape[3];

        if (dim0 < 1)
        {
            throw new ArgumentException("Batch dimension N must be at least 1.", nameof(batch));
        }

        Tensor<T> firstImage3D;

        // Heuristic: if last dimension is 3 or 4, treat as [N, H, W, C], otherwise [N, C, H, W]
        if (dim3 == 3 || dim3 == 4)
        {
            // [N, H, W, C] -> [H, W, C] for first image
            int height = dim1;
            int width = dim2;
            int channels = dim3;

            var hwc = new Tensor<T>(new[] { height, width, channels });
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < channels; c++)
                    {
                        hwc[h, w, c] = batch[0, h, w, c];
                    }
                }
            }

            // Normalize to [C, H, W] via NormalizeFormat
            firstImage3D = NormalizeFormat(hwc);
        }
        else
        {
            // [N, C, H, W] -> [C, H, W] for first image
            int channels = dim1;
            int height = dim2;
            int width = dim3;

            var chw = new Tensor<T>(new[] { channels, height, width });
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        chw[c, h, w] = batch[0, c, h, w];
                    }
                }
            }

            // Already in [C, H, W] format, but run through NormalizeFormat for RGB channel handling
            firstImage3D = NormalizeFormat(chw);
        }

        return firstImage3D;
    }

    /// <summary>
    /// Expands a single-channel image to 3 channels.
    /// </summary>
    private Tensor<T> ExpandSingleChannel(Tensor<T> image)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];
        var result = new Tensor<T>(new[] { 3, height, width });

        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[c, h, w] = image[0, h, w];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Takes only the first N channels from an image.
    /// </summary>
    private Tensor<T> TakeFirstChannels(Tensor<T> image, int numChannels)
    {
        int height = image.Shape[1];
        int width = image.Shape[2];
        var result = new Tensor<T>(new[] { numChannels, height, width });

        for (int c = 0; c < numChannels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    result[c, h, w] = image[c, h, w];
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Resizes an image using bilinear interpolation.
    /// </summary>
    private Tensor<T> Resize(Tensor<T> image, int targetHeight, int targetWidth)
    {
        int channels = image.Shape[0];
        int srcHeight = image.Shape[1];
        int srcWidth = image.Shape[2];

        var result = new Tensor<T>(new[] { channels, targetHeight, targetWidth });

        T scaleH = _numOps.Divide(_numOps.FromDouble(srcHeight), _numOps.FromDouble(targetHeight));
        T scaleW = _numOps.Divide(_numOps.FromDouble(srcWidth), _numOps.FromDouble(targetWidth));

        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < targetHeight; h++)
            {
                for (int w = 0; w < targetWidth; w++)
                {
                    // Calculate source coordinates
                    T srcH = _numOps.Multiply(_numOps.FromDouble(h + 0.5), scaleH);
                    T srcW = _numOps.Multiply(_numOps.FromDouble(w + 0.5), scaleW);
                    srcH = _numOps.Subtract(srcH, _numOps.FromDouble(0.5));
                    srcW = _numOps.Subtract(srcW, _numOps.FromDouble(0.5));

                    // Bilinear interpolation
                    result[c, h, w] = BilinearInterpolate(image, c, srcH, srcW, srcHeight, srcWidth);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Performs bilinear interpolation at a fractional position.
    /// </summary>
    private T BilinearInterpolate(Tensor<T> image, int channel, T y, T x, int maxH, int maxW)
    {
        // Use explicit floor semantics on the double representation of y/x
        double yDouble = _numOps.ToDouble(y);
        double xDouble = _numOps.ToDouble(x);

        // Cast to int performs truncation (floor for positive values)
        int y0 = Math.Max(0, Math.Min((int)Math.Floor(yDouble), maxH - 1));
        int y1 = Math.Max(0, Math.Min(y0 + 1, maxH - 1));
        int x0 = Math.Max(0, Math.Min((int)Math.Floor(xDouble), maxW - 1));
        int x1 = Math.Max(0, Math.Min(x0 + 1, maxW - 1));

        // y0/x0 are floor(y)/floor(x), so subtracting them yields the fractional parts
        T fracY = _numOps.Subtract(y, _numOps.FromDouble(y0));
        T fracX = _numOps.Subtract(x, _numOps.FromDouble(x0));

        T v00 = image[channel, y0, x0];
        T v01 = image[channel, y0, x1];
        T v10 = image[channel, y1, x0];
        T v11 = image[channel, y1, x1];

        // Interpolate along x for both y values
        T oneMinusFracX = _numOps.Subtract(_numOps.One, fracX);
        T top = _numOps.Add(
            _numOps.Multiply(v00, oneMinusFracX),
            _numOps.Multiply(v01, fracX));
        T bottom = _numOps.Add(
            _numOps.Multiply(v10, oneMinusFracX),
            _numOps.Multiply(v11, fracX));

        // Interpolate along y
        T oneMinusFracY = _numOps.Subtract(_numOps.One, fracY);
        return _numOps.Add(
            _numOps.Multiply(top, oneMinusFracY),
            _numOps.Multiply(bottom, fracY));
    }

    /// <summary>
    /// Normalizes pixel values using ImageNet statistics.
    /// </summary>
    private Tensor<T> NormalizePixels(Tensor<T> image)
    {
        int channels = image.Shape[0];
        int height = image.Shape[1];
        int width = image.Shape[2];

        var result = new Tensor<T>(new[] { channels, height, width });

        // Detect if values are 0-255 or 0-1
        T maxVal = FindMax(image);
        bool needsScaling = _numOps.ToDouble(maxVal) > 1.0;

        T scale = needsScaling ? _numOps.FromDouble(255.0) : _numOps.One;

        for (int c = 0; c < channels; c++)
        {
            T mean = _mean[c];
            T std = _std[c];

            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    T pixel = image[c, h, w];

                    // Scale to 0-1 if needed
                    if (needsScaling)
                    {
                        pixel = _numOps.Divide(pixel, scale);
                    }

                    // Normalize: (pixel - mean) / std
                    result[c, h, w] = _numOps.Divide(
                        _numOps.Subtract(pixel, mean),
                        std);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Finds the maximum value in a tensor.
    /// </summary>
    private T FindMax(Tensor<T> tensor)
    {
        T max = tensor.Data[0];
        for (int i = 1; i < tensor.Data.Length; i++)
        {
            if (_numOps.ToDouble(tensor.Data[i]) > _numOps.ToDouble(max))
            {
                max = tensor.Data[i];
            }
        }
        return max;
    }
}
