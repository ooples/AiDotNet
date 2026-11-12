using AiDotNet.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Shared helper methods for vision benchmark data loaders.
/// </summary>
internal static class VisionLoaderHelper
{
    /// <summary>
    /// Loads an image file, decodes it, resizes to target dimensions via bilinear interpolation,
    /// and returns the pixel data as a flat array in HWC (height, width, channels) order.
    /// </summary>
    /// <typeparam name="T">Numeric type.</typeparam>
    /// <param name="filePath">Path to the image file (JPEG, PNG, BMP, PPM, etc.).</param>
    /// <param name="targetHeight">Target height in pixels.</param>
    /// <param name="targetWidth">Target width in pixels.</param>
    /// <param name="channels">Target number of channels (1 for grayscale, 3 for RGB).</param>
    /// <param name="normalize">Whether to normalize to [0,1]. Images from ImageHelper are already normalized if true.</param>
    /// <returns>Flat pixel array of length targetHeight * targetWidth * channels in HWC order.</returns>
    public static T[] LoadAndResizeImage<T>(string filePath, int targetHeight, int targetWidth, int channels = 3, bool normalize = true)
    {
        if (targetHeight <= 0) throw new ArgumentOutOfRangeException(nameof(targetHeight), "Must be positive.");
        if (targetWidth <= 0) throw new ArgumentOutOfRangeException(nameof(targetWidth), "Must be positive.");
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels), "Must be positive.");
        // Guard against integer overflow in pixel allocation
        long totalPixelsLong = (long)targetHeight * targetWidth * channels;
        if (totalPixelsLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(targetHeight), $"Image dimensions too large: {targetHeight}x{targetWidth}x{channels} exceeds max array size.");

        // Use ImageHelper for proper format decoding (JPEG/PNG/BMP via ImageSharp on .NET 6+, BMP/PPM/PGM natively)
        // Returns [1, C, H, W] in CHW format, already normalized if normalize=true
        Tensor<T> imageTensor;
        try
        {
            imageTensor = ImageHelper<T>.LoadImage(filePath, normalize);
        }
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException or InvalidDataException or IOException or ArgumentException)
        {
            // Fallback for unsupported formats, corrupt files, or non-image data:
            // Read raw bytes and interpret as pixel data
            return LoadRawFallback<T>(filePath, targetHeight, targetWidth, channels, normalize);
        }

        // Bilinear resize (align_corners) from CHW source to HWC target; extra target channels repeat the source's.
        return ResizeChwToHwc(imageTensor, targetHeight, targetWidth, channels, ExtraChannelFill.RepeatSource).ToArray();
    }

    /// <summary>
    /// How target channels beyond the source's channel count are filled.
    /// </summary>
    internal enum ExtraChannelFill
    {
        /// <summary>Target channel c reads source channel c mod C (grayscale is replicated).</summary>
        RepeatSource,

        /// <summary>A grayscale source is replicated; otherwise channels beyond the source's are zero.</summary>
        Zero
    }

    /// <summary>
    /// Resizes a decoded [1, C, H, W] image to [1, targetHeight, targetWidth, targetChannels] (HWC) with
    /// bilinear interpolation (align_corners = true, torch.nn.functional.interpolate's convention).
    /// </summary>
    /// <remarks>
    /// A single target channel from a colour source is the ITU-R BT.601 luma 0.299 R + 0.587 G + 0.114 B,
    /// as PIL's convert("L") and torchvision's Grayscale compute it.
    /// </remarks>
    internal static Tensor<T> ResizeChwToHwc<T>(Tensor<T> image, int targetHeight, int targetWidth, int targetChannels,
        ExtraChannelFill extraFill)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int srcChannels = image.Shape[1];
        int srcHeight = image.Shape[2];
        int srcWidth = image.Shape[3];

        Tensor<T> selected;
        if (targetChannels == 1 && srcChannels >= 3)
        {
            var red = engine.TensorIndexSelect(image, new Tensor<int>(new[] { 0 }, new[] { 1 }), 1);
            var green = engine.TensorIndexSelect(image, new Tensor<int>(new[] { 1 }, new[] { 1 }), 1);
            var blue = engine.TensorIndexSelect(image, new Tensor<int>(new[] { 2 }, new[] { 1 }), 1);
            selected = engine.TensorAdd(
                engine.TensorAdd(
                    engine.TensorMultiplyScalar(red, numOps.FromDouble(0.299)),
                    engine.TensorMultiplyScalar(green, numOps.FromDouble(0.587))),
                engine.TensorMultiplyScalar(blue, numOps.FromDouble(0.114)));
        }
        else if (extraFill == ExtraChannelFill.Zero && srcChannels > 1 && targetChannels > srcChannels)
        {
            var zeros = new Tensor<T>(new[] { 1, targetChannels - srcChannels, srcHeight, srcWidth });
            selected = engine.TensorConcatenate(new[] { image, zeros }, 1);
        }
        else
        {
            int[] channelIndex = Enumerable.Range(0, targetChannels).Select(c => c % srcChannels).ToArray();
            selected = engine.TensorIndexSelect(image, new Tensor<int>(channelIndex, new[] { targetChannels }), 1);
        }

        var resized = engine.Interpolate(selected, new[] { targetHeight, targetWidth },
            AiDotNet.Tensors.Engines.InterpolateMode.Bilinear, true);
        return engine.TensorPermute(resized, new[] { 0, 2, 3, 1 });
    }

    /// <summary>
    /// Fallback for platforms where ImageHelper doesn't support the format.
    /// Reads raw bytes and interprets them as grayscale pixel data.
    /// </summary>
    private static T[] LoadRawFallback<T>(string filePath, int targetHeight, int targetWidth, int channels, bool normalize)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        byte[] fileBytes = File.ReadAllBytes(filePath);

        int totalPixels = targetHeight * targetWidth * channels;
        var pixels = new T[totalPixels];

        // Estimate source dimensions from byte count (assume square, 3 channels)
        int srcTotal = fileBytes.Length;
        int srcChannels = Math.Min(3, channels);
        int srcPixels = srcTotal / Math.Max(1, srcChannels);
        int srcSide = (int)Math.Sqrt(srcPixels);
        if (srcSide < 1) srcSide = 1;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                int srcY = srcSide > 1 ? y * (srcSide - 1) / Math.Max(1, targetHeight - 1) : 0;
                int srcX = srcSide > 1 ? x * (srcSide - 1) / Math.Max(1, targetWidth - 1) : 0;

                for (int c = 0; c < channels; c++)
                {
                    int srcIdx = (srcY * srcSide + srcX) * srcChannels + Math.Min(c, srcChannels - 1);
                    double value = srcIdx < fileBytes.Length ? fileBytes[srcIdx] : 0;
                    if (normalize) value /= 255.0;
                    pixels[(y * targetWidth + x) * channels + c] = numOps.FromDouble(value);
                }
            }
        }

        return pixels;
    }
}
