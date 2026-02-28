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
        var numOps = MathHelper.GetNumericOperations<T>();

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

        int srcChannels = imageTensor.Shape[1];
        int srcHeight = imageTensor.Shape[2];
        int srcWidth = imageTensor.Shape[3];
        var srcSpan = imageTensor.AsSpan();

        int totalPixels = targetHeight * targetWidth * channels;
        var pixels = new T[totalPixels];

        // Bilinear resize from CHW source to HWC target
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                double srcY = srcHeight > 1 ? (double)y * (srcHeight - 1) / (targetHeight - 1) : 0;
                double srcX = srcWidth > 1 ? (double)x * (srcWidth - 1) / (targetWidth - 1) : 0;

                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);

                double dy = srcY - y0;
                double dx = srcX - x0;

                for (int c = 0; c < channels; c++)
                {
                    double value;
                    int srcC = c < srcChannels ? c : (srcChannels == 1 ? 0 : c % srcChannels);
                    int chOffset = srcC * srcHeight * srcWidth;

                    double v00 = numOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x0]);
                    double v01 = numOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x1]);
                    double v10 = numOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x0]);
                    double v11 = numOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x1]);

                    value = v00 * (1 - dx) * (1 - dy) +
                            v01 * dx * (1 - dy) +
                            v10 * (1 - dx) * dy +
                            v11 * dx * dy;

                    // HWC output: pixels[(y * width + x) * channels + c]
                    pixels[(y * targetWidth + x) * channels + c] = numOps.FromDouble(value);
                }
            }
        }

        return pixels;
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
