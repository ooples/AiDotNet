using AiDotNet.Data.Vision.Benchmarks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Video.Benchmarks;

/// <summary>
/// Shared helper methods for video benchmark data loaders.
/// </summary>
internal static class VideoLoaderHelper
{
    internal static void LoadFrames<T>(string videoDir, T[] target, int offset, int framesPerVideo,
        int frameWidth, int frameHeight, bool normalize, INumericOperations<T> numOps)
    {
        if (string.IsNullOrWhiteSpace(videoDir))
            throw new ArgumentException("videoDir must not be null or whitespace.", nameof(videoDir));
        if (!Directory.Exists(videoDir))
            throw new DirectoryNotFoundException($"Video directory not found: {videoDir}");
        if (framesPerVideo <= 0)
            throw new ArgumentOutOfRangeException(nameof(framesPerVideo), "Must be positive.");
        if (frameWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameWidth), "Must be positive.");
        if (frameHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameHeight), "Must be positive.");
        var frameFiles = Directory.GetFiles(videoDir, "*.jpg");
        if (frameFiles.Length == 0)
            frameFiles = Directory.GetFiles(videoDir, "*.png");
        Array.Sort(frameFiles, StringComparer.OrdinalIgnoreCase);

        int framePixels = frameHeight * frameWidth * 3;
        for (int f = 0; f < framesPerVideo && f < frameFiles.Length; f++)
        {
            // Use VisionLoaderHelper to properly decode compressed images (JPEG/PNG)
            // rather than treating raw compressed bytes as pixel data
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(
                frameFiles[f], frameHeight, frameWidth, 3, normalize);
            int featureOffset = offset + f * framePixels;
            int copyLen = Math.Min(pixels.Length, framePixels);
            Array.Copy(pixels, 0, target, featureOffset, copyLen);
        }
    }

    internal static Tensor<T> ExtractTensorBatch<T>(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.ToArray().Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
