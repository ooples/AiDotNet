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
        var frameFiles = Directory.GetFiles(videoDir, "*.jpg");
        if (frameFiles.Length == 0)
            frameFiles = Directory.GetFiles(videoDir, "*.png");
        Array.Sort(frameFiles, StringComparer.OrdinalIgnoreCase);

        int framePixels = frameHeight * frameWidth * 3;
        for (int f = 0; f < framesPerVideo && f < frameFiles.Length; f++)
        {
            byte[] imageBytes = File.ReadAllBytes(frameFiles[f]);
            int featureOffset = offset + f * framePixels;
            int pixelCount = Math.Min(imageBytes.Length, framePixels);
            for (int p = 0; p < pixelCount; p++)
            {
                double val = normalize ? imageBytes[p] / 255.0 : imageBytes[p];
                target[featureOffset + p] = numOps.FromDouble(val);
            }
        }
    }

    internal static Tensor<T> ExtractTensorBatch<T>(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
