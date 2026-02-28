using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Shared helper methods for audio data loaders.
/// </summary>
internal static class AudioLoaderHelper
{
    /// <summary>
    /// Loads 16-bit PCM WAV samples from raw bytes into a target array, normalizing to [-1, 1].
    /// Skips the 44-byte WAV header. Pads with zeros if audio is shorter than maxSamples.
    /// </summary>
    internal static void LoadWavSamples<T>(byte[] wavBytes, T[] target, int offset, int maxSamples,
        INumericOperations<T> numOps)
    {
        int headerSize = 44;
        if (wavBytes.Length <= headerSize) return;

        int bytesPerSample = 2; // 16-bit PCM
        int dataLen = wavBytes.Length - headerSize;
        int numSamples = Math.Min(dataLen / bytesPerSample, maxSamples);

        for (int s = 0; s < numSamples; s++)
        {
            int byteIdx = headerSize + s * bytesPerSample;
            if (byteIdx + 1 >= wavBytes.Length) break;

            short sample = (short)(wavBytes[byteIdx] | (wavBytes[byteIdx + 1] << 8));
            double normalized = sample / 32768.0;
            target[offset + s] = numOps.FromDouble(normalized);
        }
    }

    /// <summary>
    /// Extracts a batch of samples from a tensor by index.
    /// </summary>
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
