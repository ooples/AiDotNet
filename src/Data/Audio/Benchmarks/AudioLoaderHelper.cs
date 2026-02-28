using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Shared helper methods for audio data loaders.
/// Provides proper RIFF/WAVE chunk-based WAV parsing with support for 8/16/24/32-bit PCM
/// and multi-channel to mono conversion.
/// </summary>
internal static class AudioLoaderHelper
{
    /// <summary>
    /// Loads PCM WAV samples from raw bytes into a target array, normalizing to [-1, 1].
    /// Properly parses RIFF/WAVE header chunks to locate format and data sections.
    /// Supports 8/16/24/32-bit PCM. Multi-channel audio is averaged to mono.
    /// Pads with zeros if audio is shorter than maxSamples.
    /// </summary>
    internal static void LoadWavSamples<T>(byte[] wavBytes, T[] target, int offset, int maxSamples,
        INumericOperations<T> numOps)
    {
        if (wavBytes.Length < 12) return;

        // Validate RIFF/WAVE header
        string riffTag = System.Text.Encoding.ASCII.GetString(wavBytes, 0, 4);
        string waveTag = System.Text.Encoding.ASCII.GetString(wavBytes, 8, 4);
        if (riffTag != "RIFF" || waveTag != "WAVE")
        {
            // Not a valid WAV — fall back to raw 16-bit PCM
            LoadRaw16BitFallback(wavBytes, target, offset, maxSamples, numOps);
            return;
        }

        // Walk through chunks to find 'fmt ' and 'data'
        int bitsPerSample = 16;
        int numChannels = 1;
        int dataOffset = -1;
        int dataSize = 0;

        int pos = 12;
        while (pos + 8 <= wavBytes.Length)
        {
            string chunkId = System.Text.Encoding.ASCII.GetString(wavBytes, pos, 4);
            int chunkSize = BitConverter.ToInt32(wavBytes, pos + 4);
            if (chunkSize < 0 || pos + 8 + chunkSize > wavBytes.Length)
                break;

            if (chunkId == "fmt ")
            {
                if (chunkSize >= 16 && pos + 8 + 16 <= wavBytes.Length)
                {
                    numChannels = BitConverter.ToInt16(wavBytes, pos + 10);
                    bitsPerSample = BitConverter.ToInt16(wavBytes, pos + 22);
                    if (numChannels < 1) numChannels = 1;
                    if (bitsPerSample < 8) bitsPerSample = 16;
                }
            }
            else if (chunkId == "data")
            {
                dataOffset = pos + 8;
                dataSize = chunkSize;
                break;
            }

            // Move to next chunk (chunks are word-aligned)
            pos += 8 + chunkSize;
            if (chunkSize % 2 != 0) pos++;
        }

        if (dataOffset < 0)
        {
            // No data chunk found — fall back
            LoadRaw16BitFallback(wavBytes, target, offset, maxSamples, numOps);
            return;
        }

        LoadPcmSamples(wavBytes, dataOffset, dataSize, bitsPerSample, numChannels,
            target, offset, maxSamples, numOps);
    }

    /// <summary>
    /// Decodes PCM audio data at the specified bit depth with optional multi-channel to mono averaging.
    /// All samples are normalized to the [-1, 1] range.
    /// </summary>
    private static void LoadPcmSamples<T>(byte[] rawBytes, int dataOffset, int dataSize,
        int bitsPerSample, int numChannels, T[] target, int targetOffset, int maxSamples,
        INumericOperations<T> numOps)
    {
        int bytesPerSample = bitsPerSample / 8;
        int bytesPerFrame = bytesPerSample * numChannels;
        if (bytesPerFrame == 0) return;

        int totalFrames = dataSize / bytesPerFrame;
        int framesToRead = Math.Min(totalFrames, maxSamples);

        for (int i = 0; i < framesToRead; i++)
        {
            int frameOffset = dataOffset + i * bytesPerFrame;
            if (frameOffset + bytesPerFrame > rawBytes.Length) break;

            double value = DecodeSample(rawBytes, frameOffset, bitsPerSample);

            // Multi-channel to mono: average all channels
            if (numChannels > 1)
            {
                double sum = value;
                for (int ch = 1; ch < numChannels; ch++)
                {
                    int chOffset = frameOffset + ch * bytesPerSample;
                    if (chOffset + bytesPerSample > rawBytes.Length) break;
                    sum += DecodeSample(rawBytes, chOffset, bitsPerSample);
                }
                value = sum / numChannels;
            }

            target[targetOffset + i] = numOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Decodes a single PCM sample at the given byte offset, returning a normalized [-1, 1] value.
    /// Supports 8-bit (unsigned), 16-bit, 24-bit (sign-extended), and 32-bit signed PCM.
    /// </summary>
    private static double DecodeSample(byte[] rawBytes, int byteOffset, int bitsPerSample)
    {
        switch (bitsPerSample)
        {
            case 8:
                // 8-bit PCM is unsigned (0-255, centered at 128)
                return (rawBytes[byteOffset] - 128.0) / 128.0;

            case 16:
                short pcm16 = (short)(rawBytes[byteOffset] | (rawBytes[byteOffset + 1] << 8));
                return pcm16 / 32768.0;

            case 24:
                int pcm24 = rawBytes[byteOffset] | (rawBytes[byteOffset + 1] << 8) | (rawBytes[byteOffset + 2] << 16);
                if ((pcm24 & 0x800000) != 0) pcm24 |= unchecked((int)0xFF000000); // Sign extend
                return pcm24 / 8388608.0;

            case 32:
                int pcm32 = BitConverter.ToInt32(rawBytes, byteOffset);
                return pcm32 / 2147483648.0;

            default:
                // Fall back to 16-bit interpretation
                if (byteOffset + 1 < rawBytes.Length)
                {
                    short fallback = (short)(rawBytes[byteOffset] | (rawBytes[byteOffset + 1] << 8));
                    return fallback / 32768.0;
                }
                return 0.0;
        }
    }

    /// <summary>
    /// Fallback for files without valid RIFF headers — treats raw bytes as 16-bit signed PCM.
    /// </summary>
    private static void LoadRaw16BitFallback<T>(byte[] wavBytes, T[] target, int offset, int maxSamples,
        INumericOperations<T> numOps)
    {
        int bytesPerSample = 2;
        int numSamples = Math.Min(wavBytes.Length / bytesPerSample, maxSamples);

        for (int s = 0; s < numSamples; s++)
        {
            int byteIdx = s * bytesPerSample;
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
