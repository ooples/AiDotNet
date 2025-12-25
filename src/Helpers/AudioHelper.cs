using System;
using System.IO;
using System.Text;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Helpers;

/// <summary>
/// Helper class for loading and saving audio as tensors.
/// </summary>
/// <remarks>
/// <para>
/// Supports common audio formats without external dependencies:
/// - WAV: Uncompressed PCM audio (most common for ML)
/// - RAW: Raw PCM samples with specified parameters
/// </para>
/// <para>
/// <b>For Beginners:</b> This class converts audio files into tensors for neural networks.
/// Audio is loaded as [channels, samples] or [batch, channels, samples] tensors.
/// Values are normalized to [-1, 1] range by default.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for tensor values.</typeparam>
public static class AudioHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Result of loading an audio file, including metadata.
    /// </summary>
    public class AudioLoadResult
    {
        /// <summary>Audio samples as tensor [1, channels, samples].</summary>
        public Tensor<T> Audio { get; init; } = new Tensor<T>(new[] { 1, 1, 0 });

        /// <summary>Sample rate in Hz.</summary>
        public int SampleRate { get; init; }

        /// <summary>Number of channels (1 = mono, 2 = stereo).</summary>
        public int Channels { get; init; }

        /// <summary>Bits per sample (8, 16, 24, or 32).</summary>
        public int BitsPerSample { get; init; }

        /// <summary>Duration in seconds.</summary>
        public double DurationSeconds => Audio.Shape[^1] / (double)SampleRate;
    }

    /// <summary>
    /// Loads an audio file and returns it as a tensor with metadata.
    /// </summary>
    /// <param name="filePath">Path to the audio file.</param>
    /// <param name="normalize">Whether to normalize to [-1, 1] range.</param>
    /// <param name="targetSampleRate">Optional target sample rate for resampling.</param>
    /// <returns>Audio tensor and metadata.</returns>
    /// <exception cref="FileNotFoundException">If the file does not exist.</exception>
    /// <exception cref="NotSupportedException">If the audio format is not supported.</exception>
    public static AudioLoadResult LoadAudio(string filePath, bool normalize = true, int? targetSampleRate = null)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Audio file not found: {filePath}", filePath);
        }

        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        var result = extension switch
        {
            ".wav" => LoadWav(filePath, normalize),
            ".raw" => throw new NotSupportedException("RAW format requires explicit parameters. Use LoadRaw method."),
            _ => throw new NotSupportedException($"Unsupported audio format: {extension}. Supported: .wav")
        };

        // Resample if requested
        if (targetSampleRate.HasValue && targetSampleRate.Value != result.SampleRate)
        {
            var resampled = Resample(result.Audio, result.SampleRate, targetSampleRate.Value);
            return new AudioLoadResult
            {
                Audio = resampled,
                SampleRate = targetSampleRate.Value,
                Channels = result.Channels,
                BitsPerSample = result.BitsPerSample
            };
        }

        return result;
    }

    /// <summary>
    /// Loads a WAV audio file.
    /// </summary>
    /// <param name="filePath">Path to the WAV file.</param>
    /// <param name="normalize">Whether to normalize to [-1, 1].</param>
    /// <returns>Audio tensor and metadata.</returns>
    public static AudioLoadResult LoadWav(string filePath, bool normalize = true)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new BinaryReader(stream);

        // RIFF header
        var riff = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (riff != "RIFF")
        {
            throw new InvalidDataException($"Invalid WAV file: expected RIFF header, got {riff}");
        }

        reader.ReadUInt32(); // File size
        var wave = Encoding.ASCII.GetString(reader.ReadBytes(4));
        if (wave != "WAVE")
        {
            throw new InvalidDataException($"Invalid WAV file: expected WAVE format, got {wave}");
        }

        // Find fmt and data chunks
        int sampleRate = 0;
        short channels = 0;
        short bitsPerSample = 0;
        short audioFormat = 0;
        byte[]? audioData = null;

        while (stream.Position < stream.Length)
        {
            var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
            var chunkSize = reader.ReadUInt32();

            switch (chunkId)
            {
                case "fmt ":
                    audioFormat = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    sampleRate = reader.ReadInt32();
                    reader.ReadInt32(); // Byte rate
                    reader.ReadInt16(); // Block align
                    bitsPerSample = reader.ReadInt16();

                    // Skip any extra format bytes
                    if (chunkSize > 16)
                    {
                        reader.ReadBytes((int)(chunkSize - 16));
                    }
                    break;

                case "data":
                    audioData = reader.ReadBytes((int)chunkSize);
                    break;

                default:
                    // Skip unknown chunks
                    reader.ReadBytes((int)chunkSize);
                    break;
            }
        }

        if (audioData == null)
        {
            throw new InvalidDataException("WAV file missing data chunk.");
        }

        if (audioFormat != 1 && audioFormat != 3)
        {
            throw new NotSupportedException($"Unsupported WAV format: {audioFormat}. Only PCM (1) and IEEE float (3) are supported.");
        }

        // Convert to tensor
        int bytesPerSample = bitsPerSample / 8;
        int numSamples = audioData.Length / (channels * bytesPerSample);
        var tensor = new Tensor<T>(new[] { 1, channels, numSamples });
        var span = tensor.AsWritableSpan();

        double maxVal = audioFormat == 3 ? 1.0 : Math.Pow(2, bitsPerSample - 1);
        bool isFloat = audioFormat == 3;

        int dataIdx = 0;
        for (int s = 0; s < numSamples; s++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sample;

                if (isFloat && bitsPerSample == 32)
                {
                    sample = BitConverter.ToSingle(audioData, dataIdx);
                    dataIdx += 4;
                }
                else if (isFloat && bitsPerSample == 64)
                {
                    sample = BitConverter.ToDouble(audioData, dataIdx);
                    dataIdx += 8;
                }
                else if (bitsPerSample == 8)
                {
                    // 8-bit WAV is unsigned
                    sample = (audioData[dataIdx++] - 128) / 128.0;
                }
                else if (bitsPerSample == 16)
                {
                    sample = BitConverter.ToInt16(audioData, dataIdx) / maxVal;
                    dataIdx += 2;
                }
                else if (bitsPerSample == 24)
                {
                    int val = audioData[dataIdx] | (audioData[dataIdx + 1] << 8) | (audioData[dataIdx + 2] << 16);
                    // Sign extend
                    if ((val & 0x800000) != 0)
                    {
                        val |= unchecked((int)0xFF000000);
                    }
                    sample = val / maxVal;
                    dataIdx += 3;
                }
                else if (bitsPerSample == 32)
                {
                    sample = BitConverter.ToInt32(audioData, dataIdx) / maxVal;
                    dataIdx += 4;
                }
                else
                {
                    throw new NotSupportedException($"Unsupported bits per sample: {bitsPerSample}");
                }

                if (!normalize)
                {
                    sample *= maxVal;
                }

                span[c * numSamples + s] = NumOps.FromDouble(sample);
            }
        }

        return new AudioLoadResult
        {
            Audio = tensor,
            SampleRate = sampleRate,
            Channels = channels,
            BitsPerSample = bitsPerSample
        };
    }

    /// <summary>
    /// Loads raw PCM audio data.
    /// </summary>
    /// <param name="filePath">Path to the raw audio file.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="bitsPerSample">Bits per sample (8, 16, 24, 32).</param>
    /// <param name="normalize">Whether to normalize to [-1, 1].</param>
    /// <returns>Audio tensor and metadata.</returns>
    public static AudioLoadResult LoadRaw(string filePath, int sampleRate, int channels = 1,
        int bitsPerSample = 16, bool normalize = true)
    {
        var data = File.ReadAllBytes(filePath);
        int bytesPerSample = bitsPerSample / 8;
        int numSamples = data.Length / (channels * bytesPerSample);

        var tensor = new Tensor<T>(new[] { 1, channels, numSamples });
        var span = tensor.AsWritableSpan();

        double maxVal = Math.Pow(2, bitsPerSample - 1);

        int dataIdx = 0;
        for (int s = 0; s < numSamples; s++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sample;

                if (bitsPerSample == 8)
                {
                    sample = (data[dataIdx++] - 128) / 128.0;
                }
                else if (bitsPerSample == 16)
                {
                    sample = BitConverter.ToInt16(data, dataIdx) / maxVal;
                    dataIdx += 2;
                }
                else if (bitsPerSample == 24)
                {
                    int val = data[dataIdx] | (data[dataIdx + 1] << 8) | (data[dataIdx + 2] << 16);
                    if ((val & 0x800000) != 0)
                    {
                        val |= unchecked((int)0xFF000000);
                    }
                    sample = val / maxVal;
                    dataIdx += 3;
                }
                else
                {
                    sample = BitConverter.ToInt32(data, dataIdx) / maxVal;
                    dataIdx += 4;
                }

                if (!normalize)
                {
                    sample *= maxVal;
                }

                span[c * numSamples + s] = NumOps.FromDouble(sample);
            }
        }

        return new AudioLoadResult
        {
            Audio = tensor,
            SampleRate = sampleRate,
            Channels = channels,
            BitsPerSample = bitsPerSample
        };
    }

    /// <summary>
    /// Saves audio tensor as a WAV file.
    /// </summary>
    /// <param name="audio">Audio tensor [channels, samples] or [1, channels, samples].</param>
    /// <param name="filePath">Output file path.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="bitsPerSample">Bits per sample (16 or 32).</param>
    /// <param name="denormalize">Whether to denormalize from [-1, 1].</param>
    public static void SaveWav(Tensor<T> audio, string filePath, int sampleRate,
        int bitsPerSample = 16, bool denormalize = true)
    {
        var shape = audio.Shape;
        int channels, numSamples;

        if (shape.Length == 3)
        {
            channels = shape[1];
            numSamples = shape[2];
        }
        else if (shape.Length == 2)
        {
            channels = shape[0];
            numSamples = shape[1];
        }
        else
        {
            throw new ArgumentException("Audio tensor must have 2 or 3 dimensions.");
        }

        var span = audio.AsSpan();
        int bytesPerSample = bitsPerSample / 8;
        int dataSize = numSamples * channels * bytesPerSample;

        using var stream = File.Create(filePath);
        using var writer = new BinaryWriter(stream);

        // RIFF header
        writer.Write(Encoding.ASCII.GetBytes("RIFF"));
        writer.Write(36 + dataSize); // File size - 8
        writer.Write(Encoding.ASCII.GetBytes("WAVE"));

        // fmt chunk
        writer.Write(Encoding.ASCII.GetBytes("fmt "));
        writer.Write(16); // Chunk size
        writer.Write((short)1); // PCM format
        writer.Write((short)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * channels * bytesPerSample); // Byte rate
        writer.Write((short)(channels * bytesPerSample)); // Block align
        writer.Write((short)bitsPerSample);

        // data chunk
        writer.Write(Encoding.ASCII.GetBytes("data"));
        writer.Write(dataSize);

        double maxVal = Math.Pow(2, bitsPerSample - 1) - 1;

        for (int s = 0; s < numSamples; s++)
        {
            for (int c = 0; c < channels; c++)
            {
                double sample = NumOps.ToDouble(span[c * numSamples + s]);

                if (denormalize)
                {
                    sample = Math.Clamp(sample * maxVal, -maxVal, maxVal);
                }

                if (bitsPerSample == 16)
                {
                    writer.Write((short)sample);
                }
                else if (bitsPerSample == 32)
                {
                    writer.Write((int)sample);
                }
                else
                {
                    throw new NotSupportedException($"Unsupported bits per sample for saving: {bitsPerSample}");
                }
            }
        }
    }

    /// <summary>
    /// Resamples audio to a different sample rate using linear interpolation.
    /// </summary>
    /// <param name="audio">Input audio tensor [1, channels, samples].</param>
    /// <param name="sourceSampleRate">Original sample rate.</param>
    /// <param name="targetSampleRate">Target sample rate.</param>
    /// <returns>Resampled audio tensor.</returns>
    public static Tensor<T> Resample(Tensor<T> audio, int sourceSampleRate, int targetSampleRate)
    {
        if (sourceSampleRate == targetSampleRate)
        {
            return audio;
        }

        var shape = audio.Shape;
        int channels = shape.Length == 3 ? shape[1] : shape[0];
        int srcSamples = shape[^1];
        int dstSamples = (int)((long)srcSamples * targetSampleRate / sourceSampleRate);

        var result = new Tensor<T>(new[] { 1, channels, dstSamples });
        var srcSpan = audio.AsSpan();
        var dstSpan = result.AsWritableSpan();

        double ratio = (double)(srcSamples - 1) / (dstSamples - 1);

        for (int c = 0; c < channels; c++)
        {
            int srcOffset = c * srcSamples;
            int dstOffset = c * dstSamples;

            for (int i = 0; i < dstSamples; i++)
            {
                double srcPos = i * ratio;
                int srcIdx = (int)srcPos;
                double frac = srcPos - srcIdx;

                double sample;
                if (srcIdx >= srcSamples - 1)
                {
                    sample = NumOps.ToDouble(srcSpan[srcOffset + srcSamples - 1]);
                }
                else
                {
                    double s0 = NumOps.ToDouble(srcSpan[srcOffset + srcIdx]);
                    double s1 = NumOps.ToDouble(srcSpan[srcOffset + srcIdx + 1]);
                    sample = s0 + frac * (s1 - s0);
                }

                dstSpan[dstOffset + i] = NumOps.FromDouble(sample);
            }
        }

        return result;
    }

    /// <summary>
    /// Converts stereo audio to mono by averaging channels.
    /// </summary>
    /// <param name="audio">Stereo audio tensor [1, 2, samples].</param>
    /// <returns>Mono audio tensor [1, 1, samples].</returns>
    public static Tensor<T> ToMono(Tensor<T> audio)
    {
        var shape = audio.Shape;
        int channels = shape.Length == 3 ? shape[1] : shape[0];

        if (channels == 1)
        {
            return audio;
        }

        int numSamples = shape[^1];
        var result = new Tensor<T>(new[] { 1, 1, numSamples });
        var srcSpan = audio.AsSpan();
        var dstSpan = result.AsWritableSpan();

        for (int s = 0; s < numSamples; s++)
        {
            double sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += NumOps.ToDouble(srcSpan[c * numSamples + s]);
            }
            dstSpan[s] = NumOps.FromDouble(sum / channels);
        }

        return result;
    }
}
