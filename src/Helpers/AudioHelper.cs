using System;
using System.IO;
using System.Text;
using AiDotNet.Interfaces;
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
        return LoadWav(stream, normalize);
    }

    /// <summary>
    /// Decodes a WAV file held in memory (for example the body of a text-to-speech API response).
    /// </summary>
    /// <param name="bytes">The bytes of the WAV file.</param>
    /// <param name="normalize">Whether to normalize to [-1, 1].</param>
    /// <returns>Audio tensor and metadata.</returns>
    public static AudioLoadResult DecodeWav(byte[] bytes, bool normalize = true)
    {
        if (bytes is null) throw new ArgumentNullException(nameof(bytes));
        using var stream = new MemoryStream(bytes, writable: false);
        return LoadWav(stream, normalize);
    }

    /// <summary>
    /// Reads a WAV file from a stream: PCM (8, 16, 24, 32 bits), IEEE float (32, 64 bits) and their
    /// WAVE_FORMAT_EXTENSIBLE forms. RIFF chunks are word-aligned, so an odd-sized chunk is followed by a pad byte; a
    /// data chunk whose size is unknown (0 or 0xFFFFFFFF, as streamed responses write it) extends to the end.
    /// </summary>
    /// <param name="stream">The stream positioned at the RIFF header; it is left open.</param>
    /// <param name="normalize">Whether to normalize to [-1, 1].</param>
    /// <returns>Audio tensor and metadata.</returns>
    public static AudioLoadResult LoadWav(Stream stream, bool normalize = true)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);

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

        while (stream.Length - stream.Position >= 8)
        {
            var chunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
            long chunkSize = reader.ReadUInt32();
            long remaining = stream.Length - stream.Position;

            switch (chunkId)
            {
                case "fmt ":
                    audioFormat = reader.ReadInt16();
                    channels = reader.ReadInt16();
                    sampleRate = reader.ReadInt32();
                    reader.ReadInt32(); // Byte rate
                    reader.ReadInt16(); // Block align
                    bitsPerSample = reader.ReadInt16();
                    long consumed = 16;
                    // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID's first two bytes are the actual format tag.
                    if (audioFormat == unchecked((short)0xFFFE) && chunkSize >= 40)
                    {
                        reader.ReadInt16(); // cbSize
                        reader.ReadInt16(); // Valid bits per sample
                        reader.ReadInt32(); // Channel mask
                        audioFormat = reader.ReadInt16();
                        reader.ReadBytes(14); // Rest of the sub-format GUID
                        consumed = 40;
                    }

                    // Skip any extra format bytes
                    if (chunkSize > consumed)
                    {
                        reader.ReadBytes((int)(chunkSize - consumed));
                    }
                    break;

                case "data":
                    if (chunkSize == 0 || chunkSize == 0xFFFFFFFF || chunkSize > remaining)
                    {
                        chunkSize = remaining;
                    }
                    audioData = reader.ReadBytes((int)chunkSize);
                    break;

                default:
                    // Skip unknown chunks
                    reader.ReadBytes((int)Math.Min(chunkSize, remaining));
                    break;
            }

            // Chunks are word-aligned: an odd-sized chunk carries one pad byte.
            if ((chunkSize & 1) == 1 && stream.Position < stream.Length)
            {
                reader.ReadByte();
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
        var shape = audio._shape;
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
                    sample = MathPolyfill.Clamp(sample * maxVal, -maxVal, maxVal);
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

        // Band-limited (windowed-sinc) resampling per channel: plain interpolation folds everything above the target
        // Nyquist frequency back into the band when downsampling.
        var shape = audio._shape;
        int channels = shape.Length == 3 ? shape[1] : shape.Length == 2 ? shape[0] : 1;
        int srcSamples = shape[^1];
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var source = audio.AsSpan();
        Tensor<T>? result = null;
        int dstSamples = 0;
        for (int c = 0; c < channels; c++)
        {
            var channel = new Tensor<T>(new[] { srcSamples });
            for (int i = 0; i < srcSamples; i++) channel[i] = source[c * srcSamples + i];
            var resampled = ResampleBandLimited(engine, channel, sourceSampleRate, targetSampleRate);
            if (result is null)
            {
                dstSamples = resampled.Length;
                result = new Tensor<T>(new[] { 1, channels, dstSamples });
            }
            var destination = result.AsWritableSpan();
            for (int i = 0; i < dstSamples; i++) destination[c * dstSamples + i] = resampled[i];
        }

        return result!;
    }

    /// <summary>
    /// Band-limited resampling as torchaudio's <c>transforms.Resample</c> (Hann-windowed sinc, 6 zero crossings,
    /// roll-off 0.99) of <paramref name="audio"/> <c>[samples]</c> from <paramref name="from"/> Hz to <paramref name="to"/>
    /// Hz, computed as a strided convolution so it is differentiable on the gradient tape. The output has
    /// ⌈to · samples / from⌉ samples.
    /// </summary>
    public static Tensor<T> ResampleBandLimited(IEngine engine, Tensor<T> audio, int from, int to)
    {
        if (from <= 0 || to <= 0) throw new ArgumentOutOfRangeException(nameof(from), "Sample rates must be positive.");
        if (from == to) return audio;
        int gcd = Gcd(from, to), orig = from / gcd, target = to / gcd;
        const int zeroCrossings = 6;
        const double rolloff = 0.99;
        double baseFrequency = Math.Min(orig, target) * rolloff;
        int width = (int)Math.Ceiling(zeroCrossings * orig / baseFrequency);
        int taps = 2 * width + orig;
        var kernel = new Tensor<T>(new[] { target, 1, 1, taps });
        for (int phase = 0; phase < target; phase++)
            for (int j = 0; j < taps; j++)
            {
                double t = (-(double)phase / target + (double)(j - width) / orig) * baseFrequency;
                t = Math.Max(-zeroCrossings, Math.Min(zeroCrossings, t));
                double window = Math.Pow(Math.Cos(t * Math.PI / zeroCrossings / 2), 2);
                double x = t * Math.PI;
                double sinc = x == 0 ? 1.0 : Math.Sin(x) / x;
                kernel[phase, 0, 0, j] = NumOps.FromDouble(sinc * window * baseFrequency / orig);
            }
        int length = audio.Length;
        var flat = engine.Reshape(audio, new[] { 1, 1, 1, length });
        var padded = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1, 1, 1, width }), flat, new Tensor<T>(new[] { 1, 1, 1, width + orig }) }, 3);
        var phases = engine.Conv2D(padded, kernel, new[] { 1, orig }, new[] { 0, 0 }, new[] { 1, 1 });         // [1, target, 1, n]
        int n = phases.Shape[3];
        var interleaved = engine.Reshape(engine.TensorTranspose(engine.Reshape(phases, new[] { target, n })), new[] { target * n });
        int outLength = (int)Math.Ceiling((double)target * length / orig);
        return engine.TensorSlice(interleaved, new[] { 0 }, new[] { Math.Min(outLength, target * n) });
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a;
    }

    /// <summary>
    /// Converts stereo audio to mono by averaging channels.
    /// </summary>
    /// <param name="audio">Stereo audio tensor [1, 2, samples].</param>
    /// <returns>Mono audio tensor [1, 1, samples].</returns>
    public static Tensor<T> ToMono(Tensor<T> audio)
    {
        var shape = audio._shape;
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
