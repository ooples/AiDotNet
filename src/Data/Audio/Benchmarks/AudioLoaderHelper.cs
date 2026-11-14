using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Shared helper methods for audio data loaders.
/// Provides proper RIFF/WAVE chunk-based WAV parsing with support for 8/16/24/32-bit PCM
/// and multi-channel to mono conversion. Also supports FLAC decoding via <see cref="FlacDecoder"/>.
/// </summary>
internal static class AudioLoaderHelper
{
    /// <summary>
    /// Auto-detects audio format (WAV or FLAC) and loads samples into the target array.
    /// WAV: RIFF/WAVE chunk-based parsing. FLAC: full decode via <see cref="FlacDecoder"/>.
    /// Output is normalized to [-1, 1], multi-channel averaged to mono.
    /// </summary>
    /// <remarks>
    /// Only the decoded prefix is written into <paramref name="target"/>, so the remainder keeps its
    /// existing (zero) padding.
    /// </remarks>
    internal static void LoadAudioSamples<T>(byte[] audioBytes, T[] target, int offset, int maxSamples,
        INumericOperations<T> numOps)
    {
        if (maxSamples <= 0) return;

        // Detect FLAC by magic bytes "fLaC"
        if (audioBytes.Length >= 4 &&
            audioBytes[0] == (byte)'f' && audioBytes[1] == (byte)'L' &&
            audioBytes[2] == (byte)'a' && audioBytes[3] == (byte)'C')
        {
            FlacDecoder.DecodeFlac(audioBytes, target, offset, maxSamples);
            return;
        }

        var samples = DecodeWavSamples<T>(audioBytes, maxSamples);
        if (samples.Length > 0)
        {
            samples.AsSpan().CopyTo(target.AsSpan(offset, samples.Length));
        }
    }

    /// <summary>
    /// Decodes up to <paramref name="maxSamples"/> PCM WAV samples from raw bytes, normalized to [-1, 1].
    /// Properly parses RIFF/WAVE header chunks to locate format and data sections.
    /// Supports 8/16/24/32-bit PCM. Multi-channel audio is averaged to mono.
    /// Returns only the decoded samples (callers keep their own zero padding).
    /// </summary>
    internal static Tensor<T> DecodeWavSamples<T>(byte[] wavBytes, int maxSamples)
    {
        if (wavBytes.Length < 12 || maxSamples <= 0) return new Tensor<T>(new[] { 0 });

        // Validate RIFF/WAVE header
        string riffTag = System.Text.Encoding.ASCII.GetString(wavBytes, 0, 4);
        string waveTag = System.Text.Encoding.ASCII.GetString(wavBytes, 8, 4);
        if (riffTag != "RIFF" || waveTag != "WAVE")
        {
            // Not a valid WAV, fall back to raw 16-bit PCM
            return DecodeRaw16BitFallback<T>(wavBytes, maxSamples);
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
            // No data chunk found, fall back
            return DecodeRaw16BitFallback<T>(wavBytes, maxSamples);
        }

        return DecodePcmFrames<T>(wavBytes, dataOffset, dataSize, bitsPerSample, numChannels, maxSamples);
    }

    /// <summary>
    /// Decodes interleaved PCM frames to mono samples: every channel sample is decoded with engine ops and
    /// the channels of a frame are averaged (axis reduction).
    /// </summary>
    private static Tensor<T> DecodePcmFrames<T>(byte[] rawBytes, int dataOffset, int dataSize,
        int bitsPerSample, int numChannels, int maxSamples)
    {
        int bytesPerSample = bitsPerSample / 8;
        int bytesPerFrame = bytesPerSample * numChannels;
        if (bytesPerFrame == 0) return new Tensor<T>(new[] { 0 });

        int totalFrames = dataSize / bytesPerFrame;
        // Frames that would run past the end of the buffer are not decoded.
        int framesInBuffer = Math.Max(0, rawBytes.Length - dataOffset) / bytesPerFrame;
        int frames = Math.Min(Math.Min(totalFrames, maxSamples), framesInBuffer);
        if (frames <= 0) return new Tensor<T>(new[] { 0 });

        var samples = DecodePcmSamples<T>(rawBytes, dataOffset, frames * numChannels, bitsPerSample, bytesPerSample);
        if (numChannels == 1) return samples;

        // Multi-channel to mono: average all channels of each frame
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        var channelSums = engine.ReduceSum(samples.Reshape(new[] { frames, numChannels }), new[] { 1 }, false);
        return engine.TensorDivideScalar(channelSums, numOps.FromDouble(numChannels));
    }

    /// <summary>
    /// Decodes <paramref name="count"/> consecutive little-endian PCM samples starting at
    /// <paramref name="start"/>, normalized to [-1, 1]: 8-bit is unsigned (centred at 128), 16/24/32-bit
    /// are two's-complement; other depths read the first two bytes of each sample as 16-bit PCM.
    /// </summary>
    /// <remarks>
    /// The sample bytes are widened to a <c>[count, bytes]</c> tensor and combined with the byte place
    /// values (a fixed coefficient table) in one matrix product; the sign comes from the top byte. With
    /// <paramref name="normalize"/> false the signed (8-bit: centred) integer sample values are returned.
    /// </remarks>
    internal static Tensor<T> DecodePcmSamples<T>(byte[] rawBytes, int start, int count, int bitsPerSample,
        int bytesPerSample, bool normalize = true)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        bool standardDepth = bitsPerSample == 8 || bitsPerSample == 16 || bitsPerSample == 24 || bitsPerSample == 32;

        Tensor<T> sampleBytes;
        int width;
        bool lastSampleMissing = false;
        if (standardDepth)
        {
            width = bytesPerSample;
            var bytes = new int[count * width];
            Array.Copy(rawBytes, start, bytes, 0, bytes.Length);
            sampleBytes = new Tensor<int>(bytes, new[] { count, width }).Cast<T>();
        }
        else
        {
            // Fall back to the 16-bit interpretation of the first two bytes of each sample; a sample whose
            // second byte lies past the end of the buffer decodes to 0.
            width = 2;
            int span = (count - 1) * bytesPerSample + 2;
            int available = Math.Min(rawBytes.Length - start, span);
            lastSampleMissing = available < span;
            var region = new int[span];
            Array.Copy(rawBytes, start, region, 0, available);
            var lowIndex = Enumerable.Range(0, count).Select(s => s * bytesPerSample).ToArray();
            var highIndex = Enumerable.Range(0, count).Select(s => s * bytesPerSample + 1).ToArray();
            var regionTensor = new Tensor<int>(region, new[] { span }).Cast<T>();
            var low = engine.TensorIndexSelect(regionTensor, new Tensor<int>(lowIndex, new[] { count }), 0);
            var high = engine.TensorIndexSelect(regionTensor, new Tensor<int>(highIndex, new[] { count }), 0);
            sampleBytes = engine.TensorStack(new[] { low, high }, 1);
        }

        // Unsigned value = sum of byte * 256^k (little-endian place values)
        var placeValues = new Tensor<int>(new[] { 1, 256, 65536, 16777216 }.Take(width).ToArray(), new[] { width, 1 })
            .Cast<T>();
        var unsignedValue = engine.TensorMatMul(sampleBytes, placeValues).Reshape(new[] { count });

        Tensor<T> signedValue;
        double fullScale;
        if (width == 1)
        {
            // 8-bit PCM is unsigned (0-255, centered at 128)
            signedValue = engine.TensorSubtractScalar(unsignedValue, numOps.FromDouble(128.0));
            fullScale = 128.0;
        }
        else
        {
            // Two's complement: subtract 2^(8 * width) when the top byte's sign bit is set
            var topByte = sampleBytes.Slice(1, width - 1, width).Contiguous().Reshape(new[] { count });
            var signBit = engine.TensorGreaterThan(topByte, numOps.FromDouble(127.0));
            signedValue = engine.TensorSubtract(
                unsignedValue, engine.TensorMultiplyScalar(signBit, numOps.FromDouble(Math.Pow(2, 8 * width))));
            fullScale = Math.Pow(2, 8 * width - 1);
        }

        var normalized = normalize
            ? engine.TensorDivideScalar(signedValue, numOps.FromDouble(fullScale))
            : signedValue;

        if (lastSampleMissing)
        {
            normalized[count - 1] = numOps.Zero;
        }

        return normalized;
    }

    /// <summary>
    /// Fallback: decodes raw bytes as 16-bit little-endian PCM (no header).
    /// </summary>
    private static Tensor<T> DecodeRaw16BitFallback<T>(byte[] wavBytes, int maxSamples)
    {
        int numSamples = Math.Min(wavBytes.Length / 2, maxSamples);
        if (numSamples <= 0) return new Tensor<T>(new[] { 0 });
        return DecodePcmSamples<T>(wavBytes, 0, numSamples, 16, 2);
    }

    /// <summary>
    /// Extracts a batch of samples from a tensor by index.
    /// </summary>
    internal static Tensor<T> ExtractTensorBatch<T>(Tensor<T> source, int[] indices)
    {
        var result = TensorCopyHelper.CreateEmptyBatchLike(source, indices.Length);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
