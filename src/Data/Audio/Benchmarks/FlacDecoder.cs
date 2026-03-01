using AiDotNet.Interfaces;

namespace AiDotNet.Data.Audio.Benchmarks;

/// <summary>
/// Minimal pure C# FLAC (Free Lossless Audio Codec) decoder for ML audio pipelines.
/// Supports 8/16/24-bit PCM, mono and stereo, fixed and LPC prediction with rice-coded residuals.
/// Multi-channel audio is averaged to mono. Output is normalized to [-1, 1].
/// </summary>
/// <remarks>
/// This decoder handles the subset of FLAC commonly found in ML datasets (LibriSpeech, GigaSpeech, etc.).
/// It does not implement CRC validation or seek tables. For full FLAC support, use a dedicated library.
/// </remarks>
internal static class FlacDecoder
{
    /// <summary>
    /// Decodes FLAC audio bytes into a target array of normalized [-1, 1] mono samples.
    /// </summary>
    internal static void DecodeFlac<T>(byte[] flacBytes, T[] target, int offset, int maxSamples,
        INumericOperations<T> numOps)
    {
        if (flacBytes.Length < 42) return; // Minimum FLAC: 4 magic + 38 STREAMINFO

        // Verify "fLaC" magic
        if (flacBytes[0] != (byte)'f' || flacBytes[1] != (byte)'L' ||
            flacBytes[2] != (byte)'a' || flacBytes[3] != (byte)'C')
            return;

        // Parse metadata blocks to find STREAMINFO
        int pos = 4;
        int sampleRate = 44100;
        int numChannels = 1;
        int bitsPerSample = 16;
        long totalSamples = 0;

        while (pos + 4 <= flacBytes.Length)
        {
            bool isLast = (flacBytes[pos] & 0x80) != 0;
            int blockType = flacBytes[pos] & 0x7F;
            int blockLength = (flacBytes[pos + 1] << 16) | (flacBytes[pos + 2] << 8) | flacBytes[pos + 3];
            pos += 4;

            if (pos + blockLength > flacBytes.Length) break;

            if (blockType == 0 && blockLength >= 34) // STREAMINFO
            {
                // Bytes 10-13: sample rate (20 bits), channels (3 bits), bps (5 bits)
                int b10 = flacBytes[pos + 10];
                int b11 = flacBytes[pos + 11];
                int b12 = flacBytes[pos + 12];
                int b13 = flacBytes[pos + 13];

                sampleRate = (b10 << 12) | (b11 << 4) | (b12 >> 4);
                numChannels = ((b12 >> 1) & 0x07) + 1;
                bitsPerSample = ((b12 & 0x01) << 4) | (b13 >> 4);
                bitsPerSample += 1;

                // Total samples: 36 bits across bytes 13-17
                totalSamples = ((long)(b13 & 0x0F) << 32) |
                               ((long)flacBytes[pos + 14] << 24) |
                               ((long)flacBytes[pos + 15] << 16) |
                               ((long)flacBytes[pos + 16] << 8) |
                               flacBytes[pos + 17];
            }

            pos += blockLength;
            if (isLast) break;
        }

        // Decode audio frames
        int samplesWritten = 0;
        double normFactor = 1.0 / (1L << (bitsPerSample - 1));

        while (pos + 2 < flacBytes.Length && samplesWritten < maxSamples)
        {
            // Find frame sync: 0xFF 0xF8 or 0xFF 0xF9
            if (flacBytes[pos] != 0xFF || (flacBytes[pos + 1] & 0xFE) != 0xF8)
            {
                pos++;
                continue;
            }

            var reader = new BitReader(flacBytes, pos);
            int frameStart = pos;

            try
            {
                int written = DecodeFrame(reader, numChannels, bitsPerSample, normFactor,
                    target, offset + samplesWritten, maxSamples - samplesWritten, numOps);
                samplesWritten += written;
                pos = reader.BytePosition;
                // Align to byte boundary and skip CRC-16
                if (reader.BitOffset > 0) pos++;
                pos += 2;
                if (pos <= frameStart) pos = frameStart + 2; // Safety: always advance
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // Frame decode failed (e.g., truncated data, invalid subframe type).
                // Skip to next potential sync code rather than aborting entire decode.
                System.Diagnostics.Debug.WriteLine($"FLAC frame decode failed at byte {frameStart}: {ex.Message}");
                pos = frameStart + 2;
            }
        }
    }

    private static int DecodeFrame<T>(BitReader reader, int streamChannels, int streamBps,
        double normFactor, T[] target, int offset, int maxSamples, INumericOperations<T> numOps)
    {
        // Frame header
        reader.ReadBits(14); // sync code
        if (reader.ReadBits(1) != 0) return 0; // reserved
        int blockingStrategy = reader.ReadBits(1);

        int blockSizeCode = reader.ReadBits(4);
        int sampleRateCode = reader.ReadBits(4);
        int channelAssignment = reader.ReadBits(4);
        int sampleSizeCode = reader.ReadBits(3);
        reader.ReadBits(1); // reserved

        // Frame/sample number (UTF-8-like variable length)
        ReadUtf8Long(reader);

        // Block size
        int blockSize;
        if (blockSizeCode == 1) blockSize = 192;
        else if (blockSizeCode >= 2 && blockSizeCode <= 5) blockSize = 576 * (1 << (blockSizeCode - 2));
        else if (blockSizeCode == 6) blockSize = reader.ReadBits(8) + 1;
        else if (blockSizeCode == 7) blockSize = reader.ReadBits(16) + 1;
        else if (blockSizeCode >= 8) blockSize = 256 * (1 << (blockSizeCode - 8));
        else return 0;

        // Sample rate (skip if encoded in header)
        if (sampleRateCode == 12) reader.ReadBits(8);
        else if (sampleRateCode == 13 || sampleRateCode == 14) reader.ReadBits(16);

        // Bits per sample from frame header
        int bps = streamBps;
        switch (sampleSizeCode)
        {
            case 1: bps = 8; break;
            case 2: bps = 12; break;
            case 4: bps = 16; break;
            case 5: bps = 20; break;
            case 6: bps = 24; break;
        }

        reader.ReadBits(8); // CRC-8

        // Determine actual channels from assignment
        int numChannels;
        bool isLeftSide = false, isRightSide = false, isMidSide = false;
        if (channelAssignment < 8)
        {
            numChannels = channelAssignment + 1;
        }
        else if (channelAssignment == 8)
        {
            numChannels = 2;
            isLeftSide = true;
        }
        else if (channelAssignment == 9)
        {
            numChannels = 2;
            isRightSide = true;
        }
        else if (channelAssignment == 10)
        {
            numChannels = 2;
            isMidSide = true;
        }
        else
        {
            return 0; // Reserved
        }

        // Decode subframes
        var channelData = new int[numChannels][];
        for (int ch = 0; ch < numChannels; ch++)
        {
            // For stereo decorrelation, one channel gets an extra bit
            int subframeBps = bps;
            if (isLeftSide && ch == 1) subframeBps++;
            else if (isRightSide && ch == 0) subframeBps++;
            else if (isMidSide && ch == 1) subframeBps++;

            channelData[ch] = DecodeSubframe(reader, blockSize, subframeBps);
        }

        // Apply inter-channel decorrelation
        if (isLeftSide)
        {
            for (int i = 0; i < blockSize; i++)
                channelData[1][i] = channelData[0][i] - channelData[1][i];
        }
        else if (isRightSide)
        {
            for (int i = 0; i < blockSize; i++)
                channelData[0][i] = channelData[0][i] + channelData[1][i];
        }
        else if (isMidSide)
        {
            for (int i = 0; i < blockSize; i++)
            {
                int mid = channelData[0][i];
                int side = channelData[1][i];
                // Mid-side decoding: mid is stored shifted right by 1, side is the difference.
                // Restore: mid = mid << 1 | (side & 1), then left = (mid + side) / 2, right = (mid - side) / 2
                mid = (mid << 1) | (side & 1);
                channelData[0][i] = (mid + side) >> 1;
                channelData[1][i] = (mid - side) >> 1;
            }
        }

        // Output: average channels to mono, normalize
        int samplesToWrite = Math.Min(blockSize, maxSamples);
        for (int i = 0; i < samplesToWrite; i++)
        {
            double sample;
            if (numChannels == 1)
            {
                sample = channelData[0][i] * normFactor;
            }
            else
            {
                double sum = 0;
                for (int ch = 0; ch < numChannels; ch++)
                    sum += channelData[ch][i];
                sample = (sum / numChannels) * normFactor;
            }

            target[offset + i] = numOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, sample)));
        }

        return samplesToWrite;
    }

    private static int[] DecodeSubframe(BitReader reader, int blockSize, int bps)
    {
        // Subframe header
        reader.ReadBits(1); // padding zero
        int subframeType = reader.ReadBits(6);

        // Wasted bits per sample
        int wastedBits = 0;
        if (reader.ReadBits(1) == 1)
        {
            wastedBits = 1;
            while (reader.ReadBits(1) == 0)
                wastedBits++;
        }
        int effectiveBps = bps - wastedBits;

        int[] samples;

        if (subframeType == 0)
        {
            // CONSTANT
            int value = reader.ReadSignedBits(effectiveBps);
            samples = new int[blockSize];
            for (int i = 0; i < blockSize; i++)
                samples[i] = value;
        }
        else if (subframeType == 1)
        {
            // VERBATIM
            samples = new int[blockSize];
            for (int i = 0; i < blockSize; i++)
                samples[i] = reader.ReadSignedBits(effectiveBps);
        }
        else if (subframeType >= 8 && subframeType <= 12)
        {
            // FIXED prediction, order = subframeType - 8
            int order = subframeType - 8;
            samples = DecodeFixedSubframe(reader, blockSize, effectiveBps, order);
        }
        else if (subframeType >= 32 && subframeType <= 63)
        {
            // LPC prediction, order = subframeType - 31
            int order = subframeType - 31;
            samples = DecodeLpcSubframe(reader, blockSize, effectiveBps, order);
        }
        else
        {
            // Unknown/reserved — return silence
            samples = new int[blockSize];
        }

        // Shift back wasted bits
        if (wastedBits > 0)
        {
            for (int i = 0; i < blockSize; i++)
                samples[i] <<= wastedBits;
        }

        return samples;
    }

    private static int[] DecodeFixedSubframe(BitReader reader, int blockSize, int bps, int order)
    {
        var samples = new int[blockSize];

        // Read warm-up samples
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadSignedBits(bps);

        // Decode residual
        int[] residual = DecodeResidual(reader, blockSize, order);

        // Apply fixed prediction
        for (int i = order; i < blockSize; i++)
        {
            int prediction = order switch
            {
                0 => 0,
                1 => samples[i - 1],
                2 => 2 * samples[i - 1] - samples[i - 2],
                3 => 3 * samples[i - 1] - 3 * samples[i - 2] + samples[i - 3],
                4 => 4 * samples[i - 1] - 6 * samples[i - 2] + 4 * samples[i - 3] - samples[i - 4],
                _ => 0
            };
            samples[i] = prediction + residual[i - order];
        }

        return samples;
    }

    private static int[] DecodeLpcSubframe(BitReader reader, int blockSize, int bps, int order)
    {
        var samples = new int[blockSize];

        // Read warm-up samples
        for (int i = 0; i < order; i++)
            samples[i] = reader.ReadSignedBits(bps);

        // LPC parameters
        int qlpPrecision = reader.ReadBits(4) + 1;
        int qlpShift = reader.ReadSignedBits(5);
        var qlpCoeffs = new int[order];
        for (int i = 0; i < order; i++)
            qlpCoeffs[i] = reader.ReadSignedBits(qlpPrecision);

        // Decode residual
        int[] residual = DecodeResidual(reader, blockSize, order);

        // Apply LPC prediction
        for (int i = order; i < blockSize; i++)
        {
            long prediction = 0;
            for (int j = 0; j < order; j++)
                prediction += (long)qlpCoeffs[j] * samples[i - 1 - j];
            prediction >>= qlpShift;
            samples[i] = (int)prediction + residual[i - order];
        }

        return samples;
    }

    private static int[] DecodeResidual(BitReader reader, int blockSize, int predictorOrder)
    {
        int codingMethod = reader.ReadBits(2);
        int partitionOrder = reader.ReadBits(4);
        int numPartitions = 1 << partitionOrder;
        int residualCount = blockSize - predictorOrder;
        var residual = new int[residualCount];

        int riceParamBits = codingMethod == 0 ? 4 : 5;
        int escapeCode = codingMethod == 0 ? 15 : 31;

        int sampleIdx = 0;
        for (int partition = 0; partition < numPartitions; partition++)
        {
            int samplesInPartition = partition == 0
                ? (blockSize >> partitionOrder) - predictorOrder
                : blockSize >> partitionOrder;

            int riceParam = reader.ReadBits(riceParamBits);

            if (riceParam == escapeCode)
            {
                // Escape: raw encoding
                int rawBits = reader.ReadBits(5);
                for (int i = 0; i < samplesInPartition && sampleIdx < residualCount; i++, sampleIdx++)
                    residual[sampleIdx] = reader.ReadSignedBits(rawBits);
            }
            else
            {
                // Rice coding
                for (int i = 0; i < samplesInPartition && sampleIdx < residualCount; i++, sampleIdx++)
                {
                    // Unary: count leading zeros
                    int q = 0;
                    while (reader.ReadBits(1) == 0)
                        q++;

                    int r = riceParam > 0 ? reader.ReadBits(riceParam) : 0;
                    int value = (q << riceParam) | r;

                    // Zigzag decode: even -> positive, odd -> negative
                    residual[sampleIdx] = (value & 1) == 0 ? value >> 1 : -(value >> 1) - 1;
                }
            }
        }

        return residual;
    }

    private static long ReadUtf8Long(BitReader reader)
    {
        int first = reader.ReadBits(8);
        if ((first & 0x80) == 0) return first;

        int numBytes;
        long value;
        if ((first & 0xE0) == 0xC0) { numBytes = 2; value = first & 0x1F; }
        else if ((first & 0xF0) == 0xE0) { numBytes = 3; value = first & 0x0F; }
        else if ((first & 0xF8) == 0xF0) { numBytes = 4; value = first & 0x07; }
        else if ((first & 0xFC) == 0xF8) { numBytes = 5; value = first & 0x03; }
        else if ((first & 0xFE) == 0xFC) { numBytes = 6; value = first & 0x01; }
        else { numBytes = 7; value = 0; }

        for (int i = 1; i < numBytes; i++)
            value = (value << 6) | (reader.ReadBits(8) & 0x3FL);

        return value;
    }

    /// <summary>
    /// Bit-level reader for FLAC bitstream parsing.
    /// </summary>
    private sealed class BitReader
    {
        private readonly byte[] _data;
        private int _bytePos;
        private int _bitPos; // 0-7, bits remaining in current byte (counted from MSB)

        public int BytePosition => _bytePos;
        public int BitOffset => _bitPos;

        public BitReader(byte[] data, int startByte)
        {
            _data = data;
            _bytePos = startByte;
            _bitPos = 0;
        }

        public int ReadBits(int count)
        {
            int result = 0;
            while (count > 0)
            {
                if (_bytePos >= _data.Length)
                    throw new InvalidDataException("Unexpected end of FLAC bitstream.");

                int bitsAvailable = 8 - _bitPos;
                int bitsToRead = Math.Min(count, bitsAvailable);

                int shift = bitsAvailable - bitsToRead;
                int mask = ((1 << bitsToRead) - 1) << shift;
                int value = (_data[_bytePos] & mask) >> shift;

                result = (result << bitsToRead) | value;
                count -= bitsToRead;
                _bitPos += bitsToRead;

                if (_bitPos >= 8)
                {
                    _bitPos = 0;
                    _bytePos++;
                }
            }

            return result;
        }

        public int ReadSignedBits(int count)
        {
            if (count <= 0) return 0;
            int value = ReadBits(count);
            // Sign extend
            if ((value & (1 << (count - 1))) != 0)
                value |= ~((1 << count) - 1);
            return value;
        }
    }
}
