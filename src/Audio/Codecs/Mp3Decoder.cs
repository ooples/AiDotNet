namespace AiDotNet.Audio.Codecs;

/// <summary>
/// An MPEG-1/2/2.5 Layer III (MP3) decoder: frame synchronization, side information, the bit reservoir, scale factors,
/// Huffman decoding and requantization, mid/side and intensity stereo, reordering, alias reduction, the hybrid IMDCT and
/// the polyphase synthesis filterbank.
/// </summary>
/// <remarks>
/// <para>A port of the scalar path of minimp3 by Lion (lieff), released into the public domain under CC0 1.0
/// (https://github.com/lieff/minimp3); its tables are carried over verbatim. Layers I and II are not decoded.</para>
/// <para><b>For Beginners:</b> MP3 stores audio compactly; <see cref="Decode"/> turns an MP3 file's bytes back into
/// samples in [−1, 1].</para>
/// </remarks>
public static class Mp3Decoder
{
    /// <summary>The decoded audio: interleaved samples in [−1, 1], the sample rate and the channel count.</summary>
    public sealed class Result
    {
        internal Result(float[] samples, int sampleRate, int channels)
        {
            Samples = samples;
            SampleRate = sampleRate;
            Channels = channels;
        }

        /// <summary>The samples, interleaved by channel.</summary>
        public float[] Samples { get; }

        /// <summary>The sample rate in Hz.</summary>
        public int SampleRate { get; }

        /// <summary>The number of channels.</summary>
        public int Channels { get; }
    }

    /// <summary>Decodes every Layer III frame of an MP3 stream (ID3 tags and other data between frames are skipped).</summary>
    /// <param name="mp3">The bytes of the MP3 file.</param>
    /// <returns>The decoded audio.</returns>
    /// <exception cref="InvalidDataException">No Layer III frame was found.</exception>
    public static Result Decode(byte[] mp3)
    {
        if (mp3 is null) throw new ArgumentNullException(nameof(mp3));
        var decoder = new State();
        var output = new List<float>();
        var pcm = new float[1152 * 2];
        int offset = 0, sampleRate = 0, channels = 0;
        while (offset < mp3.Length)
        {
            int samples = DecodeFrame(decoder, mp3, offset, mp3.Length - offset, pcm, out var info);
            if (info.FrameBytes == 0) break;
            offset += info.FrameBytes;
            if (samples > 0)
            {
                if (sampleRate == 0)
                {
                    sampleRate = info.Hz;
                    channels = info.Channels;
                }
                for (int i = 0; i < samples * info.Channels; i++) output.Add(pcm[i]);
            }
        }
        if (sampleRate == 0) throw new InvalidDataException("No MPEG Layer III frame was found.");
        return new Result(output.ToArray(), sampleRate, channels);
    }

    // ------------------------------------------------------------------ constants and header

    private const int MaxFreeFormatFrameSize = 2304;
    private const int MaxFrameSyncMatches = 10;
    private const int MaxL3FramePayloadBytes = MaxFreeFormatFrameSize;
    private const int MaxBitReservoirBytes = 511;
    private const int ShortBlockType = 2;
    private const int StopBlockType = 3;
    private const int HdrSize = 4;
    private const int BitsDequantizerOut = -1;
    private const int MaxScf = 255 + BitsDequantizerOut * 4 - 210;
    private const int MaxScfi = (MaxScf + 3) & ~3;

    private static bool IsMono(byte[] b, int h) => (b[h + 3] & 0xC0) == 0xC0;
    private static bool IsMsStereo(byte[] b, int h) => (b[h + 3] & 0xE0) == 0x60;
    private static bool IsFreeFormat(byte[] b, int h) => (b[h + 2] & 0xF0) == 0;
    private static bool IsCrc(byte[] b, int h) => (b[h + 1] & 1) == 0;
    private static bool TestPadding(byte[] b, int h) => (b[h + 2] & 0x2) != 0;
    private static bool TestMpeg1(byte[] b, int h) => (b[h + 1] & 0x8) != 0;
    private static bool TestNotMpeg25(byte[] b, int h) => (b[h + 1] & 0x10) != 0;
    private static bool TestIStereo(byte[] b, int h) => (b[h + 3] & 0x10) != 0;
    private static bool TestMsStereo(byte[] b, int h) => (b[h + 3] & 0x20) != 0;
    private static int GetLayer(byte[] b, int h) => (b[h + 1] >> 1) & 3;
    private static int GetBitrate(byte[] b, int h) => b[h + 2] >> 4;
    private static int GetSampleRate(byte[] b, int h) => (b[h + 2] >> 2) & 3;
    private static int GetMySampleRate(byte[] b, int h) => GetSampleRate(b, h) + (((b[h + 1] >> 3) & 1) + ((b[h + 1] >> 4) & 1)) * 3;
    private static bool IsFrame576(byte[] b, int h) => (b[h + 1] & 14) == 2;
    private static bool IsLayer1(byte[] b, int h) => (b[h + 1] & 6) == 6;

    private static bool HdrValid(byte[] b, int h)
        => b[h] == 0xff && ((b[h + 1] & 0xF0) == 0xf0 || (b[h + 1] & 0xFE) == 0xe2)
           && GetLayer(b, h) != 0 && GetBitrate(b, h) != 15 && GetSampleRate(b, h) != 3;

    private static bool HdrCompare(byte[] b, int h1, int h2)
        => HdrValid(b, h2) && ((b[h1 + 1] ^ b[h2 + 1]) & 0xFE) == 0 && ((b[h1 + 2] ^ b[h2 + 2]) & 0x0C) == 0
           && !(IsFreeFormat(b, h1) ^ IsFreeFormat(b, h2));

    private static readonly byte[,,] HalfRate =
    {
        { { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,4,8,12,16,20,24,28,32,40,48,56,64,72,80 }, { 0,16,24,28,32,40,48,56,64,72,80,88,96,112,128 } },
        { { 0,16,20,24,28,32,40,48,56,64,80,96,112,128,160 }, { 0,16,24,28,32,40,48,56,64,80,96,112,128,160,192 }, { 0,16,32,48,64,80,96,112,128,144,160,176,192,208,224 } },
    };

    private static int HdrBitrateKbps(byte[] b, int h) => 2 * HalfRate[TestMpeg1(b, h) ? 1 : 0, GetLayer(b, h) - 1, GetBitrate(b, h)];

    private static readonly int[] Hz = { 44100, 48000, 32000 };

    private static int HdrSampleRateHz(byte[] b, int h)
        => Hz[GetSampleRate(b, h)] >> (TestMpeg1(b, h) ? 0 : 1) >> (TestNotMpeg25(b, h) ? 0 : 1);

    private static int HdrFrameSamples(byte[] b, int h) => IsLayer1(b, h) ? 384 : (1152 >> (IsFrame576(b, h) ? 1 : 0));

    private static int HdrFrameBytes(byte[] b, int h, int freeFormatSize)
    {
        int frameBytes = HdrFrameSamples(b, h) * HdrBitrateKbps(b, h) * 125 / HdrSampleRateHz(b, h);
        if (IsLayer1(b, h)) frameBytes &= ~3;
        return frameBytes != 0 ? frameBytes : freeFormatSize;
    }

    private static int HdrPadding(byte[] b, int h) => TestPadding(b, h) ? (IsLayer1(b, h) ? 4 : 1) : 0;

    // ------------------------------------------------------------------ bit stream

    private sealed class BitStream
    {
        public byte[] Buf = Array.Empty<byte>();
        public int Start, Pos, Limit;

        public void Init(byte[] data, int start, int bytes)
        {
            Buf = data;
            Start = start;
            Pos = 0;
            Limit = bytes * 8;
        }

        public uint GetBits(int n)
        {
            uint next, cache = 0, s = (uint)(Pos & 7);
            int shl = n + (int)s;
            int p = Start + (Pos >> 3);
            if ((Pos += n) > Limit) return 0;
            next = (uint)(Byte(p++) & (255 >> (int)s));
            while ((shl -= 8) > 0)
            {
                cache |= next << shl;
                next = Byte(p++);
            }
            return cache | (next >> -shl);
        }

        public byte Byte(int i) => i < Buf.Length ? Buf[i] : (byte)0;
    }

    private sealed class GranuleInfo
    {
        public byte[] SfbTab = Array.Empty<byte>();
        public int Part23Length, BigValues, ScalefacCompress;
        public int GlobalGain, BlockType, MixedBlockFlag, NLongSfb, NShortSfb;
        public readonly int[] TableSelect = new int[3], RegionCount = new int[3], SubblockGain = new int[3];
        public int Preflag, ScalefacScale, Count1Table, Scfsi;
    }

    private sealed class State
    {
        public readonly float[] MdctOverlap = new float[2 * 9 * 32];
        public readonly float[] QmfState = new float[15 * 2 * 32];
        public int Reserv, FreeFormatBytes;
        public readonly byte[] Header = new byte[4];
        public readonly byte[] ReservBuf = new byte[511];
    }

    private sealed class Scratch
    {
        public readonly BitStream Bs = new();
        // Padded: the Huffman decoder reads a few bytes ahead of the bit position.
        public readonly byte[] MainData = new byte[MaxBitReservoirBytes + MaxL3FramePayloadBytes + 16];
        public readonly GranuleInfo[] GrInfo = { new(), new(), new(), new() };
        public readonly float[] GrBuf = new float[2 * 576];
        public readonly float[] Scf = new float[40];
        public readonly float[] Syn = new float[(18 + 15) * 2 * 32];
        public readonly byte[] IstPos = new byte[2 * 39];
    }

    private struct FrameInfo
    {
        public int FrameBytes, FrameOffset, Channels, Hz, Layer, BitrateKbps;
    }

    // ------------------------------------------------------------------ side information and scale factors

    private static readonly byte[][] ScfLong = new[] { new byte[] { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 }, new byte[] { 12,12,12,12,12,12,16,20,24,28,32,40,48,56,64,76,90,2,2,2,2,2,0 }, new byte[] { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 }, new byte[] { 6,6,6,6,6,6,8,10,12,14,16,18,22,26,32,38,46,54,62,70,76,36,0 }, new byte[] { 6,6,6,6,6,6,8,10,12,14,16,20,24,28,32,38,46,52,60,68,58,54,0 }, new byte[] { 4,4,4,4,4,4,6,6,8,8,10,12,16,20,24,28,34,42,50,54,76,158,0 }, new byte[] { 4,4,4,4,4,4,6,6,6,8,10,12,16,18,22,28,34,40,46,54,54,192,0 }, new byte[] { 4,4,4,4,4,4,6,6,8,10,12,16,20,24,30,38,46,56,68,84,102,26,0 } };
    private static readonly byte[][] ScfShort = new[] { new byte[] { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 }, new byte[] { 8,8,8,8,8,8,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 }, new byte[] { 4,4,4,4,4,4,4,4,4,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 } };
    private static readonly byte[][] ScfMixed = new[] { new byte[] { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 }, new byte[] { 12,12,12,4,4,4,8,8,8,12,12,12,16,16,16,20,20,20,24,24,24,28,28,28,36,36,36,2,2,2,2,2,2,2,2,2,26,26,26,0 }, new byte[] { 6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,14,14,14,18,18,18,26,26,26,32,32,32,42,42,42,18,18,18,0 }, new byte[] { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,32,32,32,44,44,44,12,12,12,0 }, new byte[] { 6,6,6,6,6,6,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,24,24,24,30,30,30,40,40,40,18,18,18,0 }, new byte[] { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,10,10,10,12,12,12,14,14,14,18,18,18,22,22,22,30,30,30,56,56,56,0 }, new byte[] { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,6,6,6,10,10,10,12,12,12,14,14,14,16,16,16,20,20,20,26,26,26,66,66,66,0 }, new byte[] { 4,4,4,4,4,4,6,6,4,4,4,6,6,6,8,8,8,12,12,12,16,16,16,20,20,20,26,26,26,34,34,34,42,42,42,12,12,12,0 } };

    private static int L3ReadSideInfo(BitStream bs, GranuleInfo[] grs, byte[] b, int h)
    {
        uint tables, scfsi = 0;
        int mainDataBegin, part23Sum = 0;
        int srIdx = GetMySampleRate(b, h);
        srIdx -= srIdx != 0 ? 1 : 0;
        int grCount = IsMono(b, h) ? 1 : 2;
        bool mpeg1 = TestMpeg1(b, h);

        if (mpeg1)
        {
            grCount *= 2;
            mainDataBegin = (int)bs.GetBits(9);
            scfsi = bs.GetBits(7 + grCount);
        }
        else
        {
            mainDataBegin = (int)(bs.GetBits(8 + grCount) >> grCount);
        }

        int g = 0;
        do
        {
            var gr = grs[g];
            if (IsMono(b, h)) scfsi <<= 4;
            gr.Part23Length = (int)bs.GetBits(12);
            part23Sum += gr.Part23Length;
            gr.BigValues = (int)bs.GetBits(9);
            if (gr.BigValues > 288) return -1;
            gr.GlobalGain = (int)bs.GetBits(8);
            gr.ScalefacCompress = (int)bs.GetBits(mpeg1 ? 4 : 9);
            gr.SfbTab = ScfLong[srIdx];
            gr.NLongSfb = 22;
            gr.NShortSfb = 0;
            if (bs.GetBits(1) != 0)
            {
                gr.BlockType = (int)bs.GetBits(2);
                if (gr.BlockType == 0) return -1;
                gr.MixedBlockFlag = (int)bs.GetBits(1);
                gr.RegionCount[0] = 7;
                gr.RegionCount[1] = 255;
                if (gr.BlockType == ShortBlockType)
                {
                    scfsi &= 0x0F0F;
                    if (gr.MixedBlockFlag == 0)
                    {
                        gr.RegionCount[0] = 8;
                        gr.SfbTab = ScfShort[srIdx];
                        gr.NLongSfb = 0;
                        gr.NShortSfb = 39;
                    }
                    else
                    {
                        gr.SfbTab = ScfMixed[srIdx];
                        gr.NLongSfb = mpeg1 ? 8 : 6;
                        gr.NShortSfb = 30;
                    }
                }
                tables = bs.GetBits(10);
                tables <<= 5;
                gr.SubblockGain[0] = (int)bs.GetBits(3);
                gr.SubblockGain[1] = (int)bs.GetBits(3);
                gr.SubblockGain[2] = (int)bs.GetBits(3);
            }
            else
            {
                gr.BlockType = 0;
                gr.MixedBlockFlag = 0;
                tables = bs.GetBits(15);
                gr.RegionCount[0] = (int)bs.GetBits(4);
                gr.RegionCount[1] = (int)bs.GetBits(3);
                gr.RegionCount[2] = 255;
            }
            gr.TableSelect[0] = (int)(tables >> 10);
            gr.TableSelect[1] = (int)((tables >> 5) & 31);
            gr.TableSelect[2] = (int)(tables & 31);
            gr.Preflag = mpeg1 ? (int)bs.GetBits(1) : (gr.ScalefacCompress >= 500 ? 1 : 0);
            gr.ScalefacScale = (int)bs.GetBits(1);
            gr.Count1Table = (int)bs.GetBits(1);
            gr.Scfsi = (int)((scfsi >> 12) & 15);
            scfsi <<= 4;
            g++;
        } while (--grCount > 0);

        if (part23Sum + bs.Pos > bs.Limit + mainDataBegin * 8) return -1;
        return mainDataBegin;
    }

    private static void L3ReadScalefactors(byte[] scf, byte[] istPos, int istOffset, byte[] scfSize, byte[] scfCount, int countOffset,
        BitStream bitbuf, int scfsi)
    {
        int s0 = 0, ip = istOffset;
        for (int i = 0; i < 4 && scfCount[countOffset + i] != 0; i++, scfsi *= 2)
        {
            int cnt = scfCount[countOffset + i];
            if ((scfsi & 8) != 0)
            {
                Array.Copy(istPos, ip, scf, s0, cnt);
            }
            else
            {
                int bits = scfSize[i];
                if (bits == 0)
                {
                    Array.Clear(scf, s0, cnt);
                    Array.Clear(istPos, ip, cnt);
                }
                else
                {
                    int maxScf = scfsi < 0 ? (1 << bits) - 1 : -1;
                    for (int k = 0; k < cnt; k++)
                    {
                        int s = (int)bitbuf.GetBits(bits);
                        istPos[ip + k] = (byte)(s == maxScf ? -1 : s);
                        scf[s0 + k] = (byte)s;
                    }
                }
            }
            ip += cnt;
            s0 += cnt;
        }
        scf[s0] = scf[s0 + 1] = scf[s0 + 2] = 0;
    }

    private static readonly float[] ExpFrac = { 9.31322575e-10f, 7.83145814e-10f, 6.58544508e-10f, 5.53767716e-10f };

    private static float L3LdexpQ2(float y, int expQ2)
    {
        int e;
        do
        {
            e = Math.Min(30 * 4, expQ2);
            y *= ExpFrac[e & 3] * (1 << 30 >> (e >> 2));
        } while ((expQ2 -= e) > 0);
        return y;
    }

    private static readonly byte[][] ScfPartitions = new[] { new byte[] { 6,5,5, 5,6,5,5,5,6,5, 7,3,11,10,0,0, 7, 7, 7,0, 6, 6,6,3, 8, 8,5,0 }, new byte[] { 8,9,6,12,6,9,9,9,6,9,12,6,15,18,0,0, 6,15,12,0, 6,12,9,6, 6,18,9,0 }, new byte[] { 9,9,6,12,9,9,9,9,9,9,12,6,18,18,0,0,12,12,12,0,12, 9,9,6,15,12,9,0 } };
    private static readonly byte[] ScfcDecode = { 0,1,2,3, 12,5,6,7, 9,10,11,13, 14,15,18,19 };
    private static readonly byte[] Mod = { 5,5,4,4,5,5,4,1,4,3,1,1,5,6,6,1,4,4,4,1,4,3,1,1 };
    private static readonly byte[] Preamp = { 1,1,1,1,2,2,3,3,3,2 };

    private static void L3DecodeScalefactors(byte[] b, int h, byte[] istPos, int istOffset, BitStream bs, GranuleInfo gr, float[] scf, int ch)
    {
        var scfPartition = ScfPartitions[(gr.NShortSfb != 0 ? 1 : 0) + (gr.NLongSfb == 0 ? 1 : 0)];
        int partitionOffset = 0;
        var scfSize = new byte[4];
        var iscf = new byte[40];
        int scfShift = gr.ScalefacScale + 1, scfsi = gr.Scfsi;

        if (TestMpeg1(b, h))
        {
            int part = ScfcDecode[gr.ScalefacCompress];
            scfSize[1] = scfSize[0] = (byte)(part >> 2);
            scfSize[3] = scfSize[2] = (byte)(part & 3);
        }
        else
        {
            int k, modprod, sfc, ist = TestIStereo(b, h) && ch != 0 ? 1 : 0;
            sfc = gr.ScalefacCompress >> ist;
            for (k = ist * 3 * 4; sfc >= 0; sfc -= modprod, k += 4)
            {
                modprod = 1;
                for (int i = 3; i >= 0; i--)
                {
                    scfSize[i] = (byte)(sfc / modprod % Mod[k + i]);
                    modprod *= Mod[k + i];
                }
            }
            partitionOffset = k;
            scfsi = -16;
        }
        L3ReadScalefactors(iscf, istPos, istOffset, scfSize, scfPartition, partitionOffset, bs, scfsi);

        if (gr.NShortSfb != 0)
        {
            int sh = 3 - scfShift;
            for (int i = 0; i < gr.NShortSfb; i += 3)
            {
                iscf[gr.NLongSfb + i + 0] = (byte)(iscf[gr.NLongSfb + i + 0] + (gr.SubblockGain[0] << sh));
                iscf[gr.NLongSfb + i + 1] = (byte)(iscf[gr.NLongSfb + i + 1] + (gr.SubblockGain[1] << sh));
                iscf[gr.NLongSfb + i + 2] = (byte)(iscf[gr.NLongSfb + i + 2] + (gr.SubblockGain[2] << sh));
            }
        }
        else if (gr.Preflag != 0)
        {
            for (int i = 0; i < 10; i++) iscf[11 + i] = (byte)(iscf[11 + i] + Preamp[i]);
        }

        int gainExp = gr.GlobalGain + BitsDequantizerOut * 4 - 210 - (IsMsStereo(b, h) ? 2 : 0);
        float gain = L3LdexpQ2(1 << (MaxScfi / 4), MaxScfi - gainExp);
        for (int i = 0; i < gr.NLongSfb + gr.NShortSfb; i++) scf[i] = L3LdexpQ2(gain, iscf[i] << scfShift);
    }

    private static readonly float[] Pow43 = { 0,-1,-2.519842f,-4.326749f,-6.349604f,-8.549880f,-10.902724f,-13.390518f,-16.000000f,-18.720754f,-21.544347f,-24.463781f,-27.473142f,-30.567351f,-33.741992f,-36.993181f, 0,1,2.519842f,4.326749f,6.349604f,8.549880f,10.902724f,13.390518f,16.000000f,18.720754f,21.544347f,24.463781f,27.473142f,30.567351f,33.741992f,36.993181f,40.317474f,43.711787f,47.173345f,50.699631f,54.288352f,57.937408f,61.644865f,65.408941f,69.227979f,73.100443f,77.024898f,81.000000f,85.024491f,89.097188f,93.216975f,97.382800f,101.593667f,105.848633f,110.146801f,114.487321f,118.869381f,123.292209f,127.755065f,132.257246f,136.798076f,141.376907f,145.993119f,150.646117f,155.335327f,160.060199f,164.820202f,169.614826f,174.443577f,179.305980f,184.201575f,189.129918f,194.090580f,199.083145f,204.107210f,209.162385f,214.248292f,219.364564f,224.510845f,229.686789f,234.892058f,240.126328f,245.389280f,250.680604f,256.000000f,261.347174f,266.721841f,272.123723f,277.552547f,283.008049f,288.489971f,293.998060f,299.532071f,305.091761f,310.676898f,316.287249f,321.922592f,327.582707f,333.267377f,338.976394f,344.709550f,350.466646f,356.247482f,362.051866f,367.879608f,373.730522f,379.604427f,385.501143f,391.420496f,397.362314f,403.326427f,409.312672f,415.320884f,421.350905f,427.402579f,433.475750f,439.570269f,445.685987f,451.822757f,457.980436f,464.158883f,470.357960f,476.577530f,482.817459f,489.077615f,495.357868f,501.658090f,507.978156f,514.317941f,520.677324f,527.056184f,533.454404f,539.871867f,546.308458f,552.764065f,559.238575f,565.731879f,572.243870f,578.774440f,585.323483f,591.890898f,598.476581f,605.080431f,611.702349f,618.342238f,625.000000f,631.675540f,638.368763f,645.079578f };

    private static float L3Pow43(int x)
    {
        float frac;
        int sign, mult = 256;
        if (x < 129) return Pow43[16 + x];
        if (x < 1024)
        {
            mult = 16;
            x <<= 3;
        }
        sign = 2 * x & 64;
        frac = (float)((x & 63) - sign) / ((x & ~63) + sign);
        return Pow43[16 + ((x + sign) >> 6)] * (1f + frac * ((4f / 3) + frac * (2f / 9))) * mult;
    }

    // ------------------------------------------------------------------ Huffman decoding

    private static readonly short[] Tabs = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 785,785,785,785,784,784,784,784,513,513,513,513,513,513,513,513,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256, -255,1313,1298,1282,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,290,288, -255,1313,1298,1282,769,769,769,769,529,529,529,529,529,529,529,529,528,528,528,528,528,528,528,528,512,512,512,512,512,512,512,512,290,288, -253,-318,-351,-367,785,785,785,785,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,819,818,547,547,275,275,275,275,561,560,515,546,289,274,288,258, -254,-287,1329,1299,1314,1312,1057,1057,1042,1042,1026,1026,784,784,784,784,529,529,529,529,529,529,529,529,769,769,769,769,768,768,768,768,563,560,306,306,291,259, -252,-413,-477,-542,1298,-575,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-383,-399,1107,1092,1106,1061,849,849,789,789,1104,1091,773,773,1076,1075,341,340,325,309,834,804,577,577,532,532,516,516,832,818,803,816,561,561,531,531,515,546,289,289,288,258, -252,-429,-493,-559,1057,1057,1042,1042,529,529,529,529,529,529,529,529,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,-382,1077,-415,1106,1061,1104,849,849,789,789,1091,1076,1029,1075,834,834,597,581,340,340,339,324,804,833,532,532,832,772,818,803,817,787,816,771,290,290,290,290,288,258, -253,-349,-414,-447,-463,1329,1299,-479,1314,1312,1057,1057,1042,1042,1026,1026,785,785,785,785,784,784,784,784,769,769,769,769,768,768,768,768,-319,851,821,-335,836,850,805,849,341,340,325,336,533,533,579,579,564,564,773,832,578,548,563,516,321,276,306,291,304,259, -251,-572,-733,-830,-863,-879,1041,1041,784,784,784,784,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,1396,1351,1381,1366,1395,1335,1380,-559,1334,1138,1138,1063,1063,1350,1392,1031,1031,1062,1062,1364,1363,1120,1120,1333,1348,881,881,881,881,375,374,359,373,343,358,341,325,791,791,1123,1122,-703,1105,1045,-719,865,865,790,790,774,774,1104,1029,338,293,323,308,-799,-815,833,788,772,818,803,816,322,292,307,320,561,531,515,546,289,274,288,258, -251,-525,-605,-685,-765,-831,-846,1298,1057,1057,1312,1282,785,785,785,785,784,784,784,784,769,769,769,769,512,512,512,512,512,512,512,512,1399,1398,1383,1367,1382,1396,1351,-511,1381,1366,1139,1139,1079,1079,1124,1124,1364,1349,1363,1333,882,882,882,882,807,807,807,807,1094,1094,1136,1136,373,341,535,535,881,775,867,822,774,-591,324,338,-671,849,550,550,866,864,609,609,293,336,534,534,789,835,773,-751,834,804,308,307,833,788,832,772,562,562,547,547,305,275,560,515,290,290, -252,-397,-477,-557,-622,-653,-719,-735,-750,1329,1299,1314,1057,1057,1042,1042,1312,1282,1024,1024,785,785,785,785,784,784,784,784,769,769,769,769,-383,1127,1141,1111,1126,1140,1095,1110,869,869,883,883,1079,1109,882,882,375,374,807,868,838,881,791,-463,867,822,368,263,852,837,836,-543,610,610,550,550,352,336,534,534,865,774,851,821,850,805,593,533,579,564,773,832,578,578,548,548,577,577,307,276,306,291,516,560,259,259, -250,-2107,-2507,-2764,-2909,-2974,-3007,-3023,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-767,-1052,-1213,-1277,-1358,-1405,-1469,-1535,-1550,-1582,-1614,-1647,-1662,-1694,-1726,-1759,-1774,-1807,-1822,-1854,-1886,1565,-1919,-1935,-1951,-1967,1731,1730,1580,1717,-1983,1729,1564,-1999,1548,-2015,-2031,1715,1595,-2047,1714,-2063,1610,-2079,1609,-2095,1323,1323,1457,1457,1307,1307,1712,1547,1641,1700,1699,1594,1685,1625,1442,1442,1322,1322,-780,-973,-910,1279,1278,1277,1262,1276,1261,1275,1215,1260,1229,-959,974,974,989,989,-943,735,478,478,495,463,506,414,-1039,1003,958,1017,927,942,987,957,431,476,1272,1167,1228,-1183,1256,-1199,895,895,941,941,1242,1227,1212,1135,1014,1014,490,489,503,487,910,1013,985,925,863,894,970,955,1012,847,-1343,831,755,755,984,909,428,366,754,559,-1391,752,486,457,924,997,698,698,983,893,740,740,908,877,739,739,667,667,953,938,497,287,271,271,683,606,590,712,726,574,302,302,738,736,481,286,526,725,605,711,636,724,696,651,589,681,666,710,364,467,573,695,466,466,301,465,379,379,709,604,665,679,316,316,634,633,436,436,464,269,424,394,452,332,438,363,347,408,393,448,331,422,362,407,392,421,346,406,391,376,375,359,1441,1306,-2367,1290,-2383,1337,-2399,-2415,1426,1321,-2431,1411,1336,-2447,-2463,-2479,1169,1169,1049,1049,1424,1289,1412,1352,1319,-2495,1154,1154,1064,1064,1153,1153,416,390,360,404,403,389,344,374,373,343,358,372,327,357,342,311,356,326,1395,1394,1137,1137,1047,1047,1365,1392,1287,1379,1334,1364,1349,1378,1318,1363,792,792,792,792,1152,1152,1032,1032,1121,1121,1046,1046,1120,1120,1030,1030,-2895,1106,1061,1104,849,849,789,789,1091,1076,1029,1090,1060,1075,833,833,309,324,532,532,832,772,818,803,561,561,531,560,515,546,289,274,288,258, -250,-1179,-1579,-1836,-1996,-2124,-2253,-2333,-2413,-2477,-2542,-2574,-2607,-2622,-2655,1314,1313,1298,1312,1282,785,785,785,785,1040,1040,1025,1025,768,768,768,768,-766,-798,-830,-862,-895,-911,-927,-943,-959,-975,-991,-1007,-1023,-1039,-1055,-1070,1724,1647,-1103,-1119,1631,1767,1662,1738,1708,1723,-1135,1780,1615,1779,1599,1677,1646,1778,1583,-1151,1777,1567,1737,1692,1765,1722,1707,1630,1751,1661,1764,1614,1736,1676,1763,1750,1645,1598,1721,1691,1762,1706,1582,1761,1566,-1167,1749,1629,767,766,751,765,494,494,735,764,719,749,734,763,447,447,748,718,477,506,431,491,446,476,461,505,415,430,475,445,504,399,460,489,414,503,383,474,429,459,502,502,746,752,488,398,501,473,413,472,486,271,480,270,-1439,-1455,1357,-1471,-1487,-1503,1341,1325,-1519,1489,1463,1403,1309,-1535,1372,1448,1418,1476,1356,1462,1387,-1551,1475,1340,1447,1402,1386,-1567,1068,1068,1474,1461,455,380,468,440,395,425,410,454,364,467,466,464,453,269,409,448,268,432,1371,1473,1432,1417,1308,1460,1355,1446,1459,1431,1083,1083,1401,1416,1458,1445,1067,1067,1370,1457,1051,1051,1291,1430,1385,1444,1354,1415,1400,1443,1082,1082,1173,1113,1186,1066,1185,1050,-1967,1158,1128,1172,1097,1171,1081,-1983,1157,1112,416,266,375,400,1170,1142,1127,1065,793,793,1169,1033,1156,1096,1141,1111,1155,1080,1126,1140,898,898,808,808,897,897,792,792,1095,1152,1032,1125,1110,1139,1079,1124,882,807,838,881,853,791,-2319,867,368,263,822,852,837,866,806,865,-2399,851,352,262,534,534,821,836,594,594,549,549,593,593,533,533,848,773,579,579,564,578,548,563,276,276,577,576,306,291,516,560,305,305,275,259, -251,-892,-2058,-2620,-2828,-2957,-3023,-3039,1041,1041,1040,1040,769,769,769,769,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,256,-511,-527,-543,-559,1530,-575,-591,1528,1527,1407,1526,1391,1023,1023,1023,1023,1525,1375,1268,1268,1103,1103,1087,1087,1039,1039,1523,-604,815,815,815,815,510,495,509,479,508,463,507,447,431,505,415,399,-734,-782,1262,-815,1259,1244,-831,1258,1228,-847,-863,1196,-879,1253,987,987,748,-767,493,493,462,477,414,414,686,669,478,446,461,445,474,429,487,458,412,471,1266,1264,1009,1009,799,799,-1019,-1276,-1452,-1581,-1677,-1757,-1821,-1886,-1933,-1997,1257,1257,1483,1468,1512,1422,1497,1406,1467,1496,1421,1510,1134,1134,1225,1225,1466,1451,1374,1405,1252,1252,1358,1480,1164,1164,1251,1251,1238,1238,1389,1465,-1407,1054,1101,-1423,1207,-1439,830,830,1248,1038,1237,1117,1223,1148,1236,1208,411,426,395,410,379,269,1193,1222,1132,1235,1221,1116,976,976,1192,1162,1177,1220,1131,1191,963,963,-1647,961,780,-1663,558,558,994,993,437,408,393,407,829,978,813,797,947,-1743,721,721,377,392,844,950,828,890,706,706,812,859,796,960,948,843,934,874,571,571,-1919,690,555,689,421,346,539,539,944,779,918,873,932,842,903,888,570,570,931,917,674,674,-2575,1562,-2591,1609,-2607,1654,1322,1322,1441,1441,1696,1546,1683,1593,1669,1624,1426,1426,1321,1321,1639,1680,1425,1425,1305,1305,1545,1668,1608,1623,1667,1592,1638,1666,1320,1320,1652,1607,1409,1409,1304,1304,1288,1288,1664,1637,1395,1395,1335,1335,1622,1636,1394,1394,1319,1319,1606,1621,1392,1392,1137,1137,1137,1137,345,390,360,375,404,373,1047,-2751,-2767,-2783,1062,1121,1046,-2799,1077,-2815,1106,1061,789,789,1105,1104,263,355,310,340,325,354,352,262,339,324,1091,1076,1029,1090,1060,1075,833,833,788,788,1088,1028,818,818,803,803,561,561,531,531,816,771,546,546,289,274,288,258, -253,-317,-381,-446,-478,-509,1279,1279,-811,-1179,-1451,-1756,-1900,-2028,-2189,-2253,-2333,-2414,-2445,-2511,-2526,1313,1298,-2559,1041,1041,1040,1040,1025,1025,1024,1024,1022,1007,1021,991,1020,975,1019,959,687,687,1018,1017,671,671,655,655,1016,1015,639,639,758,758,623,623,757,607,756,591,755,575,754,559,543,543,1009,783,-575,-621,-685,-749,496,-590,750,749,734,748,974,989,1003,958,988,973,1002,942,987,957,972,1001,926,986,941,971,956,1000,910,985,925,999,894,970,-1071,-1087,-1102,1390,-1135,1436,1509,1451,1374,-1151,1405,1358,1480,1420,-1167,1507,1494,1389,1342,1465,1435,1450,1326,1505,1310,1493,1373,1479,1404,1492,1464,1419,428,443,472,397,736,526,464,464,486,457,442,471,484,482,1357,1449,1434,1478,1388,1491,1341,1490,1325,1489,1463,1403,1309,1477,1372,1448,1418,1433,1476,1356,1462,1387,-1439,1475,1340,1447,1402,1474,1324,1461,1371,1473,269,448,1432,1417,1308,1460,-1711,1459,-1727,1441,1099,1099,1446,1386,1431,1401,-1743,1289,1083,1083,1160,1160,1458,1445,1067,1067,1370,1457,1307,1430,1129,1129,1098,1098,268,432,267,416,266,400,-1887,1144,1187,1082,1173,1113,1186,1066,1050,1158,1128,1143,1172,1097,1171,1081,420,391,1157,1112,1170,1142,1127,1065,1169,1049,1156,1096,1141,1111,1155,1080,1126,1154,1064,1153,1140,1095,1048,-2159,1125,1110,1137,-2175,823,823,1139,1138,807,807,384,264,368,263,868,838,853,791,867,822,852,837,866,806,865,790,-2319,851,821,836,352,262,850,805,849,-2399,533,533,835,820,336,261,578,548,563,577,532,532,832,772,562,562,547,547,305,275,560,515,290,290,288,258 };
    private static readonly byte[] Tab32 = { 130,162,193,209,44,28,76,140,9,9,9,9,9,9,9,9,190,254,222,238,126,94,157,157,109,61,173,205 };
    private static readonly byte[] Tab33 = { 252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12 };
    private static readonly short[] TabIndex = { 0,32,64,98,0,132,180,218,292,364,426,538,648,746,0,1126,1460,1460,1460,1460,1460,1460,1460,1460,1842,1842,1842,1842,1842,1842,1842,1842 };
    private static readonly byte[] LinBits = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,6,8,10,13,4,5,6,7,8,9,11,13 };

    private static void L3Huffman(float[] dst, int dstOffset, BitStream bs, GranuleInfo grInfo, float[] scf, int layer3grLimit)
    {
        float one = 0f;
        int ireg = 0, bigValCnt = grInfo.BigValues;
        var sfbTab = grInfo.SfbTab;
        int sfb = 0, scfIdx = 0, d = dstOffset;
        var buf = bs.Buf;
        int next = bs.Start + bs.Pos / 8;
        uint cache = (((uint)bs.Byte(next) * 256u + bs.Byte(next + 1)) * 256u + bs.Byte(next + 2)) * 256u + bs.Byte(next + 3);
        cache <<= bs.Pos & 7;
        int pairsToDecode, np, sh = (bs.Pos & 7) - 8;
        next += 4;

        uint Peek(int n) => cache >> (32 - n);
        void Flush(int n)
        {
            cache <<= n;
            sh += n;
        }
        void Check()
        {
            while (sh >= 0)
            {
                cache |= (uint)bs.Byte(next++) << sh;
                sh -= 8;
            }
        }
        int BsPos() => (next - bs.Start) * 8 - 24 + sh;

        while (bigValCnt > 0)
        {
            int tabNum = grInfo.TableSelect[ireg];
            int sfbCnt = grInfo.RegionCount[ireg++];
            int codebook = TabIndex[tabNum];
            int linbits = LinBits[tabNum];
            do
            {
                np = sfbTab[sfb++] / 2;
                pairsToDecode = Math.Min(bigValCnt, np);
                one = scf[scfIdx++];
                do
                {
                    int w = 5;
                    int leaf = Tabs[codebook + (int)Peek(w)];
                    while (leaf < 0)
                    {
                        Flush(w);
                        w = leaf & 7;
                        leaf = Tabs[codebook + (int)Peek(w) - (leaf >> 3)];
                    }
                    Flush(leaf >> 8);

                    for (int j = 0; j < 2; j++, d++, leaf >>= 4)
                    {
                        int lsb = leaf & 0x0F;
                        if (linbits != 0 && lsb == 15)
                        {
                            lsb += (int)Peek(linbits);
                            Flush(linbits);
                            Check();
                            dst[d] = one * L3Pow43(lsb) * ((int)cache < 0 ? -1 : 1);
                        }
                        else
                        {
                            dst[d] = Pow43[16 + lsb - 16 * (int)(cache >> 31)] * one;
                        }
                        Flush(lsb != 0 ? 1 : 0);
                    }
                    Check();
                } while (--pairsToDecode > 0);
            } while ((bigValCnt -= np) > 0 && --sfbCnt >= 0);
        }

        for (np = 1 - bigValCnt; ; d += 4)
        {
            var codebookCount1 = grInfo.Count1Table != 0 ? Tab33 : Tab32;
            int leaf = codebookCount1[Peek(4)];
            if ((leaf & 8) == 0)
                leaf = codebookCount1[(leaf >> 3) + (int)(cache << 4 >> (32 - (leaf & 3)))];
            Flush(leaf & 7);
            if (BsPos() > layer3grLimit) break;

            if (--np == 0)
            {
                np = sfbTab[sfb++] / 2;
                if (np == 0) break;
                one = scf[scfIdx++];
            }
            if ((leaf & (128 >> 0)) != 0) { dst[d + 0] = (int)cache < 0 ? -one : one; Flush(1); }
            if ((leaf & (128 >> 1)) != 0) { dst[d + 1] = (int)cache < 0 ? -one : one; Flush(1); }
            if (--np == 0)
            {
                np = sfbTab[sfb++] / 2;
                if (np == 0) break;
                one = scf[scfIdx++];
            }
            if ((leaf & (128 >> 2)) != 0) { dst[d + 2] = (int)cache < 0 ? -one : one; Flush(1); }
            if ((leaf & (128 >> 3)) != 0) { dst[d + 3] = (int)cache < 0 ? -one : one; Flush(1); }
            Check();
        }
        bs.Pos = layer3grLimit;
    }

    // ------------------------------------------------------------------ stereo

    private static void L3MidsideStereo(float[] buf, int left, int n)
    {
        int right = left + 576;
        for (int i = 0; i < n; i++)
        {
            float a = buf[left + i], b = buf[right + i];
            buf[left + i] = a + b;
            buf[right + i] = a - b;
        }
    }

    private static void L3IntensityStereoBand(float[] buf, int left, int n, float kl, float kr)
    {
        for (int i = 0; i < n; i++)
        {
            buf[left + i + 576] = buf[left + i] * kr;
            buf[left + i] = buf[left + i] * kl;
        }
    }

    private static void L3StereoTopBand(float[] buf, int right, byte[] sfb, int nbands, int[] maxBand)
    {
        maxBand[0] = maxBand[1] = maxBand[2] = -1;
        for (int i = 0; i < nbands; i++)
        {
            for (int k = 0; k < sfb[i]; k += 2)
            {
                if (buf[right + k] != 0 || buf[right + k + 1] != 0)
                {
                    maxBand[i % 3] = i;
                    break;
                }
            }
            right += sfb[i];
        }
    }

    private static readonly float[] Pan = { 0,1,0.21132487f,0.78867513f,0.36602540f,0.63397460f,0.5f,0.5f,0.63397460f,0.36602540f,0.78867513f,0.21132487f,1,0 };

    private static void L3StereoProcess(float[] buf, int left, byte[] istPos, int istOffset, byte[] sfb, byte[] b, int h, int[] maxBand, int mpeg2Sh)
    {
        uint maxPos = TestMpeg1(b, h) ? 7u : 64u;
        for (int i = 0; sfb[i] != 0; i++)
        {
            uint ipos = istPos[istOffset + i];
            if (i > maxBand[i % 3] && ipos < maxPos)
            {
                float kl, kr, s = TestMsStereo(b, h) ? 1.41421356f : 1;
                if (TestMpeg1(b, h))
                {
                    kl = Pan[2 * ipos];
                    kr = Pan[2 * ipos + 1];
                }
                else
                {
                    kl = 1;
                    kr = L3LdexpQ2(1, (int)((ipos + 1) >> 1 << mpeg2Sh));
                    if ((ipos & 1) != 0)
                    {
                        kl = kr;
                        kr = 1;
                    }
                }
                L3IntensityStereoBand(buf, left, sfb[i], kl * s, kr * s);
            }
            else if (TestMsStereo(b, h))
            {
                L3MidsideStereo(buf, left, sfb[i]);
            }
            left += sfb[i];
        }
    }

    private static void L3IntensityStereo(float[] buf, byte[] istPos, int istOffset, GranuleInfo[] grs, int g, byte[] b, int h)
    {
        var gr = grs[g];
        var maxBand = new int[3];
        int nSfb = gr.NLongSfb + gr.NShortSfb;
        int maxBlocks = gr.NShortSfb != 0 ? 3 : 1;
        L3StereoTopBand(buf, 576, gr.SfbTab, nSfb, maxBand);
        if (gr.NLongSfb != 0)
            maxBand[0] = maxBand[1] = maxBand[2] = Math.Max(Math.Max(maxBand[0], maxBand[1]), maxBand[2]);
        for (int i = 0; i < maxBlocks; i++)
        {
            int defaultPos = TestMpeg1(b, h) ? 3 : 0;
            int itop = nSfb - maxBlocks + i;
            int prev = itop - maxBlocks;
            istPos[istOffset + itop] = (byte)(maxBand[i] >= prev ? defaultPos : istPos[istOffset + prev]);
        }
        L3StereoProcess(buf, 0, istPos, istOffset, gr.SfbTab, b, h, maxBand, grs[g + 1].ScalefacCompress & 1);
    }

    // ------------------------------------------------------------------ reorder, alias reduction, IMDCT

    private static void L3Reorder(float[] buf, int grbuf, float[] scratch, byte[] sfb, int sfbOffset)
    {
        int src = grbuf, dst = 0, len;
        for (; (len = sfb[sfbOffset]) != 0; sfbOffset += 3, src += 2 * len)
        {
            for (int i = 0; i < len; i++, src++)
            {
                scratch[dst++] = buf[src + 0 * len];
                scratch[dst++] = buf[src + 1 * len];
                scratch[dst++] = buf[src + 2 * len];
            }
        }
        Array.Copy(scratch, 0, buf, grbuf, dst);
    }

    private static readonly float[][] Aa =
    {
        new[] { 0.85749293f, 0.88174200f, 0.94962865f, 0.98331459f, 0.99551782f, 0.99916056f, 0.99989920f, 0.99999316f },
        new[] { 0.51449576f, 0.47173197f, 0.31337745f, 0.18191320f, 0.09457419f, 0.04096558f, 0.01419856f, 0.00369997f },
    };

    private static void L3Antialias(float[] buf, int grbuf, int nbands)
    {
        for (; nbands > 0; nbands--, grbuf += 18)
        {
            for (int i = 0; i < 8; i++)
            {
                float u = buf[grbuf + 18 + i];
                float d = buf[grbuf + 17 - i];
                buf[grbuf + 18 + i] = u * Aa[0][i] - d * Aa[1][i];
                buf[grbuf + 17 - i] = u * Aa[1][i] + d * Aa[0][i];
            }
        }
    }

    private static void L3Dct3_9(float[] y)
    {
        float s0, s1, s2, s3, s4, s5, s6, s7, s8, t0, t2, t4;
        s0 = y[0]; s2 = y[2]; s4 = y[4]; s6 = y[6]; s8 = y[8];
        t0 = s0 + s6 * 0.5f;
        s0 -= s6;
        t4 = (s4 + s2) * 0.93969262f;
        t2 = (s8 + s2) * 0.76604444f;
        s6 = (s4 - s8) * 0.17364818f;
        s4 += s8 - s2;

        s2 = s0 - s4 * 0.5f;
        y[4] = s4 + s0;
        s8 = t0 - t2 + s6;
        s0 = t0 - t4 + t2;
        s4 = t0 + t4 - s6;

        s1 = y[1]; s3 = y[3]; s5 = y[5]; s7 = y[7];

        s3 *= 0.86602540f;
        t0 = (s5 + s1) * 0.98480775f;
        t4 = (s5 - s7) * 0.34202014f;
        t2 = (s1 + s7) * 0.64278761f;
        s1 = (s1 - s5 - s7) * 0.86602540f;

        s5 = t0 - s3 - t2;
        s7 = t4 - s3 - t0;
        s3 = t4 + s3 - t2;

        y[0] = s4 - s7;
        y[1] = s2 + s1;
        y[2] = s0 - s3;
        y[3] = s8 + s5;
        y[5] = s8 - s5;
        y[6] = s0 + s3;
        y[7] = s2 - s1;
        y[8] = s4 + s7;
    }

    private static readonly float[] Twid9 = { 0.73727734f,0.79335334f,0.84339145f,0.88701083f,0.92387953f,0.95371695f,0.97629601f,0.99144486f,0.99904822f,0.67559021f,0.60876143f,0.53729961f,0.46174861f,0.38268343f,0.30070580f,0.21643961f,0.13052619f,0.04361938f };

    private static void L3Imdct36(float[] buf, int grbuf, float[] overlapBuf, int overlap, float[] window, int nbands)
    {
        var co = new float[9];
        var si = new float[9];
        for (int j = 0; j < nbands; j++, grbuf += 18, overlap += 9)
        {
            co[0] = -buf[grbuf + 0];
            si[0] = buf[grbuf + 17];
            for (int i = 0; i < 4; i++)
            {
                si[8 - 2 * i] = buf[grbuf + 4 * i + 1] - buf[grbuf + 4 * i + 2];
                co[1 + 2 * i] = buf[grbuf + 4 * i + 1] + buf[grbuf + 4 * i + 2];
                si[7 - 2 * i] = buf[grbuf + 4 * i + 4] - buf[grbuf + 4 * i + 3];
                co[2 + 2 * i] = -(buf[grbuf + 4 * i + 3] + buf[grbuf + 4 * i + 4]);
            }
            L3Dct3_9(co);
            L3Dct3_9(si);
            si[1] = -si[1];
            si[3] = -si[3];
            si[5] = -si[5];
            si[7] = -si[7];
            for (int i = 0; i < 9; i++)
            {
                float ovl = overlapBuf[overlap + i];
                float sum = co[i] * Twid9[9 + i] + si[i] * Twid9[0 + i];
                overlapBuf[overlap + i] = co[i] * Twid9[0 + i] - si[i] * Twid9[9 + i];
                buf[grbuf + i] = ovl * window[0 + i] - sum * window[9 + i];
                buf[grbuf + 17 - i] = ovl * window[9 + i] + sum * window[0 + i];
            }
        }
    }

    private static void L3Idct3(float x0, float x1, float x2, float[] dst)
    {
        float m1 = x1 * 0.86602540f;
        float a1 = x0 - x2 * 0.5f;
        dst[1] = x0 + x2;
        dst[0] = a1 + m1;
        dst[2] = a1 - m1;
    }

    private static readonly float[] Twid3 = { 0.79335334f, 0.92387953f, 0.99144486f, 0.60876143f, 0.38268343f, 0.13052619f };

    private static void L3Imdct12(float[] x, int xi, float[] dstBuf, int dst, float[] overlapBuf, int overlap)
    {
        var co = new float[3];
        var si = new float[3];
        L3Idct3(-x[xi + 0], x[xi + 6] + x[xi + 3], x[xi + 12] + x[xi + 9], co);
        L3Idct3(x[xi + 15], x[xi + 12] - x[xi + 9], x[xi + 6] - x[xi + 3], si);
        si[1] = -si[1];
        for (int i = 0; i < 3; i++)
        {
            float ovl = overlapBuf[overlap + i];
            float sum = co[i] * Twid3[3 + i] + si[i] * Twid3[0 + i];
            overlapBuf[overlap + i] = co[i] * Twid3[0 + i] - si[i] * Twid3[3 + i];
            dstBuf[dst + i] = ovl * Twid3[2 - i] - sum * Twid3[5 - i];
            dstBuf[dst + 5 - i] = ovl * Twid3[5 - i] + sum * Twid3[2 - i];
        }
    }

    private static void L3ImdctShort(float[] buf, int grbuf, float[] overlapBuf, int overlap, int nbands)
    {
        var tmp = new float[18];
        for (; nbands > 0; nbands--, overlap += 9, grbuf += 18)
        {
            Array.Copy(buf, grbuf, tmp, 0, 18);
            Array.Copy(overlapBuf, overlap, buf, grbuf, 6);
            L3Imdct12(tmp, 0, buf, grbuf + 6, overlapBuf, overlap + 6);
            L3Imdct12(tmp, 1, buf, grbuf + 12, overlapBuf, overlap + 6);
            L3Imdct12(tmp, 2, overlapBuf, overlap, overlapBuf, overlap + 6);
        }
    }

    private static void L3ChangeSign(float[] buf, int grbuf)
    {
        grbuf += 18;
        for (int b = 0; b < 32; b += 2, grbuf += 36)
            for (int i = 1; i < 18; i += 2)
                buf[grbuf + i] = -buf[grbuf + i];
    }

    private static readonly float[][] MdctWindow =
    {
        new[] { 0.99904822f, 0.99144486f, 0.97629601f, 0.95371695f, 0.92387953f, 0.88701083f, 0.84339145f, 0.79335334f, 0.73727734f,
                0.04361938f, 0.13052619f, 0.21643961f, 0.30070580f, 0.38268343f, 0.46174861f, 0.53729961f, 0.60876143f, 0.67559021f },
        new[] { 1f, 1f, 1f, 1f, 1f, 1f, 0.99144486f, 0.92387953f, 0.79335334f, 0f, 0f, 0f, 0f, 0f, 0f, 0.13052619f, 0.38268343f, 0.60876143f },
    };

    private static void L3ImdctGr(float[] buf, int grbuf, float[] overlapBuf, int overlap, int blockType, int nLongBands)
    {
        if (nLongBands != 0)
        {
            L3Imdct36(buf, grbuf, overlapBuf, overlap, MdctWindow[0], nLongBands);
            grbuf += 18 * nLongBands;
            overlap += 9 * nLongBands;
        }
        if (blockType == ShortBlockType)
            L3ImdctShort(buf, grbuf, overlapBuf, overlap, 32 - nLongBands);
        else
            L3Imdct36(buf, grbuf, overlapBuf, overlap, MdctWindow[blockType == StopBlockType ? 1 : 0], 32 - nLongBands);
    }

    // ------------------------------------------------------------------ bit reservoir

    private static void L3SaveReservoir(State h, Scratch s)
    {
        int pos = (s.Bs.Pos + 7) / 8;
        int remains = s.Bs.Limit / 8 - pos;
        if (remains > MaxBitReservoirBytes)
        {
            pos += remains - MaxBitReservoirBytes;
            remains = MaxBitReservoirBytes;
        }
        if (remains > 0) Array.Copy(s.MainData, pos, h.ReservBuf, 0, remains);
        h.Reserv = remains;
    }

    private static bool L3RestoreReservoir(State h, BitStream bs, Scratch s, int mainDataBegin)
    {
        int frameBytes = (bs.Limit - bs.Pos) / 8;
        int bytesHave = Math.Min(h.Reserv, mainDataBegin);
        Array.Copy(h.ReservBuf, Math.Max(0, h.Reserv - mainDataBegin), s.MainData, 0, Math.Min(h.Reserv, mainDataBegin));
        Array.Copy(bs.Buf, bs.Start + bs.Pos / 8, s.MainData, bytesHave, frameBytes);
        Array.Clear(s.MainData, bytesHave + frameBytes, s.MainData.Length - bytesHave - frameBytes);
        s.Bs.Init(s.MainData, 0, bytesHave + frameBytes);
        return h.Reserv >= mainDataBegin;
    }

    private static void L3Decode(State h, Scratch s, int g, int nch)
    {
        var hdr = h.Header;
        for (int ch = 0; ch < nch; ch++)
        {
            int layer3grLimit = s.Bs.Pos + s.GrInfo[g + ch].Part23Length;
            L3DecodeScalefactors(hdr, 0, s.IstPos, ch * 39, s.Bs, s.GrInfo[g + ch], s.Scf, ch);
            L3Huffman(s.GrBuf, ch * 576, s.Bs, s.GrInfo[g + ch], s.Scf, layer3grLimit);
        }

        if (TestIStereo(hdr, 0))
            L3IntensityStereo(s.GrBuf, s.IstPos, 39, s.GrInfo, g, hdr, 0);
        else if (IsMsStereo(hdr, 0))
            L3MidsideStereo(s.GrBuf, 0, 576);

        for (int ch = 0; ch < nch; ch++)
        {
            var gr = s.GrInfo[g + ch];
            int aaBands = 31;
            int nLongBands = (gr.MixedBlockFlag != 0 ? 2 : 0) << (GetMySampleRate(hdr, 0) == 2 ? 1 : 0);
            if (gr.NShortSfb != 0)
            {
                aaBands = nLongBands - 1;
                L3Reorder(s.GrBuf, ch * 576 + nLongBands * 18, s.Syn, gr.SfbTab, gr.NLongSfb);
            }
            L3Antialias(s.GrBuf, ch * 576, aaBands);
            L3ImdctGr(s.GrBuf, ch * 576, h.MdctOverlap, ch * 9 * 32, gr.BlockType, nLongBands);
            L3ChangeSign(s.GrBuf, ch * 576);
        }
    }

    // ------------------------------------------------------------------ synthesis filterbank

    private static readonly float[] Sec = { 10.19000816f,0.50060302f,0.50241929f,3.40760851f,0.50547093f,0.52249861f,2.05778098f,0.51544732f,0.56694406f,1.48416460f,0.53104258f,0.64682180f,1.16943991f,0.55310392f,0.78815460f,0.97256821f,0.58293498f,1.06067765f,0.83934963f,0.62250412f,1.72244716f,0.74453628f,0.67480832f,5.10114861f };

    private static void DctII(float[] buf, int grbuf, int n)
    {
        var t = new float[4 * 8];
        for (int k = 0; k < n; k++)
        {
            int y = grbuf + k;
            for (int i = 0; i < 8; i++)
            {
                float x0 = buf[y + i * 18];
                float x1 = buf[y + (15 - i) * 18];
                float x2 = buf[y + (16 + i) * 18];
                float x3 = buf[y + (31 - i) * 18];
                float t0 = x0 + x3;
                float t1 = x1 + x2;
                float t2 = (x1 - x2) * Sec[3 * i + 0];
                float t3 = (x0 - x3) * Sec[3 * i + 1];
                t[i + 0] = t0 + t1;
                t[i + 8] = (t0 - t1) * Sec[3 * i + 2];
                t[i + 16] = t3 + t2;
                t[i + 24] = (t3 - t2) * Sec[3 * i + 2];
            }
            for (int r = 0; r < 4; r++)
            {
                int x = r * 8;
                float x0 = t[x + 0], x1 = t[x + 1], x2 = t[x + 2], x3 = t[x + 3], x4 = t[x + 4], x5 = t[x + 5], x6 = t[x + 6], x7 = t[x + 7], xt;
                xt = x0 - x7; x0 += x7;
                x7 = x1 - x6; x1 += x6;
                x6 = x2 - x5; x2 += x5;
                x5 = x3 - x4; x3 += x4;
                x4 = x0 - x3; x0 += x3;
                x3 = x1 - x2; x1 += x2;
                t[x + 0] = x0 + x1;
                t[x + 4] = (x0 - x1) * 0.70710677f;
                x5 = x5 + x6;
                x6 = (x6 + x7) * 0.70710677f;
                x7 = x7 + xt;
                x3 = (x3 + x4) * 0.70710677f;
                x5 -= x7 * 0.198912367f;
                x7 += x5 * 0.382683432f;
                x5 -= x7 * 0.198912367f;
                x0 = xt - x6; xt += x6;
                t[x + 1] = (xt + x7) * 0.50979561f;
                t[x + 2] = (x4 + x3) * 0.54119611f;
                t[x + 3] = (x0 - x5) * 0.60134488f;
                t[x + 5] = (x0 + x5) * 0.89997619f;
                t[x + 6] = (x4 - x3) * 1.30656302f;
                t[x + 7] = (xt - x7) * 2.56291556f;
            }
            for (int i = 0; i < 7; i++, y += 4 * 18)
            {
                buf[y + 0 * 18] = t[0 * 8 + i];
                buf[y + 1 * 18] = t[2 * 8 + i] + t[3 * 8 + i] + t[3 * 8 + i + 1];
                buf[y + 2 * 18] = t[1 * 8 + i] + t[1 * 8 + i + 1];
                buf[y + 3 * 18] = t[2 * 8 + i + 1] + t[3 * 8 + i] + t[3 * 8 + i + 1];
            }
            buf[y + 0 * 18] = t[0 * 8 + 7];
            buf[y + 1 * 18] = t[2 * 8 + 7] + t[3 * 8 + 7];
            buf[y + 2 * 18] = t[1 * 8 + 7];
            buf[y + 3 * 18] = t[3 * 8 + 7];
        }
    }

    private static float ScalePcm(float sample) => sample * (1f / 32768f);

    private static void SynthPair(float[] pcm, int p, int nch, float[] z, int zi)
    {
        float a;
        a = (z[zi + 14 * 64] - z[zi + 0]) * 29;
        a += (z[zi + 1 * 64] + z[zi + 13 * 64]) * 213;
        a += (z[zi + 12 * 64] - z[zi + 2 * 64]) * 459;
        a += (z[zi + 3 * 64] + z[zi + 11 * 64]) * 2037;
        a += (z[zi + 10 * 64] - z[zi + 4 * 64]) * 5153;
        a += (z[zi + 5 * 64] + z[zi + 9 * 64]) * 6574;
        a += (z[zi + 8 * 64] - z[zi + 6 * 64]) * 37489;
        a += z[zi + 7 * 64] * 75038;
        pcm[p] = ScalePcm(a);

        zi += 2;
        a = z[zi + 14 * 64] * 104;
        a += z[zi + 12 * 64] * 1567;
        a += z[zi + 10 * 64] * 9727;
        a += z[zi + 8 * 64] * 64019;
        a += z[zi + 6 * 64] * -9975;
        a += z[zi + 4 * 64] * -45;
        a += z[zi + 2 * 64] * 146;
        a += z[zi + 0 * 64] * -5;
        pcm[p + 16 * nch] = ScalePcm(a);
    }

    private static readonly float[] Win = { -1,26,-31,208,218,401,-519,2063,2000,4788,-5517,7134,5959,35640,-39336,74992, -1,24,-35,202,222,347,-581,2080,1952,4425,-5879,7640,5288,33791,-41176,74856, -1,21,-38,196,225,294,-645,2087,1893,4063,-6237,8092,4561,31947,-43006,74630, -1,19,-41,190,227,244,-711,2085,1822,3705,-6589,8492,3776,30112,-44821,74313, -1,17,-45,183,228,197,-779,2075,1739,3351,-6935,8840,2935,28289,-46617,73908, -1,16,-49,176,228,153,-848,2057,1644,3004,-7271,9139,2037,26482,-48390,73415, -2,14,-53,169,227,111,-919,2032,1535,2663,-7597,9389,1082,24694,-50137,72835, -2,13,-58,161,224,72,-991,2001,1414,2330,-7910,9592,70,22929,-51853,72169, -2,11,-63,154,221,36,-1064,1962,1280,2006,-8209,9750,-998,21189,-53534,71420, -2,10,-68,147,215,2,-1137,1919,1131,1692,-8491,9863,-2122,19478,-55178,70590, -3,9,-73,139,208,-29,-1210,1870,970,1388,-8755,9935,-3300,17799,-56778,69679, -3,8,-79,132,200,-57,-1283,1817,794,1095,-8998,9966,-4533,16155,-58333,68692, -4,7,-85,125,189,-83,-1356,1759,605,814,-9219,9959,-5818,14548,-59838,67629, -4,7,-91,117,177,-106,-1428,1698,402,545,-9416,9916,-7154,12980,-61289,66494, -5,6,-97,111,163,-127,-1498,1634,185,288,-9585,9838,-8540,11455,-62684,65290 };

    private static void Synth(float[] x, int xl, float[] pcm, int dstl, int nch, float[] lins, int linsOffset)
    {
        int xr = xl + 576 * (nch - 1);
        int dstr = dstl + (nch - 1);
        int zlin = linsOffset + 15 * 64;
        int w = 0;
        var z = lins;

        z[zlin + 4 * 15] = x[xl + 18 * 16];
        z[zlin + 4 * 15 + 1] = x[xr + 18 * 16];
        z[zlin + 4 * 15 + 2] = x[xl + 0];
        z[zlin + 4 * 15 + 3] = x[xr + 0];

        z[zlin + 4 * 31] = x[xl + 1 + 18 * 16];
        z[zlin + 4 * 31 + 1] = x[xr + 1 + 18 * 16];
        z[zlin + 4 * 31 + 2] = x[xl + 1];
        z[zlin + 4 * 31 + 3] = x[xr + 1];

        SynthPair(pcm, dstr, nch, z, linsOffset + 4 * 15 + 1);
        SynthPair(pcm, dstr + 32 * nch, nch, z, linsOffset + 4 * 15 + 64 + 1);
        SynthPair(pcm, dstl, nch, z, linsOffset + 4 * 15);
        SynthPair(pcm, dstl + 32 * nch, nch, z, linsOffset + 4 * 15 + 64);

        var a = new float[4];
        var b = new float[4];
        for (int i = 14; i >= 0; i--)
        {
            z[zlin + 4 * i] = x[xl + 18 * (31 - i)];
            z[zlin + 4 * i + 1] = x[xr + 18 * (31 - i)];
            z[zlin + 4 * i + 2] = x[xl + 1 + 18 * (31 - i)];
            z[zlin + 4 * i + 3] = x[xr + 1 + 18 * (31 - i)];
            z[zlin + 4 * (i + 16)] = x[xl + 1 + 18 * (1 + i)];
            z[zlin + 4 * (i + 16) + 1] = x[xr + 1 + 18 * (1 + i)];
            z[zlin + 4 * (i - 16) + 2] = x[xl + 18 * (1 + i)];
            z[zlin + 4 * (i - 16) + 3] = x[xr + 18 * (1 + i)];

            for (int k = 0; k < 8; k++)
            {
                float w0 = Win[w++], w1 = Win[w++];
                int vz = zlin + 4 * i - k * 64, vy = zlin + 4 * i - (15 - k) * 64;
                for (int j = 0; j < 4; j++)
                {
                    float bb = z[vz + j] * w1 + z[vy + j] * w0;
                    float aa = k == 0 ? z[vz + j] * w0 - z[vy + j] * w1        // S0
                        : (k & 1) == 1 ? z[vy + j] * w1 - z[vz + j] * w0      // S2 (odd k)
                        : z[vz + j] * w0 - z[vy + j] * w1;                    // S1 (even k)
                    if (k == 0)
                    {
                        b[j] = bb;
                        a[j] = aa;
                    }
                    else
                    {
                        b[j] += bb;
                        a[j] += aa;
                    }
                }
            }

            pcm[dstr + (15 - i) * nch] = ScalePcm(a[1]);
            pcm[dstr + (17 + i) * nch] = ScalePcm(b[1]);
            pcm[dstl + (15 - i) * nch] = ScalePcm(a[0]);
            pcm[dstl + (17 + i) * nch] = ScalePcm(b[0]);
            pcm[dstr + (47 - i) * nch] = ScalePcm(a[3]);
            pcm[dstr + (49 + i) * nch] = ScalePcm(b[3]);
            pcm[dstl + (47 - i) * nch] = ScalePcm(a[2]);
            pcm[dstl + (49 + i) * nch] = ScalePcm(b[2]);
        }
    }

    private static void SynthGranule(float[] qmfState, float[] grbuf, int nbands, int nch, float[] pcm, int pcmOffset, float[] lins)
    {
        for (int i = 0; i < nch; i++) DctII(grbuf, 576 * i, nbands);
        Array.Copy(qmfState, 0, lins, 0, 15 * 64);
        for (int i = 0; i < nbands; i += 2) Synth(grbuf, i, pcm, pcmOffset + 32 * nch * i, nch, lins, i * 64);
        if (nch == 1)
        {
            for (int i = 0; i < 15 * 64; i += 2) qmfState[i] = lins[nbands * 64 + i];
        }
        else
        {
            Array.Copy(lins, nbands * 64, qmfState, 0, 15 * 64);
        }
    }

    // ------------------------------------------------------------------ framing

    private static bool MatchFrame(byte[] b, int h, int mp3Bytes, int frameBytes)
    {
        int i = 0;
        for (int nmatch = 0; nmatch < MaxFrameSyncMatches; nmatch++)
        {
            i += HdrFrameBytes(b, h + i, frameBytes) + HdrPadding(b, h + i);
            if (i + HdrSize > mp3Bytes) return nmatch > 0;
            if (!HdrCompare(b, h, h + i)) return false;
        }
        return true;
    }

    private static int FindFrame(byte[] b, int start, int mp3Bytes, ref int freeFormatBytes, out int frameBytesOut)
    {
        for (int i = 0; i < mp3Bytes - HdrSize; i++)
        {
            int m = start + i;
            if (!HdrValid(b, m)) continue;
            int frameBytes = HdrFrameBytes(b, m, freeFormatBytes);
            int frameAndPadding = frameBytes + HdrPadding(b, m);
            for (int k = HdrSize; frameBytes == 0 && k < MaxFreeFormatFrameSize && i + 2 * k < mp3Bytes - HdrSize; k++)
            {
                if (HdrCompare(b, m, m + k))
                {
                    int fb = k - HdrPadding(b, m);
                    int nextfb = fb + HdrPadding(b, m + k);
                    if (i + k + nextfb + HdrSize > mp3Bytes || !HdrCompare(b, m, m + k + nextfb)) continue;
                    frameAndPadding = k;
                    frameBytes = fb;
                    freeFormatBytes = fb;
                }
            }
            if ((frameBytes != 0 && i + frameAndPadding <= mp3Bytes && MatchFrame(b, m, mp3Bytes - i, frameBytes))
                || (i == 0 && frameAndPadding == mp3Bytes))
            {
                frameBytesOut = frameAndPadding;
                return i;
            }
            freeFormatBytes = 0;
        }
        frameBytesOut = 0;
        return mp3Bytes;
    }

    private static int DecodeFrame(State dec, byte[] b, int start, int mp3Bytes, float[] pcm, out FrameInfo info)
    {
        info = default;
        int i = 0, frameSize = 0;
        bool success = true;
        if (mp3Bytes > 4 && dec.Header[0] == 0xff && HdrCompare(Concat(dec.Header, b, start), 0, 4))
        {
            frameSize = HdrFrameBytes(b, start, dec.FreeFormatBytes) + HdrPadding(b, start);
            if (frameSize != mp3Bytes && (frameSize + HdrSize > mp3Bytes || !HdrCompare(b, start, start + frameSize)))
                frameSize = 0;
        }
        if (frameSize == 0)
        {
            Reset(dec);
            i = FindFrame(b, start, mp3Bytes, ref dec.FreeFormatBytes, out frameSize);
            if (frameSize == 0 || i + frameSize > mp3Bytes)
            {
                info.FrameBytes = i;
                return 0;
            }
        }

        int hdr = start + i;
        Array.Copy(b, hdr, dec.Header, 0, HdrSize);
        info.FrameBytes = i + frameSize;
        info.FrameOffset = i;
        info.Channels = IsMono(b, hdr) ? 1 : 2;
        info.Hz = HdrSampleRateHz(b, hdr);
        info.Layer = 4 - GetLayer(b, hdr);
        info.BitrateKbps = HdrBitrateKbps(b, hdr);
        if (info.Layer != 3) return 0;

        var bsFrame = new BitStream();
        bsFrame.Init(b, hdr + HdrSize, frameSize - HdrSize);
        if (IsCrc(b, hdr)) bsFrame.GetBits(16);

        var scratch = new Scratch();
        int mainDataBegin = L3ReadSideInfo(bsFrame, scratch.GrInfo, b, hdr);
        if (mainDataBegin < 0 || bsFrame.Pos > bsFrame.Limit)
        {
            Reset(dec);
            return 0;
        }
        success = L3RestoreReservoir(dec, bsFrame, scratch, mainDataBegin);
        if (success)
        {
            int pcmOffset = 0;
            for (int igr = 0; igr < (TestMpeg1(b, hdr) ? 2 : 1); igr++, pcmOffset += 576 * info.Channels)
            {
                Array.Clear(scratch.GrBuf, 0, scratch.GrBuf.Length);
                L3Decode(dec, scratch, igr * info.Channels, info.Channels);
                SynthGranule(dec.QmfState, scratch.GrBuf, 18, info.Channels, pcm, pcmOffset, scratch.Syn);
            }
        }
        L3SaveReservoir(dec, scratch);
        return success ? HdrFrameSamples(dec.Header, 0) : 0;
    }

    // The previous header followed by the current one, so the two can be compared by offset.
    private static byte[] Concat(byte[] header, byte[] b, int start)
    {
        var both = new byte[8];
        Array.Copy(header, 0, both, 0, 4);
        Array.Copy(b, start, both, 4, Math.Min(4, b.Length - start));
        return both;
    }

    private static void Reset(State dec)
    {
        Array.Clear(dec.MdctOverlap, 0, dec.MdctOverlap.Length);
        Array.Clear(dec.QmfState, 0, dec.QmfState.Length);
        Array.Clear(dec.Header, 0, dec.Header.Length);
        Array.Clear(dec.ReservBuf, 0, dec.ReservBuf.Length);
        dec.Reserv = 0;
        dec.FreeFormatBytes = 0;
    }
}
