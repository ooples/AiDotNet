using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class GgufQuantTests
    {
        private const ushort Half05 = 0x3800; // 0.5 in IEEE half
        private const ushort Half10 = 0x3C00; // 1.0 in IEEE half

        // Wraps a single quantized tensor in a minimal GGUF buffer. elementCount is the tensor's dim0
        // (32 for a classic block, 256 for a k-quant super-block).
        private static byte[] BuildGguf(uint ggmlType, byte[] tensorData, ulong elementCount = 32)
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(0x46554747u); // "GGUF"
                w.Write(3u);          // version
                w.Write((ulong)1);    // tensor count
                w.Write((ulong)0);    // metadata count

                var nameBytes = Encoding.UTF8.GetBytes("w");
                w.Write((ulong)nameBytes.Length);
                w.Write(nameBytes);
                w.Write(1u);              // n_dims
                w.Write(elementCount);    // dim0
                w.Write(ggmlType);
                w.Write((ulong)0);    // offset

                w.Flush();
                var pad = (int)((32 - (ms.Position % 32)) % 32);
                for (var i = 0; i < pad; i++)
                {
                    w.Write((byte)0);
                }

                w.Write(tensorData);
            }

            return ms.ToArray();
        }

        [Fact(Timeout = 60000)]
        public async Task Q8_0_Dequantizes()
        {
            // Yield first so xUnit's timeout guard is armed before any work runs.
            await Task.Yield();

            // scale 0.5, all quants = 4 -> value 2.0
            var data = new byte[2 + 32];
            BitConverter.GetBytes(Half05).CopyTo(data, 0);
            for (var i = 0; i < 32; i++)
            {
                data[2 + i] = 4;
            }

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ8_0, data));
            var values = file.ReadAsDouble("w");

            Assert.Equal(32, values.Length);
            Assert.All(values, v => Assert.Equal(2.0, v, 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_0_Dequantizes()
        {
            await Task.Yield();

            // scale 0.5, every nibble = 9 -> (9-8)*0.5 = 0.5
            var data = new byte[2 + 16];
            BitConverter.GetBytes(Half05).CopyTo(data, 0);
            for (var i = 0; i < 16; i++)
            {
                data[2 + i] = 0x99; // low and high nibble both 9
            }

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ4_0, data));
            var values = file.ReadAsDouble("w");

            Assert.Equal(32, values.Length);
            Assert.All(values, v => Assert.Equal(0.5, v, 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_1_Dequantizes_WithMin()
        {
            await Task.Yield();

            // scale 0.5, min 1.0, every nibble = 2 -> 2*0.5 + 1.0 = 2.0
            var data = new byte[2 + 2 + 16];
            BitConverter.GetBytes(Half05).CopyTo(data, 0);
            BitConverter.GetBytes(Half10).CopyTo(data, 2);
            for (var i = 0; i < 16; i++)
            {
                data[4 + i] = 0x22; // both nibbles = 2
            }

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ4_1, data));
            var values = file.ReadAsDouble("w");

            Assert.Equal(32, values.Length);
            Assert.All(values, v => Assert.Equal(2.0, v, 3));
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_K_Dequantizes_SuperBlock()
        {
            await Task.Yield();

            // d=1, dmin=1. scales encode all 8 sub-blocks to scale=2, min=1 (see get_scale_min_k4 packing):
            //   scales[0..3]=2 (j<4 scale), scales[4..7]=1 (j<4 min), scales[8..11]=0x12 (j>=4: low nibble 2 -> scale,
            //   high nibble 1 -> min; the 2 high bits come from scales[0..7]>>6 which are 0 here).
            // value = d*scale*q - dmin*min = 2q - 1. qs all 0x33 (nibble 3 -> 5), except qs[0]=0x51:
            //   low nibble 1 -> y[0]=1, high nibble 5 -> y[32]=9.
            var data = new byte[2 + 2 + 12 + 128];
            BitConverter.GetBytes(Half10).CopyTo(data, 0); // d = 1.0
            BitConverter.GetBytes(Half10).CopyTo(data, 2); // dmin = 1.0
            for (var i = 0; i < 4; i++) data[4 + i] = 2;
            for (var i = 4; i < 8; i++) data[4 + i] = 1;
            for (var i = 8; i < 12; i++) data[4 + i] = 0x12;
            var qsAt = 2 + 2 + 12;
            for (var i = 0; i < 128; i++) data[qsAt + i] = 0x33;
            data[qsAt] = 0x51;

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ4_K, data, 256));
            var values = file.ReadAsDouble("w");

            Assert.Equal(256, values.Length);
            Assert.Equal(1.0, values[0], 3);
            Assert.Equal(9.0, values[32], 3);
            Assert.Equal(5.0, values[1], 3);
            Assert.Equal(5.0, values[255], 3);
        }

        [Fact(Timeout = 60000)]
        public async Task Q6_K_Dequantizes_SuperBlock()
        {
            await Task.Yield();

            // d=1. ql all 0x44 (both nibbles=4), qh all 0xAA (every 2-bit group=2) -> q6 = 4 | (2<<4) = 36;
            // value = d*scale*(q6-32) = scale*4. scales all 1 except scales[0]=2: with is=l/16, the first
            // 16 low-quant outputs (y[0..15]) use scales[0]=2 -> 8; everything else uses scale 1 -> 4.
            var data = new byte[128 + 64 + 16 + 2];
            for (var i = 0; i < 128; i++) data[i] = 0x44;
            for (var i = 0; i < 64; i++) data[128 + i] = 0xAA;
            for (var i = 0; i < 16; i++) data[128 + 64 + i] = 1;
            data[128 + 64] = 2; // scales[0] = 2
            BitConverter.GetBytes(Half10).CopyTo(data, 128 + 64 + 16); // d = 1.0

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ6_K, data, 256));
            var values = file.ReadAsDouble("w");

            Assert.Equal(256, values.Length);
            Assert.Equal(8.0, values[0], 3);   // l=0, is=0 -> scales[0]=2
            Assert.Equal(8.0, values[15], 3);  // l=15, is=0 -> scales[0]=2
            Assert.Equal(4.0, values[16], 3);  // l=16, is=1 -> scales[1]=1
            Assert.Equal(4.0, values[32], 3);  // q2 group uses scales[2]=1
            Assert.Equal(4.0, values[255], 3);
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_0_RoundTripsVaryingNibbles()
        {
            await Task.Yield();

            // Distinct low/high nibbles to verify packing: byte 0x70 -> low=0, high=7.
            var data = new byte[2 + 16];
            BitConverter.GetBytes(Half05).CopyTo(data, 0);
            data[2] = 0x70; // element 0 low nibble = 0 -> (0-8)*0.5 = -4 ; element 16 high nibble = 7 -> (7-8)*0.5 = -0.5
            for (var i = 1; i < 16; i++)
            {
                data[2 + i] = 0x88; // both nibbles 8 -> 0
            }

            var file = GgufReader.Read(BuildGguf(GgufTensorInfo.TypeQ4_0, data));
            var values = file.ReadAsDouble("w");

            Assert.Equal(-4.0, values[0], 3);
            Assert.Equal(-0.5, values[16], 3);
            Assert.Equal(0.0, values[1], 3);
        }
    }
}
