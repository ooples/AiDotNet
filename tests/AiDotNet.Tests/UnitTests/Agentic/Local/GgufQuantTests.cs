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

        // Wraps a single quantized tensor (32 values) in a minimal GGUF buffer.
        private static byte[] BuildGguf(uint ggmlType, byte[] tensorData)
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
                w.Write(1u);          // n_dims
                w.Write((ulong)32);   // dim0 = one block
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
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_0_Dequantizes()
        {
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
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_1_Dequantizes_WithMin()
        {
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
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task Q4_0_RoundTripsVaryingNibbles()
        {
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
            await Task.CompletedTask;
        }
    }
}
