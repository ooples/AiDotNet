using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class GgufReaderTests
    {
        private static void WriteString(BinaryWriter w, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }

        // Builds a minimal GGUF: 2 metadata KVs (string + array) + general.alignment, one F32 tensor [3].
        private static byte[] BuildGguf()
        {
            using var ms = new MemoryStream();
            using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                w.Write(0x46554747u);  // "GGUF"
                w.Write(3u);           // version
                w.Write((ulong)1);     // tensor count
                w.Write((ulong)3);     // metadata count

                WriteString(w, "general.name");
                w.Write(8u);           // STRING
                WriteString(w, "test-model");

                WriteString(w, "general.alignment");
                w.Write(4u);           // UINT32
                w.Write(32u);

                WriteString(w, "tokenizer.tokens");
                w.Write(9u);           // ARRAY
                w.Write(8u);           // element type STRING
                w.Write((ulong)2);     // count
                WriteString(w, "a");
                WriteString(w, "b");

                // tensor info
                WriteString(w, "weight");
                w.Write(1u);           // n_dims
                w.Write((ulong)3);     // dim0
                w.Write(0u);           // ggml type F32
                w.Write((ulong)0);     // offset within data section

                w.Flush();
                // Pad to alignment 32 before the data section.
                var pos = ms.Position;
                var pad = (int)((32 - (pos % 32)) % 32);
                for (var i = 0; i < pad; i++)
                {
                    w.Write((byte)0);
                }

                // tensor data: 3 floats
                w.Write(1.0f);
                w.Write(2.0f);
                w.Write(3.0f);
            }

            return ms.ToArray();
        }

        [Fact(Timeout = 60000)]
        public async Task ParsesHeader_Metadata_AndTensorDirectory()
        {
            var file = GgufReader.Read(BuildGguf());

            Assert.Equal(3u, file.Version);
            Assert.Equal("test-model", file.GetMetadata("general.name"));

            var tokens = Assert.IsType<object[]>(file.GetMetadata("tokenizer.tokens"));
            Assert.Equal(new object[] { "a", "b" }, tokens);

            var tensor = Assert.Single(file.Tensors);
            Assert.Equal("weight", tensor.Name);
            Assert.Equal(new long[] { 3 }, tensor.Dimensions.ToArray());
            Assert.Equal(GgufTensorInfo.TypeF32, tensor.GgmlType);
            Assert.Equal(3, tensor.ElementCount);

            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task ReadsF32TensorValues_RespectingAlignment()
        {
            var file = GgufReader.Read(BuildGguf());
            Assert.Equal(new[] { 1.0, 2.0, 3.0 }, file.ReadAsDouble("weight"));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task ReadsFromStream()
        {
            using var stream = new MemoryStream(BuildGguf());
            var file = GgufReader.Read(stream);
            Assert.Equal("test-model", file.GetMetadata("general.name"));
            await Task.CompletedTask;
        }

        [Fact(Timeout = 60000)]
        public async Task BadMagic_Throws()
        {
            Assert.Throws<InvalidDataException>(() => GgufReader.Read(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }));
            await Task.CompletedTask;
        }
    }
}
