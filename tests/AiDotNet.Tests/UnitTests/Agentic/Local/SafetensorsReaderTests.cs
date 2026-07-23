using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models.Local;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Local
{
    public class SafetensorsReaderTests
    {
        private static byte[] Build(string headerJson, byte[] data)
        {
            var headerBytes = Encoding.UTF8.GetBytes(headerJson);
            var buffer = new byte[8 + headerBytes.Length + data.Length];
            BitConverter.GetBytes((ulong)headerBytes.Length).CopyTo(buffer, 0);
            headerBytes.CopyTo(buffer, 8);
            data.CopyTo(buffer, 8 + headerBytes.Length);
            return buffer;
        }

        [Fact(Timeout = 60000)]
        public async Task Reads_F32_And_F64_Tensors_WithShapesAndValues()
        {
            // Yield first so xUnit's timeout guard is armed before any work runs.
            await Task.Yield();

            var data = new byte[16];
            BitConverter.GetBytes(1.0f).CopyTo(data, 0);
            BitConverter.GetBytes(2.0f).CopyTo(data, 4);
            BitConverter.GetBytes(3.5).CopyTo(data, 8);

            const string header =
                @"{""a"":{""dtype"":""F32"",""shape"":[2],""data_offsets"":[0,8]}," +
                @"""b"":{""dtype"":""F64"",""shape"":[1],""data_offsets"":[8,16]}," +
                @"""__metadata__"":{""format"":""pt""}}";

            var file = SafetensorsReader.Read(Build(header, data));

            // __metadata__ is excluded from the tensor list.
            Assert.Equal(2, file.Tensors.Count);
            Assert.Contains("a", file.Names);
            Assert.Contains("b", file.Names);

            var a = file.Get("a");
            Assert.NotNull(a);
            Assert.Equal(new long[] { 2 }, a.Shape.ToArray());
            Assert.Equal(2, a.ElementCount);
            Assert.Equal(new[] { 1.0, 2.0 }, file.ReadAsDouble("a"));
            Assert.Equal(new[] { 3.5 }, file.ReadAsDouble("b"));
        }

        [Fact(Timeout = 60000)]
        public async Task Reads_F16_Tensor()
        {
            await Task.Yield();

            // 1.0 in IEEE half = 0x3C00, 2.0 = 0x4000.
            var data = new byte[4];
            BitConverter.GetBytes((ushort)0x3C00).CopyTo(data, 0);
            BitConverter.GetBytes((ushort)0x4000).CopyTo(data, 2);
            const string header = @"{""h"":{""dtype"":""F16"",""shape"":[2],""data_offsets"":[0,4]}}";

            var file = SafetensorsReader.Read(Build(header, data));
            var values = file.ReadAsDouble("h");

            Assert.Equal(1.0, values[0], 3);
            Assert.Equal(2.0, values[1], 3);
        }

        [Fact(Timeout = 60000)]
        public async Task RawBytes_AreAccessible_ForAnyDtype()
        {
            await Task.Yield();

            var data = new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            const string header = @"{""ids"":{""dtype"":""I64"",""shape"":[1],""data_offsets"":[0,8]}}";

            var file = SafetensorsReader.Read(Build(header, data));
            Assert.Equal(data, file.GetRawBytes("ids"));
            // Unsupported-for-double dtype throws a clear error rather than guessing.
            Assert.Throws<NotSupportedException>(() => file.ReadAsDouble("ids"));
        }

        [Fact(Timeout = 60000)]
        public async Task ReadsFromStream()
        {
            await Task.Yield();

            var data = new byte[4];
            BitConverter.GetBytes(9.0f).CopyTo(data, 0);
            const string header = @"{""x"":{""dtype"":""F32"",""shape"":[1],""data_offsets"":[0,4]}}";
            using var stream = new MemoryStream(Build(header, data));

            var file = SafetensorsReader.Read(stream);
            Assert.Equal(new[] { 9.0 }, file.ReadAsDouble("x"));
        }

        [Fact(Timeout = 60000)]
        public async Task Malformed_Throws()
        {
            await Task.Yield();

            Assert.Throws<InvalidDataException>(() => SafetensorsReader.Read(new byte[] { 1, 2, 3 }));

            // Header length larger than the buffer.
            var bad = new byte[8];
            BitConverter.GetBytes((ulong)1000).CopyTo(bad, 0);
            Assert.Throws<InvalidDataException>(() => SafetensorsReader.Read(bad));
        }
    }
}
