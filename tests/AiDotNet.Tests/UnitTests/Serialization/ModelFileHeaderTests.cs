namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.IO;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

/// <summary>
/// A minimal IModelSerializer stub for testing AIMF header wrapping.
/// </summary>
internal sealed class StubModelSerializer : IModelSerializer, IModelShape
{
    private byte[] _data = Array.Empty<byte>();

    public byte[] Payload { get; set; } = Array.Empty<byte>();

    public int[] InputShapeValue { get; set; } = Array.Empty<int>();

    public int[] OutputShapeValue { get; set; } = Array.Empty<int>();

    public byte[] Serialize() => Payload;

    public void Deserialize(byte[] data)
    {
        _data = data ?? Array.Empty<byte>();
    }

    public byte[] GetDeserializedData() => _data;

    public void SaveModel(string filePath)
    {
        byte[] raw = Serialize();
        byte[] enveloped = ModelFileHeader.WrapWithHeader(
            raw, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(filePath, enveloped);
    }

    public void LoadModel(string filePath)
    {
        byte[] raw = File.ReadAllBytes(filePath);
        if (ModelFileHeader.HasHeader(raw))
        {
            raw = ModelFileHeader.ExtractPayload(raw);
        }
        Deserialize(raw);
    }

    public int[] GetInputShape() => InputShapeValue;

    public int[] GetOutputShape() => OutputShapeValue;
}

public class ModelFileHeaderTests
{
    [Fact]
    public void WrapWithHeader_ThrowsOnNullPayload()
    {
        var model = new StubModelSerializer();
        Assert.Throws<ArgumentNullException>(() =>
            ModelFileHeader.WrapWithHeader(null, model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary));
    }

    [Fact]
    public void WrapWithHeader_ThrowsOnNullModel()
    {
        Assert.Throws<ArgumentNullException>(() =>
            ModelFileHeader.WrapWithHeader(new byte[] { 1, 2, 3 }, null, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary));
    }

    [Fact]
    public void HasHeader_ReturnsFalseForNull()
    {
        Assert.False(ModelFileHeader.HasHeader((byte[])null));
    }

    [Fact]
    public void HasHeader_ReturnsFalseForTooShort()
    {
        Assert.False(ModelFileHeader.HasHeader(new byte[] { 0x41, 0x49 }));
    }

    [Fact]
    public void HasHeader_ReturnsFalseForRandomData()
    {
        Assert.False(ModelFileHeader.HasHeader(new byte[] { 0x00, 0x00, 0x00, 0x00, 0xFF }));
    }

    [Fact]
    public void HasHeader_ReturnsTrueForAimfMagic()
    {
        // AIMF magic: 0x41494D46 in little-endian = bytes 0x46 0x4D 0x49 0x41
        var data = BitConverter.GetBytes(ModelFileHeader.AimfMagic);
        // Need at least 4 bytes
        Assert.True(ModelFileHeader.HasHeader(data));
    }

    [Fact]
    public void WrapAndReadHeader_RoundTrip_PreservesMetadata()
    {
        var model = new StubModelSerializer
        {
            Payload = Encoding.UTF8.GetBytes("test-payload-data"),
            InputShapeValue = new[] { 784 },
            OutputShapeValue = new[] { 10 }
        };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, model.GetInputShape(), model.GetOutputShape(), SerializationFormat.Binary);

        Assert.True(ModelFileHeader.HasHeader(wrapped));

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        Assert.Equal(ModelFileHeader.CurrentEnvelopeVersion, info.EnvelopeVersion);
        Assert.Equal(SerializationFormat.Binary, info.Format);
        Assert.Equal(typeof(StubModelSerializer).Name, info.TypeName);
        Assert.Contains(typeof(StubModelSerializer).FullName ?? "", info.AssemblyQualifiedName);
        Assert.Equal(new[] { 784 }, info.InputShape);
        Assert.Equal(new[] { 10 }, info.OutputShape);
        Assert.Equal(model.Payload.Length, info.PayloadLength);
        Assert.True(info.HeaderLength > 0);
    }

    [Fact]
    public void WrapAndExtractPayload_RoundTrip_PreservesPayload()
    {
        var originalPayload = new byte[] { 0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x00, 0xFF };
        var model = new StubModelSerializer { Payload = originalPayload };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            originalPayload, model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Json);

        byte[] extracted = ModelFileHeader.ExtractPayload(wrapped);

        Assert.Equal(originalPayload, extracted);
    }

    [Fact]
    public void WrapAndExtractPayload_EmptyPayload_RoundTrip()
    {
        var model = new StubModelSerializer { Payload = Array.Empty<byte>() };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            Array.Empty<byte>(), model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary);

        Assert.True(ModelFileHeader.HasHeader(wrapped));

        byte[] extracted = ModelFileHeader.ExtractPayload(wrapped);
        Assert.Empty(extracted);
    }

    [Fact]
    public void WrapAndExtractPayload_LargePayload_RoundTrip()
    {
        // 1 MB payload
        var payload = new byte[1024 * 1024];
        new Random(42).NextBytes(payload);
        var model = new StubModelSerializer { Payload = payload };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            payload, model, new[] { 3, 224, 224 }, new[] { 1000 }, SerializationFormat.Binary);

        Assert.True(ModelFileHeader.HasHeader(wrapped));

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);
        Assert.Equal(new[] { 3, 224, 224 }, info.InputShape);
        Assert.Equal(new[] { 1000 }, info.OutputShape);
        Assert.Equal(payload.Length, info.PayloadLength);

        byte[] extracted = ModelFileHeader.ExtractPayload(wrapped);
        Assert.Equal(payload, extracted);
    }

    [Fact]
    public void ReadHeader_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() => ModelFileHeader.ReadHeader(null));
    }

    [Fact]
    public void ReadHeader_ThrowsOnNonAimfData()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            ModelFileHeader.ReadHeader(new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05 }));
        Assert.Contains("AIMF", ex.Message);
    }

    [Fact]
    public void ExtractPayload_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() => ModelFileHeader.ExtractPayload(null));
    }

    [Fact]
    public void SerializationFormat_AllFormats_RoundTrip()
    {
        var payload = new byte[] { 1, 2, 3 };
        var model = new StubModelSerializer { Payload = payload };

        foreach (SerializationFormat format in Enum.GetValues(typeof(SerializationFormat)))
        {
            byte[] wrapped = ModelFileHeader.WrapWithHeader(
                payload, model, Array.Empty<int>(), Array.Empty<int>(), format);

            ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);
            Assert.Equal(format, info.Format);

            byte[] extracted = ModelFileHeader.ExtractPayload(wrapped, info);
            Assert.Equal(payload, extracted);
        }
    }

    [Fact]
    public void HasHeader_FileOverload_ReturnsFalseForMissingFile()
    {
        Assert.False(ModelFileHeader.HasHeader("nonexistent_path_12345.bin"));
    }

    [Fact]
    public void HasHeader_FileOverload_ReturnsFalseForEmptyPath()
    {
        Assert.False(ModelFileHeader.HasHeader(string.Empty));
    }

    [Fact]
    public void SaveModel_LoadModel_FileRoundTrip()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_test_{Guid.NewGuid():N}.bin");
        try
        {
            var payload = Encoding.UTF8.GetBytes("hello-aimf-world");
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 100 },
                OutputShapeValue = new[] { 5 }
            };

            model.SaveModel(tempFile);

            // Verify the file has AIMF header
            Assert.True(ModelFileHeader.HasHeader(tempFile));

            // Load it back
            var loaded = new StubModelSerializer();
            loaded.LoadModel(tempFile);

            Assert.Equal(payload, loaded.GetDeserializedData());
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void LoadModel_LegacyFile_FallsBackWithoutHeader()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"legacy_test_{Guid.NewGuid():N}.bin");
        try
        {
            // Write raw data WITHOUT AIMF header (simulating legacy format)
            var legacyPayload = Encoding.UTF8.GetBytes("legacy-model-data");
            File.WriteAllBytes(tempFile, legacyPayload);

            Assert.False(ModelFileHeader.HasHeader(tempFile));

            // LoadModel should handle legacy format
            var model = new StubModelSerializer();
            model.LoadModel(tempFile);

            Assert.Equal(legacyPayload, model.GetDeserializedData());
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void MultiDimensionalShapes_RoundTrip()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0xAA, 0xBB },
            InputShapeValue = new[] { 3, 224, 224 },
            OutputShapeValue = new[] { 10, 10 }
        };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, model.GetInputShape(), model.GetOutputShape(), SerializationFormat.HybridBinary);

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        Assert.Equal(new[] { 3, 224, 224 }, info.InputShape);
        Assert.Equal(new[] { 10, 10 }, info.OutputShape);
        Assert.Equal(SerializationFormat.HybridBinary, info.Format);
    }

    [Fact]
    public void EmptyShapes_RoundTrip()
    {
        var model = new StubModelSerializer
        {
            Payload = new byte[] { 0x01 },
            InputShapeValue = Array.Empty<int>(),
            OutputShapeValue = Array.Empty<int>()
        };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, model.GetInputShape(), model.GetOutputShape(), SerializationFormat.Binary);

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        Assert.Empty(info.InputShape);
        Assert.Empty(info.OutputShape);
    }

    [Fact]
    public void NullShapes_TreatedAsEmpty()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 0x01 } };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, null, null, SerializationFormat.Binary);

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        Assert.Empty(info.InputShape);
        Assert.Empty(info.OutputShape);
    }

    [Fact]
    public void HeaderLength_IsPositive()
    {
        var model = new StubModelSerializer { Payload = new byte[] { 0x01 } };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            model.Payload, model, new[] { 5 }, new[] { 1 }, SerializationFormat.Binary);

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        Assert.True(info.HeaderLength > 0);
        Assert.Equal(wrapped.Length, info.HeaderLength + info.PayloadLength);
    }

    [Fact]
    public void ExtractPayload_WithPreParsedInfo_MatchesDirect()
    {
        var payload = new byte[] { 10, 20, 30, 40, 50 };
        var model = new StubModelSerializer { Payload = payload };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            payload, model, new[] { 5 }, new[] { 1 }, SerializationFormat.Binary);

        ModelFileInfo info = ModelFileHeader.ReadHeader(wrapped);

        // Both methods should return the same payload
        byte[] withInfo = ModelFileHeader.ExtractPayload(wrapped, info);
        byte[] withoutInfo = ModelFileHeader.ExtractPayload(wrapped);

        Assert.Equal(withInfo, withoutInfo);
        Assert.Equal(payload, withInfo);
    }
}
