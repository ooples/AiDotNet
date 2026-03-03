namespace AiDotNet.Tests.UnitTests.Serialization;

using System;
using System.IO;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Xunit;

public class ModelLoaderTests
{
    [Fact]
    public void IsSelfDescribing_ReturnsFalseForMissingFile()
    {
        Assert.False(ModelLoader.IsSelfDescribing("nonexistent_file_99999.bin"));
    }

    [Fact]
    public void IsSelfDescribing_ReturnsTrueForAimfFile()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_isd_{Guid.NewGuid():N}.bin");
        try
        {
            var model = new StubModelSerializer { Payload = new byte[] { 1, 2, 3 } };
            byte[] wrapped = ModelFileHeader.WrapWithHeader(
                model.Payload, model, Array.Empty<int>(), Array.Empty<int>(), SerializationFormat.Binary);
            File.WriteAllBytes(tempFile, wrapped);

            Assert.True(ModelLoader.IsSelfDescribing(tempFile));
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
    public void IsSelfDescribing_ReturnsFalseForLegacyFile()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"legacy_isd_{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(tempFile, new byte[] { 0xFF, 0xFE, 0xFD, 0xFC, 0x00 });

            Assert.False(ModelLoader.IsSelfDescribing(tempFile));
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
    public void Inspect_ThrowsOnNullPath()
    {
        Assert.Throws<ArgumentException>(() => ModelLoader.Inspect(null));
    }

    [Fact]
    public void Inspect_ThrowsOnEmptyPath()
    {
        Assert.Throws<ArgumentException>(() => ModelLoader.Inspect(""));
    }

    [Fact]
    public void Inspect_ThrowsOnMissingFile()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ModelLoader.Inspect("definitely_not_a_real_file_abc123.bin"));
    }

    [Fact]
    public void Inspect_ReturnsHeaderInfo()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_inspect_{Guid.NewGuid():N}.bin");
        try
        {
            var model = new StubModelSerializer
            {
                Payload = new byte[] { 10, 20, 30 },
                InputShapeValue = new[] { 28, 28 },
                OutputShapeValue = new[] { 10 }
            };
            byte[] wrapped = ModelFileHeader.WrapWithHeader(
                model.Payload, model, model.GetInputShape(), model.GetOutputShape(), SerializationFormat.Binary);
            File.WriteAllBytes(tempFile, wrapped);

            ModelFileInfo info = ModelLoader.Inspect(tempFile);

            Assert.Equal(typeof(StubModelSerializer).Name, info.TypeName);
            Assert.Equal(new[] { 28, 28 }, info.InputShape);
            Assert.Equal(new[] { 10 }, info.OutputShape);
            Assert.Equal(SerializationFormat.Binary, info.Format);
            Assert.Equal(3, info.PayloadLength);
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
    public void Load_ThrowsOnNullPath()
    {
        Assert.Throws<ArgumentException>(() => ModelLoader.Load<double>(null));
    }

    [Fact]
    public void Load_ThrowsOnEmptyPath()
    {
        Assert.Throws<ArgumentException>(() => ModelLoader.Load<double>(""));
    }

    [Fact]
    public void Load_ThrowsOnMissingFile()
    {
        Assert.Throws<FileNotFoundException>(() =>
            ModelLoader.Load<double>("no_such_model_xyz.aimf"));
    }

    [Fact]
    public void Load_ThrowsOnLegacyFile()
    {
        var tempFile = Path.Combine(Path.GetTempPath(), $"legacy_load_{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(tempFile, Encoding.UTF8.GetBytes("not-aimf-data"));

            var ex = Assert.Throws<InvalidOperationException>(() => ModelLoader.Load<double>(tempFile));
            Assert.Contains("AIMF", ex.Message);
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
    public void LoadFromBytes_ThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() => ModelLoader.LoadFromBytes<double>(null));
    }

    [Fact]
    public void LoadFromBytes_ThrowsOnNonAimfData()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            ModelLoader.LoadFromBytes<double>(new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04 }));
        Assert.Contains("AIMF", ex.Message);
    }

    [Fact]
    public void LoadFromBytes_StubModel_RoundTrip()
    {
        // Register the stub so the registry can resolve it
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var payload = new byte[] { 0xCA, 0xFE, 0xBA, 0xBE };
        var model = new StubModelSerializer { Payload = payload };

        byte[] wrapped = ModelFileHeader.WrapWithHeader(
            payload, model, new[] { 4 }, new[] { 1 }, SerializationFormat.Binary);

        var loaded = ModelLoader.LoadFromBytes<double>(wrapped);

        Assert.NotNull(loaded);
        Assert.IsType<StubModelSerializer>(loaded);

        var loadedStub = (StubModelSerializer)loaded;
        Assert.Equal(payload, loadedStub.GetDeserializedData());
    }

    [Fact]
    public void Load_FileRoundTrip()
    {
        // Register the stub so the registry can resolve it
        ModelTypeRegistry.Register(typeof(StubModelSerializer).Name, typeof(StubModelSerializer));

        var tempFile = Path.Combine(Path.GetTempPath(), $"aimf_load_rt_{Guid.NewGuid():N}.bin");
        try
        {
            var payload = new byte[] { 0x11, 0x22, 0x33, 0x44, 0x55 };
            var model = new StubModelSerializer
            {
                Payload = payload,
                InputShapeValue = new[] { 5 },
                OutputShapeValue = new[] { 1 }
            };

            byte[] wrapped = ModelFileHeader.WrapWithHeader(
                payload, model, model.GetInputShape(), model.GetOutputShape(), SerializationFormat.Binary);
            File.WriteAllBytes(tempFile, wrapped);

            var loaded = ModelLoader.Load<double>(tempFile);

            Assert.NotNull(loaded);
            Assert.IsType<StubModelSerializer>(loaded);

            var loadedStub = (StubModelSerializer)loaded;
            Assert.Equal(payload, loadedStub.GetDeserializedData());
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
    public void LoadFromBytes_UnknownType_Throws()
    {
        // Create AIMF data with a type name that won't be in the registry
        // We'll manually craft the header with a fake type name
        var payload = new byte[] { 0x01 };
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write(ModelFileHeader.AimfMagic);
        writer.Write(ModelFileHeader.CurrentEnvelopeVersion);
        writer.Write((int)SerializationFormat.Binary);
        writer.Write("CompletelyFakeModel_NoSuchType_99999");
        writer.Write("Fake.Assembly.QualifiedName");
        writer.Write(0); // input shape rank
        writer.Write(0); // output shape rank
        writer.Write(0); // v2: dynamic input dimension count
        writer.Write(0); // v2: dynamic output dimension count
        writer.Write((long)payload.Length);
        writer.Write(payload);
        writer.Flush();

        var data = ms.ToArray();

        var ex = Assert.Throws<InvalidOperationException>(() =>
            ModelLoader.LoadFromBytes<double>(data));
        Assert.Contains("Cannot resolve model type", ex.Message);
    }
}
