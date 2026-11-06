using System;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Optimizers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

public class AdamWMomentFormatHeaderTests
{
    private static byte[] Header()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream);
        AdamWMomentFormatHeader.Write(writer, OptimizerMomentStorage.Int8Blockwise, 256, 4096);
        return stream.ToArray();
    }

    [Fact]
    public void CompatibleHeader_AfterBasePayload_IsConsumedExactly()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        writer.Write(12345);
        AdamWMomentFormatHeader.Write(writer, OptimizerMomentStorage.Int8Blockwise, 256, 4096);
        stream.Position = sizeof(int);
        using var reader = new BinaryReader(stream);
        AdamWMomentFormatHeader.Validate(reader, OptimizerMomentStorage.Int8Blockwise, 256, 4096);
        Assert.Equal(stream.Length, stream.Position);
    }

    [Theory]
    [InlineData(OptimizerMomentStorage.BFloat16, 256, 4096)]
    [InlineData(OptimizerMomentStorage.Float32, 256, 4096)]
    [InlineData(OptimizerMomentStorage.Int8Blockwise, 64, 4096)]
    [InlineData(OptimizerMomentStorage.Int8Blockwise, 256, 128)]
    public void IncompatibleSettings_AreRejected(OptimizerMomentStorage storage, int blockSize, int minLength)
    {
        using var reader = new BinaryReader(new MemoryStream(Header()));
        Assert.Throws<InvalidOperationException>(() => AdamWMomentFormatHeader.Validate(reader, storage, blockSize, minLength));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(-1)]
    public void TruncatedHeader_IsRejected(int length)
    {
        byte[] bytes = Header();
        Array.Resize(ref bytes, length < 0 ? bytes.Length - 1 : length);
        using var reader = new BinaryReader(new MemoryStream(bytes));
        Assert.Throws<InvalidOperationException>(() =>
            AdamWMomentFormatHeader.Validate(reader, OptimizerMomentStorage.Int8Blockwise, 256, 4096));
    }

    [Theory]
    [InlineData(OptimizerMomentStorage.BFloat16)]
    [InlineData(OptimizerMomentStorage.Int8Blockwise)]
    public void LegacyHeaderlessCheckpoint_CannotRestoreCompressedMoments(OptimizerMomentStorage storage)
    {
        using var reader = new BinaryReader(new MemoryStream());
        Assert.Throws<InvalidOperationException>(() => AdamWMomentFormatHeader.Validate(reader, storage, 256, 4096));
    }

    [Fact]
    public void LegacyHeaderlessFloat32Checkpoint_RemainsSupported()
    {
        using var reader = new BinaryReader(new MemoryStream());
        AdamWMomentFormatHeader.Validate(reader, OptimizerMomentStorage.Float32, 256, 4096);
    }

    [Fact]
    public void TrailingData_IsRejected()
    {
        var bytes = Header();
        Array.Resize(ref bytes, bytes.Length + 1);
        using var reader = new BinaryReader(new MemoryStream(bytes));
        Assert.Throws<InvalidOperationException>(() =>
            AdamWMomentFormatHeader.Validate(reader, OptimizerMomentStorage.Int8Blockwise, 256, 4096));
    }
}
