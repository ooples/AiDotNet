using System;
using System.IO;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// ChunkedMemoryStream replaces MemoryStream where a layer's whole state is serialized in memory to clone it: a
/// MemoryStream is capped at 2 GB and a large layer's clone threw "Array dimensions exceeded supported range".
/// These pin the stream semantics the clone path relies on, across block boundaries.
/// </summary>
public class ChunkedMemoryStreamTests
{
    private static byte[] Pattern(int length, int seed)
    {
        var data = new byte[length];
        new Random(seed).NextBytes(data);
        return data;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(64 * 1024)]            // exactly the first block
    [InlineData(64 * 1024 + 1)]        // one byte into the second block
    [InlineData(3 * 1024 * 1024 + 17)] // several doubling blocks
    public void Bytes_written_in_uneven_pieces_read_back_identically(int length)
    {
        var data = Pattern(length, length);
        using var stream = new ChunkedMemoryStream();
        int offset = 0, piece = 1;
        while (offset < length)
        {
            int n = Math.Min(piece, length - offset);
            stream.Write(data, offset, n);
            offset += n;
            piece = piece * 3 + 7;   // piece sizes that straddle block boundaries
        }
        Assert.Equal(length, stream.Length);

        stream.Position = 0;
        var read = new byte[length];
        int total = 0, got;
        while ((got = stream.Read(read, total, Math.Min(4099, length - total))) > 0) total += got;
        Assert.Equal(length, total);
        Assert.Equal(data, read);
        Assert.Equal(0, stream.Read(new byte[1], 0, 1));
    }

    [Fact]
    public void A_binary_writer_and_reader_round_trip_through_it_as_the_clone_path_does()
    {
        using var stream = new ChunkedMemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            for (int i = 0; i < 200_000; i++) writer.Write(i * 0.5);
            writer.Write("tail");
        }
        stream.Position = 0;
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        for (int i = 0; i < 200_000; i++) Assert.Equal(i * 0.5, reader.ReadDouble());
        Assert.Equal("tail", reader.ReadString());
        Assert.Equal(stream.Length, stream.Position);
    }

    [Fact]
    public void Seeking_reads_from_the_right_block_and_truncated_bytes_come_back_as_zero()
    {
        var data = Pattern(200_000, 5);
        using var stream = new ChunkedMemoryStream();
        stream.Write(data, 0, data.Length);

        stream.Seek(150_000, SeekOrigin.Begin);
        var slice = new byte[1000];
        Assert.Equal(1000, stream.Read(slice, 0, 1000));
        Assert.Equal(data.AsSpan(150_000, 1000).ToArray(), slice);

        stream.SetLength(100_000);
        stream.SetLength(120_000);
        stream.Position = 100_000;
        var extended = new byte[20_000];
        Assert.Equal(20_000, stream.Read(extended, 0, 20_000));
        Assert.All(extended, b => Assert.Equal(0, b));
    }
}
