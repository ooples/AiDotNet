using System;
using System.IO;
using System.IO.Compression;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Helpers;

/// <summary>
/// ImageHelper decodes PNG/JPEG/GIF/TGA/PSD/HDR through StbImageSharp on every target framework. The PNGs here are
/// built byte by byte, so the expected pixels are known exactly and the test needs no encoder.
/// </summary>
public class ImageHelperEncodedFormatTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "aidotnet-imagehelper-" + Guid.NewGuid().ToString("N"));

    public ImageHelperEncodedFormatTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); }
        catch (Exception ex) when (ex is IOException || ex is UnauthorizedAccessException)
        { /* best-effort temp cleanup; a locked file must not fail the test */ }
    }

    // 2x2 RGB: red, green / blue, (10, 20, 30).
    private static readonly byte[,] Pixels =
    {
        { 255, 0, 0 }, { 0, 255, 0 },
        { 0, 0, 255 }, { 10, 20, 30 },
    };

    private string WritePng(string name) => WritePng(name, declaredWidth: 2, declaredHeight: 2);

    /// <summary>
    /// The 2x2 image, with the IHDR able to declare other dimensions so a header can lie about its size.
    /// </summary>
    private string WritePng(string name, uint declaredWidth, uint declaredHeight)
    {
        var raw = new MemoryStream();
        for (int y = 0; y < 2; y++)
        {
            raw.WriteByte(0);   // filter type None
            for (int x = 0; x < 2; x++)
                for (int c = 0; c < 3; c++)
                    raw.WriteByte(Pixels[y * 2 + x, c]);
        }

        var png = new MemoryStream();
        png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
        var ihdr = new MemoryStream();
        WriteBigEndian(ihdr, declaredWidth);
        WriteBigEndian(ihdr, declaredHeight);
        ihdr.Write(new byte[] { 8, 2, 0, 0, 0 }, 0, 5);   // 8-bit, RGB
        WriteChunk(png, "IHDR", ihdr.ToArray());
        WriteChunk(png, "IDAT", Zlib(raw.ToArray()));
        WriteChunk(png, "IEND", Array.Empty<byte>());
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, png.ToArray());
        return path;
    }

    private static byte[] Zlib(byte[] data)
    {
        var output = new MemoryStream();
        output.WriteByte(0x78); output.WriteByte(0x01);
        using (var deflate = new DeflateStream(output, CompressionLevel.Optimal, leaveOpen: true))
            deflate.Write(data, 0, data.Length);
        uint a = 1, b = 0;
        foreach (byte value in data) { a = (a + value) % 65521; b = (b + a) % 65521; }
        uint adler = (b << 16) | a;
        output.WriteByte((byte)(adler >> 24)); output.WriteByte((byte)(adler >> 16));
        output.WriteByte((byte)(adler >> 8)); output.WriteByte((byte)adler);
        return output.ToArray();
    }

    private static void WriteChunk(Stream stream, string type, byte[] data)
    {
        WriteBigEndian(stream, (uint)data.Length);
        var typeBytes = System.Text.Encoding.ASCII.GetBytes(type);
        stream.Write(typeBytes, 0, 4);
        stream.Write(data, 0, data.Length);
        uint crc = 0xFFFFFFFF;
        foreach (byte value in typeBytes) crc = Crc(crc, value);
        foreach (byte value in data) crc = Crc(crc, value);
        WriteBigEndian(stream, crc ^ 0xFFFFFFFF);
    }

    private static uint Crc(uint crc, byte value)
    {
        crc ^= value;
        for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? 0xEDB88320 ^ (crc >> 1) : crc >> 1;
        return crc;
    }

    private static void WriteBigEndian(Stream stream, uint value)
    {
        stream.WriteByte((byte)(value >> 24)); stream.WriteByte((byte)(value >> 16));
        stream.WriteByte((byte)(value >> 8)); stream.WriteByte((byte)value);
    }

    [Fact]
    public void LoadImage_Png_DecodesEveryPixelIntoChannelPlanes()
    {
        var tensor = ImageHelper<double>.LoadImage(WritePng("pixels.png"), normalize: false);

        Assert.Equal(new[] { 1, 3, 2, 2 }, tensor.Shape.ToArray());
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(Pixels[y * 2 + x, c], tensor[0, c, y, x]);
    }

    [Fact]
    public void LoadImage_Png_NormalizesToUnitRange()
    {
        var tensor = ImageHelper<double>.LoadImage(WritePng("norm.png"), normalize: true);

        Assert.Equal(1.0, tensor[0, 0, 0, 0], 12);
        Assert.Equal(20.0 / 255.0, tensor[0, 1, 1, 1], 12);
    }

    [Fact]
    public void LoadImage_CorruptPng_ThrowsInvalidData()
    {
        string path = WritePng("corrupt.png");
        var bytes = File.ReadAllBytes(path);
        // Keep the signature and IHDR (so the format is recognised) and cut the compressed pixel data short:
        // the cut lands four bytes into IDAT's data, located from the chunk itself rather than assumed.
        int idatType = IndexOf(bytes, System.Text.Encoding.ASCII.GetBytes("IDAT"));
        Assert.True(idatType > 0, "the PNG has an IDAT chunk");
        int idatLength = (bytes[idatType - 4] << 24) | (bytes[idatType - 3] << 16) | (bytes[idatType - 2] << 8) | bytes[idatType - 1];
        Assert.True(idatLength > 4, "the cut must land inside the compressed data");
        Array.Resize(ref bytes, idatType + 4 + 4);
        File.WriteAllBytes(path, bytes);

        Assert.Throws<InvalidDataException>(() => ImageHelper<double>.LoadImage(path));
    }

    [Fact]
    public void LoadImage_HeaderDeclaringHugeDimensions_IsRejectedBeforeDecoding()
    {
        // 60000 x 60000 needs 14.4 GB of RGBA: more than any managed array. The file itself is tiny, so
        // only the header check stands between it and the decoder's allocation.
        string path = WritePng("huge.png", declaredWidth: 60000, declaredHeight: 60000);

        var error = Assert.Throws<InvalidDataException>(() => ImageHelper<double>.LoadImage(path));
        Assert.Contains("too large", error.Message);
    }

    private static int IndexOf(byte[] haystack, byte[] needle)
    {
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            int j = 0;
            while (j < needle.Length && haystack[i + j] == needle[j]) j++;
            if (j == needle.Length) return i;
        }
        return -1;
    }

    [Fact]
    public void LoadImage_Tiff_IsReportedAsUnsupported()
    {
        string path = Path.Combine(_dir, "image.tif");
        File.WriteAllBytes(path, new byte[] { 0x49, 0x49, 0x2A, 0x00, 8, 0, 0, 0, 0, 0 });

        var error = Assert.Throws<NotSupportedException>(() => ImageHelper<double>.LoadImage(path));
        Assert.Contains("Supported:", error.Message);
    }
}
