namespace AiDotNet.Logging;

/// <summary>
/// Image summary data.
/// </summary>
internal class ImageSummary
{
    public int Height { get; set; }
    public int Width { get; set; }
    public int Colorspace { get; set; }
    public byte[] EncodedData { get; set; } = Array.Empty<byte>();

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Field 1: height (int32)
        writer.Write((byte)0x08);
        VarintHelper.WriteVarint(writer, Height);

        // Field 2: width (int32)
        writer.Write((byte)0x10);
        VarintHelper.WriteVarint(writer, Width);

        // Field 3: colorspace (int32)
        writer.Write((byte)0x18);
        VarintHelper.WriteVarint(writer, Colorspace);

        // Field 4: encoded_image_string (bytes)
        writer.Write((byte)0x22);
        VarintHelper.WriteVarint(writer, EncodedData.Length);
        writer.Write(EncodedData);

        return ms.ToArray();
    }
}
