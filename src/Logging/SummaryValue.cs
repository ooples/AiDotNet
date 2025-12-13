using System.Text;

namespace AiDotNet.Logging;

/// <summary>
/// Individual summary value.
/// </summary>
internal class SummaryValue
{
    public string Tag { get; set; } = "";
    public float? SimpleValue { get; set; }
    public HistogramSummary? Histogram { get; set; }
    public ImageSummary? Image { get; set; }
    public TextSummary? Text { get; set; }

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Field 1: tag (string)
        var tagBytes = Encoding.UTF8.GetBytes(Tag);
        writer.Write((byte)0x0A); // field 1, wire type 2
        VarintHelper.WriteVarint(writer, tagBytes.Length);
        writer.Write(tagBytes);

        if (SimpleValue.HasValue)
        {
            // Field 2: simple_value (float)
            writer.Write((byte)0x15); // field 2, wire type 5 (32-bit)
            writer.Write(SimpleValue.Value);
        }

        if (Histogram != null)
        {
            // Field 4: histo (message)
            var histoBytes = Histogram.ToBytes();
            writer.Write((byte)0x22); // field 4, wire type 2
            VarintHelper.WriteVarint(writer, histoBytes.Length);
            writer.Write(histoBytes);
        }

        if (Image != null)
        {
            // Field 3: image (message)
            var imageBytes = Image.ToBytes();
            writer.Write((byte)0x1A); // field 3, wire type 2
            VarintHelper.WriteVarint(writer, imageBytes.Length);
            writer.Write(imageBytes);
        }

        if (Text != null)
        {
            // Field 8: tensor (with text plugin)
            var textBytes = Text.ToBytes();
            writer.Write((byte)0x42); // field 8, wire type 2
            VarintHelper.WriteVarint(writer, textBytes.Length);
            writer.Write(textBytes);
        }

        return ms.ToArray();
    }
}
