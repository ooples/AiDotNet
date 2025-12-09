using System.Text;

namespace AiDotNet.Logging;

/// <summary>
/// Text summary data.
/// </summary>
internal class TextSummary
{
    public string Text { get; set; } = "";

    public byte[] ToBytes()
    {
        // Text is stored as a tensor with string dtype
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Simplified: just encode the text directly
        var textBytes = Encoding.UTF8.GetBytes(Text);

        // Field 1: dtype = DT_STRING (7)
        writer.Write((byte)0x08);
        VarintHelper.WriteVarint(writer, 7);

        // Field 4: string_val (repeated string)
        writer.Write((byte)0x22);
        VarintHelper.WriteVarint(writer, textBytes.Length);
        writer.Write(textBytes);

        return ms.ToArray();
    }
}
