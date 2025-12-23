namespace AiDotNet.Logging;

/// <summary>
/// Summary containing multiple values.
/// </summary>
internal class Summary
{
    public List<SummaryValue> Values { get; } = new List<SummaryValue>();

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        foreach (var value in Values)
        {
            var valueBytes = value.ToBytes();
            writer.Write((byte)0x0A); // field 1, wire type 2
            VarintHelper.WriteVarint(writer, valueBytes.Length);
            writer.Write(valueBytes);
        }

        return ms.ToArray();
    }
}
