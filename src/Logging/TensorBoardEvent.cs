using System.Text;

namespace AiDotNet.Logging;

/// <summary>
/// TensorBoard event containing summary data.
/// </summary>
internal class TensorBoardEvent
{
    public double WallTime { get; set; }
    public long Step { get; set; }
    public string? FileVersion { get; set; }
    public Summary? Summary { get; set; }

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Simplified protobuf-like encoding
        // Field 1: wall_time (double)
        writer.Write((byte)0x09); // field 1, wire type 1 (64-bit)
        writer.Write(WallTime);

        // Field 2: step (int64)
        writer.Write((byte)0x10); // field 2, wire type 0 (varint)
        VarintHelper.WriteVarint(writer, Step);

        if (FileVersion != null)
        {
            // Field 3: file_version (string)
            var bytes = Encoding.UTF8.GetBytes(FileVersion);
            writer.Write((byte)0x1A); // field 3, wire type 2 (length-delimited)
            VarintHelper.WriteVarint(writer, bytes.Length);
            writer.Write(bytes);
        }

        if (Summary != null)
        {
            // Field 5: summary (message)
            var summaryBytes = Summary.ToBytes();
            writer.Write((byte)0x2A); // field 5, wire type 2
            VarintHelper.WriteVarint(writer, summaryBytes.Length);
            writer.Write(summaryBytes);
        }

        return ms.ToArray();
    }
}
