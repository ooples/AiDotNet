namespace AiDotNet.Logging;

/// <summary>
/// Helper for writing variable-length integers in protobuf format.
/// </summary>
internal static class VarintHelper
{
    /// <summary>
    /// Writes a variable-length integer to the binary writer.
    /// </summary>
    public static void WriteVarint(BinaryWriter writer, long value)
    {
        ulong v = (ulong)value;
        while (v >= 0x80)
        {
            writer.Write((byte)(v | 0x80));
            v >>= 7;
        }
        writer.Write((byte)v);
    }
}
