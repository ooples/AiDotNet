namespace AiDotNet.Logging;

/// <summary>
/// Histogram summary data.
/// </summary>
internal class HistogramSummary
{
    public double Min { get; set; }
    public double Max { get; set; }
    public double Num { get; set; }
    public double Sum { get; set; }
    public double SumSquares { get; set; }
    public List<double> BucketLimits { get; } = new List<double>();
    public List<double> BucketCounts { get; } = new List<double>();

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Field 1: min (double)
        writer.Write((byte)0x09);
        writer.Write(Min);

        // Field 2: max (double)
        writer.Write((byte)0x11);
        writer.Write(Max);

        // Field 3: num (double)
        writer.Write((byte)0x19);
        writer.Write(Num);

        // Field 4: sum (double)
        writer.Write((byte)0x21);
        writer.Write(Sum);

        // Field 5: sum_squares (double)
        writer.Write((byte)0x29);
        writer.Write(SumSquares);

        // Field 6: bucket_limit (repeated double, packed)
        if (BucketLimits.Count > 0)
        {
            writer.Write((byte)0x32); // field 6, wire type 2 (packed)
            var limitsBytes = new byte[BucketLimits.Count * 8];
            for (int i = 0; i < BucketLimits.Count; i++)
            {
                var bytes = BitConverter.GetBytes(BucketLimits[i]);
                Array.Copy(bytes, 0, limitsBytes, i * 8, 8);
            }
            VarintHelper.WriteVarint(writer, limitsBytes.Length);
            writer.Write(limitsBytes);
        }

        // Field 7: bucket (repeated double, packed)
        if (BucketCounts.Count > 0)
        {
            writer.Write((byte)0x3A); // field 7, wire type 2 (packed)
            var countsBytes = new byte[BucketCounts.Count * 8];
            for (int i = 0; i < BucketCounts.Count; i++)
            {
                var bytes = BitConverter.GetBytes(BucketCounts[i]);
                Array.Copy(bytes, 0, countsBytes, i * 8, 8);
            }
            VarintHelper.WriteVarint(writer, countsBytes.Length);
            writer.Write(countsBytes);
        }

        return ms.ToArray();
    }
}
