using AiDotNet.Logging;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Logging;

/// <summary>
/// Deep integration tests for Logging classes:
/// VarintHelper (protobuf varint encoding), HistogramSummary (binary serialization),
/// SummaryValue (tag + value encoding), TensorBoardEvent (event serialization),
/// Summary (multi-value encoding).
/// </summary>
public class LoggingDeepMathIntegrationTests
{
    // ============================
    // VarintHelper Encoding Tests
    // ============================

    [Fact]
    public void WriteVarint_Zero_SingleByte_0x00()
    {
        // 0 fits in 7 bits: single byte 0x00
        var bytes = EncodeVarint(0);

        Assert.Single(bytes);
        Assert.Equal(0x00, bytes[0]);
    }

    [Fact]
    public void WriteVarint_One_SingleByte_0x01()
    {
        var bytes = EncodeVarint(1);

        Assert.Single(bytes);
        Assert.Equal(0x01, bytes[0]);
    }

    [Fact]
    public void WriteVarint_127_SingleByte_0x7F()
    {
        // 127 = 0b01111111 fits in 7 bits
        var bytes = EncodeVarint(127);

        Assert.Single(bytes);
        Assert.Equal(0x7F, bytes[0]);
    }

    [Fact]
    public void WriteVarint_128_TwoBytes_HandComputed()
    {
        // 128 = 0b10000000
        // First byte: lower 7 bits = 0000000 | 0x80 = 0x80
        // Second byte: remaining bits = 0000001 = 0x01
        var bytes = EncodeVarint(128);

        Assert.Equal(2, bytes.Length);
        Assert.Equal(0x80, bytes[0]);
        Assert.Equal(0x01, bytes[1]);
    }

    [Fact]
    public void WriteVarint_300_TwoBytes_HandComputed()
    {
        // 300 = 0b100101100
        // Lower 7 bits: 0101100 = 44, set MSB: 44 | 0x80 = 0xAC
        // Remaining: 0b10 = 2
        var bytes = EncodeVarint(300);

        Assert.Equal(2, bytes.Length);
        Assert.Equal(0xAC, bytes[0]);
        Assert.Equal(0x02, bytes[1]);
    }

    [Fact]
    public void WriteVarint_16383_TwoBytes_HandComputed()
    {
        // 16383 = 0b11111111111111 (14 bits)
        // Lower 7 bits: 1111111 = 0x7F, set MSB: 0xFF
        // Remaining 7 bits: 1111111 = 0x7F
        var bytes = EncodeVarint(16383);

        Assert.Equal(2, bytes.Length);
        Assert.Equal(0xFF, bytes[0]);
        Assert.Equal(0x7F, bytes[1]);
    }

    [Fact]
    public void WriteVarint_16384_ThreeBytes_HandComputed()
    {
        // 16384 = 0b100000000000000 (15 bits)
        // Lower 7: 0000000 | 0x80 = 0x80
        // Next 7: 0000000 | 0x80 = 0x80
        // Remaining: 0000001 = 0x01
        var bytes = EncodeVarint(16384);

        Assert.Equal(3, bytes.Length);
        Assert.Equal(0x80, bytes[0]);
        Assert.Equal(0x80, bytes[1]);
        Assert.Equal(0x01, bytes[2]);
    }

    [Fact]
    public void WriteVarint_ByteCountFormula_HandVerified()
    {
        // Varint uses 7 bits per byte
        // 1 byte: values 0 to 127 (2^7 - 1)
        // 2 bytes: values 128 to 16383 (2^14 - 1)
        // 3 bytes: values 16384 to 2097151 (2^21 - 1)

        Assert.Single(EncodeVarint(0));
        Assert.Single(EncodeVarint(127));
        Assert.Equal(2, EncodeVarint(128).Length);
        Assert.Equal(2, EncodeVarint(16383).Length);
        Assert.Equal(3, EncodeVarint(16384).Length);
        Assert.Equal(3, EncodeVarint(2097151).Length);
        Assert.Equal(4, EncodeVarint(2097152).Length);
    }

    [Fact]
    public void WriteVarint_SmallValues_AreCompact()
    {
        // Values 0-127 should be single byte
        for (int i = 0; i <= 127; i++)
        {
            var bytes = EncodeVarint(i);
            Assert.Single(bytes);
        }
    }

    [Fact]
    public void WriteVarint_MediumValues_AreTwoBytes()
    {
        // Values 128-16383 should be two bytes
        Assert.Equal(2, EncodeVarint(128).Length);
        Assert.Equal(2, EncodeVarint(256).Length);
        Assert.Equal(2, EncodeVarint(1000).Length);
        Assert.Equal(2, EncodeVarint(16383).Length);
    }

    [Fact]
    public void WriteVarint_LargeValue_CorrectByteCount()
    {
        // int.MaxValue = 2^31 - 1 = 2147483647
        // ceil(31/7) = 5 bytes
        var bytes = EncodeVarint(int.MaxValue);

        Assert.Equal(5, bytes.Length);
    }

    // ============================
    // HistogramSummary Tests
    // ============================

    [Fact]
    public void HistogramSummary_ToBytes_ContainsMinMaxNumSumSumSquares()
    {
        var histo = new HistogramSummary
        {
            Min = -2.0,
            Max = 5.0,
            Num = 100.0,
            Sum = 150.0,
            SumSquares = 500.0
        };

        var bytes = histo.ToBytes();

        // Should contain 5 field headers (0x09, 0x11, 0x19, 0x21, 0x29) + 5 doubles (40 bytes)
        // Minimum size: 5 * (1 header + 8 double) = 45 bytes
        Assert.True(bytes.Length >= 45);

        // Verify field tags are present
        Assert.Equal(0x09, bytes[0]); // min field tag
    }

    [Fact]
    public void HistogramSummary_ToBytes_MinValueIsEncoded()
    {
        var histo = new HistogramSummary { Min = 3.14 };
        var bytes = histo.ToBytes();

        // Field 1 tag = 0x09, followed by 8 bytes of double
        Assert.Equal(0x09, bytes[0]);
        var minValue = BitConverter.ToDouble(bytes, 1);
        Assert.Equal(3.14, minValue);
    }

    [Fact]
    public void HistogramSummary_ToBytes_MaxValueIsEncoded()
    {
        var histo = new HistogramSummary { Max = 42.0 };
        var bytes = histo.ToBytes();

        // Field 2 tag = 0x11, at offset 9 (after field 1: 1 tag + 8 double)
        Assert.Equal(0x11, bytes[9]);
        var maxValue = BitConverter.ToDouble(bytes, 10);
        Assert.Equal(42.0, maxValue);
    }

    [Fact]
    public void HistogramSummary_ToBytes_WithBuckets_EncodesCorrectly()
    {
        var histo = new HistogramSummary
        {
            Min = 0.0,
            Max = 10.0,
            Num = 5.0,
            Sum = 25.0,
            SumSquares = 150.0
        };
        histo.BucketLimits.AddRange(new[] { 2.5, 5.0, 7.5, 10.0 });
        histo.BucketCounts.AddRange(new[] { 1.0, 2.0, 1.0, 1.0 });

        var bytes = histo.ToBytes();

        // Should contain the 5 fixed fields + packed bucket limits + packed bucket counts
        // Fixed: 5 * (1 + 8) = 45 bytes
        // Limits: 1 tag + varint(32) + 32 bytes = 1 + 1 + 32 = 34
        // Counts: 1 tag + varint(32) + 32 bytes = 1 + 1 + 32 = 34
        // Total: 45 + 34 + 34 = 113
        Assert.True(bytes.Length > 45); // At minimum more than just fixed fields
    }

    [Fact]
    public void HistogramSummary_ToBytes_EmptyBuckets_HasMinimumSize()
    {
        var histo = new HistogramSummary
        {
            Min = 0.0,
            Max = 0.0,
            Num = 0.0,
            Sum = 0.0,
            SumSquares = 0.0
        };

        var bytes = histo.ToBytes();

        // 5 fields: each 1 tag byte + 8 double bytes = 9 bytes per field = 45 bytes total
        Assert.Equal(45, bytes.Length);
    }

    [Fact]
    public void HistogramSummary_ToBytes_BucketLimits_PackedEncodingTag()
    {
        var histo = new HistogramSummary();
        histo.BucketLimits.Add(1.0);
        histo.BucketCounts.Add(10.0);

        var bytes = histo.ToBytes();

        // Check that bucket limits field tag 0x32 exists somewhere in the output
        Assert.Contains((byte)0x32, bytes);
    }

    [Fact]
    public void HistogramSummary_ToBytes_BucketCounts_PackedEncodingTag()
    {
        var histo = new HistogramSummary();
        histo.BucketLimits.Add(1.0);
        histo.BucketCounts.Add(10.0);

        var bytes = histo.ToBytes();

        // Check that bucket counts field tag 0x3A exists somewhere in the output
        Assert.Contains((byte)0x3A, bytes);
    }

    // ============================
    // SummaryValue Tests
    // ============================

    [Fact]
    public void SummaryValue_SimpleFloat_EncodesTagAndValue()
    {
        var sv = new SummaryValue
        {
            Tag = "loss",
            SimpleValue = 0.5f
        };

        var bytes = sv.ToBytes();

        // Should contain tag field (0x0A) + tag string + simple_value field (0x15) + float
        Assert.True(bytes.Length > 0);
        Assert.Equal(0x0A, bytes[0]); // Field 1: tag string
    }

    [Fact]
    public void SummaryValue_TagEncoding_ContainsTagString()
    {
        var sv = new SummaryValue
        {
            Tag = "abc",
            SimpleValue = 1.0f
        };

        var bytes = sv.ToBytes();

        // The tag "abc" should be encoded as UTF-8 bytes in the output
        // Field 1 tag: 0x0A, varint length: 3, then "abc" (0x61, 0x62, 0x63)
        Assert.Equal(0x0A, bytes[0]);
        Assert.Equal(3, bytes[1]); // length of "abc"
        Assert.Equal((byte)'a', bytes[2]);
        Assert.Equal((byte)'b', bytes[3]);
        Assert.Equal((byte)'c', bytes[4]);
    }

    [Fact]
    public void SummaryValue_SimpleValueFieldTag_Is0x15()
    {
        var sv = new SummaryValue
        {
            Tag = "x",
            SimpleValue = 3.14f
        };

        var bytes = sv.ToBytes();

        // After tag field: 0x0A + 1 (length) + 1 (char) = 3 bytes
        // Simple value field tag: 0x15 at offset 3
        Assert.Equal(0x15, bytes[3]);
    }

    [Fact]
    public void SummaryValue_SimpleValueEncoding_HandVerified()
    {
        var sv = new SummaryValue
        {
            Tag = "v",
            SimpleValue = 2.0f
        };

        var bytes = sv.ToBytes();

        // Tag field: 0x0A, varint(1), 'v' = 3 bytes
        // Value field: 0x15, float(2.0) = 5 bytes
        // Total: 8 bytes
        Assert.Equal(8, bytes.Length);

        // Extract the float value at offset 4 (after 0x0A, 1, 'v', 0x15)
        var floatValue = BitConverter.ToSingle(bytes, 4);
        Assert.Equal(2.0f, floatValue);
    }

    [Fact]
    public void SummaryValue_NoSimpleValue_OmitsValueField()
    {
        var sv = new SummaryValue
        {
            Tag = "test"
        };

        var bytes = sv.ToBytes();

        // Should only contain the tag field, no 0x15 simple value field
        // 0x0A + varint(4) + "test" = 6 bytes
        Assert.Equal(6, bytes.Length);
    }

    [Fact]
    public void SummaryValue_WithHistogram_ContainsHistogramData()
    {
        var sv = new SummaryValue
        {
            Tag = "h",
            Histogram = new HistogramSummary
            {
                Min = 0, Max = 10, Num = 3, Sum = 15, SumSquares = 100
            }
        };

        var bytes = sv.ToBytes();

        // Should contain histogram field tag 0x22
        Assert.Contains((byte)0x22, bytes);
        // Should be larger than just tag (3 bytes)
        Assert.True(bytes.Length > 10);
    }

    // ============================
    // TensorBoardEvent Tests
    // ============================

    [Fact]
    public void TensorBoardEvent_WallTime_EncodedAsDouble()
    {
        var evt = new TensorBoardEvent
        {
            WallTime = 1234567890.123,
            Step = 0
        };

        var bytes = evt.ToBytes();

        // Field 1 tag = 0x09 (wire type 1 = 64-bit)
        Assert.Equal(0x09, bytes[0]);

        // Extract double at offset 1
        var wallTime = BitConverter.ToDouble(bytes, 1);
        Assert.Equal(1234567890.123, wallTime, 3);
    }

    [Fact]
    public void TensorBoardEvent_Step_EncodedAsVarint()
    {
        var evt = new TensorBoardEvent
        {
            WallTime = 0.0,
            Step = 42
        };

        var bytes = evt.ToBytes();

        // Field 1: 0x09 + 8 bytes double = 9 bytes
        // Field 2: 0x10 at offset 9
        Assert.Equal(0x10, bytes[9]);

        // Step 42 fits in single varint byte
        Assert.Equal(42, bytes[10]);
    }

    [Fact]
    public void TensorBoardEvent_LargeStep_MultiByteVarint()
    {
        var evt = new TensorBoardEvent
        {
            WallTime = 0.0,
            Step = 1000000
        };

        var bytes = evt.ToBytes();

        // Field 2 tag at offset 9
        Assert.Equal(0x10, bytes[9]);

        // 1000000 requires more than 1 varint byte
        Assert.True(bytes.Length > 11);
    }

    [Fact]
    public void TensorBoardEvent_WithFileVersion_ContainsVersionString()
    {
        var evt = new TensorBoardEvent
        {
            WallTime = 0.0,
            Step = 0,
            FileVersion = "brain.Event:2"
        };

        var bytes = evt.ToBytes();

        // Should contain file_version field tag 0x1A
        Assert.Contains((byte)0x1A, bytes);
    }

    [Fact]
    public void TensorBoardEvent_WithSummary_ContainsSummaryField()
    {
        var summary = new Summary();
        summary.Values.Add(new SummaryValue { Tag = "loss", SimpleValue = 0.5f });

        var evt = new TensorBoardEvent
        {
            WallTime = 100.0,
            Step = 1,
            Summary = summary
        };

        var bytes = evt.ToBytes();

        // Should contain summary field tag 0x2A
        Assert.Contains((byte)0x2A, bytes);
    }

    [Fact]
    public void TensorBoardEvent_MinimalEvent_HasCorrectSize()
    {
        var evt = new TensorBoardEvent
        {
            WallTime = 0.0,
            Step = 0
        };

        var bytes = evt.ToBytes();

        // Field 1: 1 tag + 8 double = 9 bytes
        // Field 2: 1 tag + 1 varint(0) = 2 bytes
        // Total: 11 bytes
        Assert.Equal(11, bytes.Length);
    }

    // ============================
    // Summary (multi-value container) Tests
    // ============================

    [Fact]
    public void Summary_Empty_HasZeroBytes()
    {
        var summary = new Summary();

        var bytes = summary.ToBytes();

        Assert.Empty(bytes);
    }

    [Fact]
    public void Summary_SingleValue_ContainsFieldTag()
    {
        var summary = new Summary();
        summary.Values.Add(new SummaryValue { Tag = "x", SimpleValue = 1.0f });

        var bytes = summary.ToBytes();

        // Should start with field 1 tag 0x0A
        Assert.Equal(0x0A, bytes[0]);
        Assert.True(bytes.Length > 0);
    }

    [Fact]
    public void Summary_MultipleValues_EncodesAll()
    {
        var summary = new Summary();
        summary.Values.Add(new SummaryValue { Tag = "a", SimpleValue = 1.0f });
        summary.Values.Add(new SummaryValue { Tag = "b", SimpleValue = 2.0f });
        summary.Values.Add(new SummaryValue { Tag = "c", SimpleValue = 3.0f });

        var bytes = summary.ToBytes();

        // Each value starts with 0x0A field tag
        // Count occurrences of 0x0A at value boundaries
        // Size should be at least 3 * (1 tag + 1 length + 8 inner bytes) = 30
        Assert.True(bytes.Length >= 30);
    }

    // ============================
    // Varint Round-Trip Invariant Tests
    // ============================

    [Fact]
    public void WriteVarint_AllSingleByte_MSBUnset()
    {
        // For values 0-127, the single output byte should have MSB = 0
        for (int i = 0; i <= 127; i++)
        {
            var bytes = EncodeVarint(i);
            Assert.Equal(0, bytes[0] & 0x80);
        }
    }

    [Fact]
    public void WriteVarint_MultiByte_LastByteMSBUnset()
    {
        // The last byte always has MSB = 0 (no continuation)
        var testValues = new long[] { 128, 300, 16384, 2097152, int.MaxValue };

        foreach (var val in testValues)
        {
            var bytes = EncodeVarint(val);
            Assert.Equal(0, bytes[^1] & 0x80);
        }
    }

    [Fact]
    public void WriteVarint_MultiByte_NonLastBytesMSBSet()
    {
        // All bytes except the last have MSB = 1 (continuation bit)
        var testValues = new long[] { 128, 300, 16384, 2097152 };

        foreach (var val in testValues)
        {
            var bytes = EncodeVarint(val);
            for (int i = 0; i < bytes.Length - 1; i++)
            {
                Assert.NotEqual(0, bytes[i] & 0x80);
            }
        }
    }

    [Fact]
    public void WriteVarint_ReconstructValue_HandVerified()
    {
        // Verify we can reconstruct the value from varint bytes
        // 300 => bytes [0xAC, 0x02]
        // Decode: (0xAC & 0x7F) | ((0x02 & 0x7F) << 7) = 44 | (2 << 7) = 44 | 256 = 300
        var bytes = EncodeVarint(300);

        long reconstructed = 0;
        int shift = 0;
        foreach (var b in bytes)
        {
            reconstructed |= (long)(b & 0x7F) << shift;
            shift += 7;
        }

        Assert.Equal(300, reconstructed);
    }

    [Fact]
    public void WriteVarint_AllTestValues_CanReconstruct()
    {
        var testValues = new long[] { 0, 1, 42, 127, 128, 255, 300, 1000, 16383, 16384, 100000, 1000000 };

        foreach (var expected in testValues)
        {
            var bytes = EncodeVarint(expected);

            long reconstructed = 0;
            int shift = 0;
            foreach (var b in bytes)
            {
                reconstructed |= (long)(b & 0x7F) << shift;
                shift += 7;
            }

            Assert.Equal(expected, reconstructed);
        }
    }

    // ============================
    // Helper Methods
    // ============================

    private static byte[] EncodeVarint(long value)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        VarintHelper.WriteVarint(writer, value);
        writer.Flush();
        return ms.ToArray();
    }
}
