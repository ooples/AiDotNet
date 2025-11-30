using System.Text;

namespace AiDotNet.Logging;

/// <summary>
/// Low-level TensorBoard event file writer.
/// </summary>
/// <remarks>
/// <para>
/// TensorBoard event files use a specific binary format consisting of records.
/// Each record contains: length (8 bytes), masked CRC of length (4 bytes),
/// data (variable), and masked CRC of data (4 bytes).
/// </para>
/// <para><b>For Beginners:</b> TensorBoard is a visualization tool from TensorFlow.
///
/// This writer creates event files that TensorBoard can read and display.
/// It's like writing a diary in a specific format that TensorBoard knows
/// how to read and show as beautiful charts and graphs.
///
/// Event files contain:
/// - Scalar values (loss, accuracy over time)
/// - Histograms (weight distributions)
/// - Images (sample outputs, feature maps)
/// - Text (descriptions, annotations)
/// - Graphs (model architecture)
/// </para>
/// </remarks>
public class TensorBoardWriter : IDisposable
{
    private readonly string _logDir;
    private readonly FileStream _stream;
    private readonly BinaryWriter _writer;
    private readonly object _lock = new();
    private readonly string _fileName;
    private bool _disposed;

    /// <summary>
    /// Gets the log directory path.
    /// </summary>
    public string LogDir => _logDir;

    /// <summary>
    /// Gets the event file path.
    /// </summary>
    public string FilePath => Path.Combine(_logDir, _fileName);

    /// <summary>
    /// Creates a new TensorBoard event file writer.
    /// </summary>
    /// <param name="logDir">Directory to write event files.</param>
    /// <param name="filename">Optional filename prefix. Uses default format if not specified.</param>
    public TensorBoardWriter(string logDir, string? filename = null)
    {
        _logDir = logDir ?? throw new ArgumentNullException(nameof(logDir));

        // Create directory if it doesn't exist
        Directory.CreateDirectory(_logDir);

        // Generate filename: events.out.tfevents.{timestamp}.{hostname}
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var hostname = Environment.MachineName.ToLowerInvariant();
        _fileName = filename ?? $"events.out.tfevents.{timestamp}.{hostname}";

        var filePath = Path.Combine(_logDir, _fileName);
        _stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.Read);
        _writer = new BinaryWriter(_stream);

        // Write initial file version event
        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = 0,
            FileVersion = "brain.Event:2"
        });
    }

    /// <summary>
    /// Writes a scalar summary to the event file.
    /// </summary>
    /// <param name="tag">The tag name for this scalar (e.g., "loss/train").</param>
    /// <param name="value">The scalar value.</param>
    /// <param name="step">The global step number.</param>
    public void WriteScalar(string tag, float value, long step)
    {
        var summary = new Summary();
        summary.Values.Add(new SummaryValue
        {
            Tag = tag,
            SimpleValue = value
        });

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes multiple scalars as a group.
    /// </summary>
    /// <param name="mainTag">Main tag prefix.</param>
    /// <param name="tagValuePairs">Dictionary of tag suffixes to values.</param>
    /// <param name="step">The global step number.</param>
    public void WriteScalars(string mainTag, Dictionary<string, float> tagValuePairs, long step)
    {
        var summary = new Summary();
        foreach (var (subTag, value) in tagValuePairs)
        {
            summary.Values.Add(new SummaryValue
            {
                Tag = $"{mainTag}/{subTag}",
                SimpleValue = value
            });
        }

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes a histogram summary.
    /// </summary>
    /// <param name="tag">The tag name for this histogram.</param>
    /// <param name="values">Array of values to create histogram from.</param>
    /// <param name="step">The global step number.</param>
    public void WriteHistogram(string tag, float[] values, long step)
    {
        if (values == null || values.Length == 0)
            return;

        var histogram = CreateHistogram(values);
        var summary = new Summary();
        summary.Values.Add(new SummaryValue
        {
            Tag = tag,
            Histogram = histogram
        });

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes a histogram summary from a tensor.
    /// </summary>
    /// <param name="tag">The tag name for this histogram.</param>
    /// <param name="values">Span of values to create histogram from.</param>
    /// <param name="step">The global step number.</param>
    public void WriteHistogram(string tag, ReadOnlySpan<float> values, long step)
    {
        if (values.IsEmpty)
            return;

        var histogram = CreateHistogram(values.ToArray());
        var summary = new Summary();
        summary.Values.Add(new SummaryValue
        {
            Tag = tag,
            Histogram = histogram
        });

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes an image summary.
    /// </summary>
    /// <param name="tag">The tag name for this image.</param>
    /// <param name="imageData">PNG-encoded image data.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="width">Image width in pixels.</param>
    /// <param name="step">The global step number.</param>
    public void WriteImage(string tag, byte[] imageData, int height, int width, long step)
    {
        var image = new ImageSummary
        {
            Height = height,
            Width = width,
            Colorspace = 3, // RGB
            EncodedData = imageData
        };

        var summary = new Summary();
        summary.Values.Add(new SummaryValue
        {
            Tag = tag,
            Image = image
        });

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes raw image data (HWC format, values 0-255).
    /// </summary>
    /// <param name="tag">The tag name for this image.</param>
    /// <param name="pixels">Raw pixel data in HWC format (height x width x channels).</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="channels">Number of channels (1=grayscale, 3=RGB, 4=RGBA).</param>
    /// <param name="step">The global step number.</param>
    public void WriteImageRaw(string tag, byte[] pixels, int height, int width, int channels, long step)
    {
        // Encode as simple PNG
        var pngData = EncodePng(pixels, height, width, channels);
        WriteImage(tag, pngData, height, width, step);
    }

    /// <summary>
    /// Writes text summary.
    /// </summary>
    /// <param name="tag">The tag name for this text.</param>
    /// <param name="text">The text content.</param>
    /// <param name="step">The global step number.</param>
    public void WriteText(string tag, string text, long step)
    {
        var textSummary = new TextSummary
        {
            Text = text
        };

        var summary = new Summary();
        summary.Values.Add(new SummaryValue
        {
            Tag = tag,
            Text = textSummary
        });

        WriteEvent(new TensorBoardEvent
        {
            WallTime = GetWallTime(),
            Step = step,
            Summary = summary
        });
    }

    /// <summary>
    /// Writes an embedding with optional metadata and sprite.
    /// </summary>
    /// <param name="tag">The tag name for this embedding.</param>
    /// <param name="embeddings">2D array of embeddings (samples x dimensions).</param>
    /// <param name="metadata">Optional metadata labels for each sample.</param>
    /// <param name="step">The global step number.</param>
    public void WriteEmbedding(string tag, float[,] embeddings, string[]? metadata, long step)
    {
        // TensorBoard embeddings require writing separate files
        // Save embeddings as TSV
        var embeddingsPath = Path.Combine(_logDir, $"{tag}_embeddings.tsv");
        using (var writer = new StreamWriter(embeddingsPath))
        {
            int samples = embeddings.GetLength(0);
            int dims = embeddings.GetLength(1);

            for (int i = 0; i < samples; i++)
            {
                var values = new string[dims];
                for (int j = 0; j < dims; j++)
                {
                    values[j] = embeddings[i, j].ToString("G");
                }
                writer.WriteLine(string.Join("\t", values));
            }
        }

        // Save metadata if provided
        if (metadata != null)
        {
            var metadataPath = Path.Combine(_logDir, $"{tag}_metadata.tsv");
            File.WriteAllLines(metadataPath, metadata);
        }

        // Write projector config
        WriteProjectorConfig(tag, embeddings.GetLength(1), metadata != null);
    }

    /// <summary>
    /// Flushes pending writes to disk.
    /// </summary>
    public void Flush()
    {
        lock (_lock)
        {
            _writer.Flush();
            _stream.Flush();
        }
    }

    /// <summary>
    /// Releases resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        Flush();
        _writer.Dispose();
        _stream.Dispose();
    }

    private void WriteEvent(TensorBoardEvent evt)
    {
        var data = evt.ToBytes();
        lock (_lock)
        {
            WriteRecord(data);
        }
    }

    private void WriteRecord(byte[] data)
    {
        // TensorBoard record format:
        // uint64 length
        // uint32 masked_crc32_of_length
        // byte   data[length]
        // uint32 masked_crc32_of_data

        var length = (ulong)data.Length;
        var lengthBytes = BitConverter.GetBytes(length);

        _writer.Write(lengthBytes);
        _writer.Write(MaskedCrc32(lengthBytes));
        _writer.Write(data);
        _writer.Write(MaskedCrc32(data));
    }

    private static uint MaskedCrc32(byte[] data)
    {
        var crc = Crc32C(data);
        return ((crc >> 15) | (crc << 17)) + 0xa282ead8;
    }

    private static uint Crc32C(byte[] data)
    {
        // CRC32C (Castagnoli) implementation
        uint crc = 0xFFFFFFFF;
        foreach (var b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
            {
                crc = (crc >> 1) ^ (0x82F63B78 * (crc & 1));
            }
        }
        return crc ^ 0xFFFFFFFF;
    }

    private static double GetWallTime()
    {
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() / 1000.0;
    }

    private static HistogramSummary CreateHistogram(float[] values)
    {
        // Sort values for percentile calculation
        var sorted = values.OrderBy(v => v).ToArray();
        int n = sorted.Length;

        var histogram = new HistogramSummary
        {
            Min = sorted[0],
            Max = sorted[n - 1],
            Num = n,
            Sum = values.Sum(),
            SumSquares = values.Sum(v => (double)v * v)
        };

        // Create bucket limits using exponential spacing
        var bucketLimits = GenerateBucketLimits(histogram.Min, histogram.Max);
        histogram.BucketLimits.AddRange(bucketLimits);

        // Count values in each bucket
        var bucketCounts = new double[bucketLimits.Count];
        int bucketIndex = 0;
        foreach (var value in sorted)
        {
            while (bucketIndex < bucketLimits.Count - 1 && value > bucketLimits[bucketIndex])
            {
                bucketIndex++;
            }
            bucketCounts[bucketIndex]++;
        }
        histogram.BucketCounts.AddRange(bucketCounts);

        return histogram;
    }

    private static List<double> GenerateBucketLimits(double min, double max)
    {
        // Generate ~30 buckets with exponential spacing
        var limits = new List<double>();

        if (min >= max)
        {
            limits.Add(min);
            limits.Add(min + 1);
            return limits;
        }

        // Handle negative values
        if (min < 0)
        {
            // Add negative buckets
            double negMax = Math.Abs(min);
            double step = Math.Pow(negMax, 1.0 / 15);
            for (int i = 15; i >= 1; i--)
            {
                limits.Add(-Math.Pow(step, i));
            }
        }

        // Add zero if range spans it
        if (min <= 0 && max >= 0)
        {
            limits.Add(0);
        }

        // Add positive buckets
        if (max > 0)
        {
            double posMax = max;
            double step = Math.Pow(posMax, 1.0 / 15);
            for (int i = 1; i <= 15; i++)
            {
                limits.Add(Math.Pow(step, i));
            }
        }

        // Ensure proper bounds
        if (limits.Count == 0 || limits[0] > min)
            limits.Insert(0, min);
        if (limits[^1] < max)
            limits.Add(max);

        return limits.Distinct().OrderBy(x => x).ToList();
    }

    private static byte[] EncodePng(byte[] pixels, int height, int width, int channels)
    {
        // Minimal PNG encoder for uncompressed data
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // PNG signature
        writer.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });

        // IHDR chunk
        var ihdr = new byte[13];
        WriteBigEndianInt32(ihdr, 0, width);
        WriteBigEndianInt32(ihdr, 4, height);
        ihdr[8] = 8; // bit depth
        ihdr[9] = (byte)(channels == 1 ? 0 : (channels == 4 ? 6 : 2)); // color type
        ihdr[10] = 0; // compression
        ihdr[11] = 0; // filter
        ihdr[12] = 0; // interlace
        WriteChunk(writer, "IHDR", ihdr);

        // IDAT chunk (uncompressed for simplicity)
        // Each row needs filter byte (0 = none)
        var rowSize = width * channels + 1;
        var imageData = new byte[height * rowSize];
        for (int y = 0; y < height; y++)
        {
            imageData[y * rowSize] = 0; // filter byte
            Array.Copy(pixels, y * width * channels, imageData, y * rowSize + 1, width * channels);
        }

        // Compress with deflate (zlib format)
        var compressed = CompressZlib(imageData);
        WriteChunk(writer, "IDAT", compressed);

        // IEND chunk
        WriteChunk(writer, "IEND", Array.Empty<byte>());

        return ms.ToArray();
    }

    private static void WriteChunk(BinaryWriter writer, string type, byte[] data)
    {
        var typeBytes = Encoding.ASCII.GetBytes(type);
        var lengthBytes = new byte[4];
        WriteBigEndianInt32(lengthBytes, 0, data.Length);

        writer.Write(lengthBytes);
        writer.Write(typeBytes);
        writer.Write(data);

        // CRC32 of type + data
        var crcData = new byte[4 + data.Length];
        Array.Copy(typeBytes, 0, crcData, 0, 4);
        Array.Copy(data, 0, crcData, 4, data.Length);
        var crc = Crc32Png(crcData);
        var crcBytes = new byte[4];
        WriteBigEndianInt32(crcBytes, 0, (int)crc);
        writer.Write(crcBytes);
    }

    private static void WriteBigEndianInt32(byte[] buffer, int offset, int value)
    {
        buffer[offset] = (byte)(value >> 24);
        buffer[offset + 1] = (byte)(value >> 16);
        buffer[offset + 2] = (byte)(value >> 8);
        buffer[offset + 3] = (byte)value;
    }

    private static byte[] CompressZlib(byte[] data)
    {
        using var output = new MemoryStream();

        // Zlib header
        output.WriteByte(0x78); // CMF: deflate, 32K window
        output.WriteByte(0x9C); // FLG: default compression

        // Deflate data (stored blocks, uncompressed for simplicity)
        int offset = 0;
        while (offset < data.Length)
        {
            int blockSize = Math.Min(65535, data.Length - offset);
            bool lastBlock = offset + blockSize >= data.Length;

            output.WriteByte((byte)(lastBlock ? 0x01 : 0x00)); // BFINAL, BTYPE=00 (stored)
            output.WriteByte((byte)(blockSize & 0xFF));
            output.WriteByte((byte)((blockSize >> 8) & 0xFF));
            output.WriteByte((byte)(~blockSize & 0xFF));
            output.WriteByte((byte)((~blockSize >> 8) & 0xFF));
            output.Write(data, offset, blockSize);

            offset += blockSize;
        }

        // Adler-32 checksum
        uint adler = Adler32(data);
        output.WriteByte((byte)(adler >> 24));
        output.WriteByte((byte)(adler >> 16));
        output.WriteByte((byte)(adler >> 8));
        output.WriteByte((byte)adler);

        return output.ToArray();
    }

    private static uint Adler32(byte[] data)
    {
        uint a = 1, b = 0;
        foreach (var d in data)
        {
            a = (a + d) % 65521;
            b = (b + a) % 65521;
        }
        return (b << 16) | a;
    }

    private static uint Crc32Png(byte[] data)
    {
        // CRC32 (ISO 3309) for PNG
        uint crc = 0xFFFFFFFF;
        foreach (var b in data)
        {
            crc ^= b;
            for (int i = 0; i < 8; i++)
            {
                crc = (crc >> 1) ^ (0xEDB88320 * (crc & 1));
            }
        }
        return crc ^ 0xFFFFFFFF;
    }

    private void WriteProjectorConfig(string tag, int dimensions, bool hasMetadata)
    {
        var configPath = Path.Combine(_logDir, "projector_config.pbtxt");
        var configBuilder = new StringBuilder();

        // Read existing config if present
        if (File.Exists(configPath))
        {
            configBuilder.Append(File.ReadAllText(configPath));
        }

        // Append new embedding config
        configBuilder.AppendLine("embeddings {");
        configBuilder.AppendLine($"  tensor_name: \"{tag}\"");
        configBuilder.AppendLine($"  tensor_path: \"{tag}_embeddings.tsv\"");
        if (hasMetadata)
        {
            configBuilder.AppendLine($"  metadata_path: \"{tag}_metadata.tsv\"");
        }
        configBuilder.AppendLine("}");

        File.WriteAllText(configPath, configBuilder.ToString());
    }
}

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
        WriteVarint(writer, Step);

        if (FileVersion != null)
        {
            // Field 3: file_version (string)
            var bytes = Encoding.UTF8.GetBytes(FileVersion);
            writer.Write((byte)0x1A); // field 3, wire type 2 (length-delimited)
            WriteVarint(writer, bytes.Length);
            writer.Write(bytes);
        }

        if (Summary != null)
        {
            // Field 5: summary (message)
            var summaryBytes = Summary.ToBytes();
            writer.Write((byte)0x2A); // field 5, wire type 2
            WriteVarint(writer, summaryBytes.Length);
            writer.Write(summaryBytes);
        }

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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

/// <summary>
/// Summary containing multiple values.
/// </summary>
internal class Summary
{
    public List<SummaryValue> Values { get; } = new();

    public byte[] ToBytes()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        foreach (var value in Values)
        {
            var valueBytes = value.ToBytes();
            writer.Write((byte)0x0A); // field 1, wire type 2
            WriteVarint(writer, valueBytes.Length);
            writer.Write(valueBytes);
        }

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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
        WriteVarint(writer, tagBytes.Length);
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
            WriteVarint(writer, histoBytes.Length);
            writer.Write(histoBytes);
        }

        if (Image != null)
        {
            // Field 3: image (message)
            var imageBytes = Image.ToBytes();
            writer.Write((byte)0x1A); // field 3, wire type 2
            WriteVarint(writer, imageBytes.Length);
            writer.Write(imageBytes);
        }

        if (Text != null)
        {
            // Field 8: tensor (with text plugin)
            var textBytes = Text.ToBytes();
            writer.Write((byte)0x42); // field 8, wire type 2
            WriteVarint(writer, textBytes.Length);
            writer.Write(textBytes);
        }

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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
    public List<double> BucketLimits { get; } = new();
    public List<double> BucketCounts { get; } = new();

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
            WriteVarint(writer, limitsBytes.Length);
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
            WriteVarint(writer, countsBytes.Length);
            writer.Write(countsBytes);
        }

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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
        WriteVarint(writer, Height);

        // Field 2: width (int32)
        writer.Write((byte)0x10);
        WriteVarint(writer, Width);

        // Field 3: colorspace (int32)
        writer.Write((byte)0x18);
        WriteVarint(writer, Colorspace);

        // Field 4: encoded_image_string (bytes)
        writer.Write((byte)0x22);
        WriteVarint(writer, EncodedData.Length);
        writer.Write(EncodedData);

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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
        WriteVarint(writer, 7);

        // Field 4: string_val (repeated string)
        writer.Write((byte)0x22);
        WriteVarint(writer, textBytes.Length);
        writer.Write(textBytes);

        return ms.ToArray();
    }

    private static void WriteVarint(BinaryWriter writer, long value)
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
