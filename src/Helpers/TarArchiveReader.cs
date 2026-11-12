using System.IO.Compression;
using System.Text;

namespace AiDotNet.Helpers;

/// <summary>
/// Reads members of a tar archive (POSIX ustar or GNU, optionally gzip-compressed), as NeMo's <c>.nemo</c> checkpoints
/// are packed; .NET Framework has no tar reader.
/// </summary>
internal static class TarArchiveReader
{
    private const int BlockSize = 512;

    /// <summary>Copies the first regular file whose name (without a leading <c>./</c>) is <paramref name="memberName"/> to
    /// <paramref name="destination"/>; false when the archive has none.</summary>
    public static bool TryExtract(string archivePath, string memberName, string destination)
    {
        using var file = File.OpenRead(archivePath);
        Stream stream = file;
        int first = file.ReadByte(), second = file.ReadByte();
        file.Position = 0;
        if (first == 0x1f && second == 0x8b) stream = new GZipStream(file, CompressionMode.Decompress);
        try
        {
            var header = new byte[BlockSize];
            string? longName = null;
            while (ReadExactly(stream, header))
            {
                if (header.All(b => b == 0)) return false;
                string name = longName ?? Field(header, 0, 100);
                longName = null;
                string prefix = Encoding.ASCII.GetString(header, 257, 5) == "ustar" ? Field(header, 345, 155) : "";
                if (prefix.Length > 0) name = prefix + "/" + name;
                long size = Octal(header, 124, 12);
                char type = (char)header[156];
                long padded = (size + BlockSize - 1) / BlockSize * BlockSize;
                if (type == 'L')
                {
                    var buffer = new byte[padded];
                    if (!ReadExactly(stream, buffer)) throw new InvalidDataException("The archive ends inside a long name.");
                    longName = Encoding.UTF8.GetString(buffer, 0, (int)size).TrimEnd('\0');
                    continue;
                }
                string normalized = name.StartsWith("./", StringComparison.Ordinal) ? name.Substring(2) : name;
                if ((type == '0' || type == '\0') && normalized == memberName)
                {
                    using (var output = File.Create(destination)) Copy(stream, output, size);
                    return true;
                }
                Skip(stream, padded);
            }
            return false;
        }
        finally
        {
            if (!ReferenceEquals(stream, file)) stream.Dispose();
        }
    }

    private static string Field(byte[] header, int offset, int length)
    {
        int end = offset;
        while (end < offset + length && header[end] != 0) end++;
        return Encoding.UTF8.GetString(header, offset, end - offset);
    }

    private static long Octal(byte[] header, int offset, int length)
    {
        // GNU's base-256 form for sizes past 8 GB: the high bit of the first byte set.
        if ((header[offset] & 0x80) != 0)
        {
            long value = header[offset] & 0x7f;
            for (int i = 1; i < length; i++) value = (value << 8) | header[offset + i];
            return value;
        }
        string text = Encoding.ASCII.GetString(header, offset, length).Trim('\0', ' ');
        return text.Length == 0 ? 0 : Convert.ToInt64(text, 8);
    }

    private static bool ReadExactly(Stream stream, byte[] buffer)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = stream.Read(buffer, read, buffer.Length - read);
            if (n == 0) return read == 0 ? false : throw new InvalidDataException("The archive ends inside a block.");
            read += n;
        }
        return true;
    }

    private static void Copy(Stream input, Stream output, long size)
    {
        var buffer = new byte[1 << 20];
        long left = size;
        while (left > 0)
        {
            int n = input.Read(buffer, 0, (int)Math.Min(buffer.Length, left));
            if (n == 0) throw new InvalidDataException("The archive ends inside a member.");
            output.Write(buffer, 0, n);
            left -= n;
        }
        Skip(input, (size + BlockSize - 1) / BlockSize * BlockSize - size);
    }

    private static void Skip(Stream stream, long count)
    {
        if (stream.CanSeek)
        {
            stream.Seek(count, SeekOrigin.Current);
            return;
        }
        var buffer = new byte[Math.Min(count, 1 << 20)];
        while (count > 0)
        {
            int n = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, count));
            if (n == 0) throw new InvalidDataException("The archive ends inside a member.");
            count -= n;
        }
    }
}
