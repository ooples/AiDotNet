using System.IO.Compression;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Reads samples from TAR archives with sequential I/O, inspired by the WebDataset format.
/// </summary>
/// <remarks>
/// <para>
/// WebDataset stores training data in TAR archives where each sample is a group of files
/// sharing the same basename (e.g., "000001.jpg", "000001.txt", "000001.json"). This format
/// enables efficient sequential I/O and is widely used for large-scale training with cloud storage.
/// </para>
/// <para>
/// Each sample is returned as a dictionary mapping extensions to byte arrays. Consumers
/// can then decode the bytes as needed (images, text, JSON, etc.).
/// </para>
/// </remarks>
internal class WebDataset : IDisposable
{
    private readonly string[] _tarPaths;
    private readonly WebDatasetOptions _options;
    private readonly HashSet<string>? _normalizedExtensions;
    private bool _disposed;

    /// <summary>
    /// Creates a new WebDataset from one or more TAR files.
    /// </summary>
    /// <param name="tarPaths">Paths to the TAR archive files (shards).</param>
    /// <param name="options">Optional configuration.</param>
    public WebDataset(string[] tarPaths, WebDatasetOptions? options = null)
    {
        if (tarPaths is null || tarPaths.Length == 0)
            throw new ArgumentException("At least one TAR path is required.", nameof(tarPaths));

        _tarPaths = tarPaths;
        _options = options ?? new WebDatasetOptions();

        // Normalize extensions once for case-insensitive comparison
        if (_options.IncludeExtensions is not null)
        {
            _normalizedExtensions = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var ext in _options.IncludeExtensions)
            {
                if (string.IsNullOrWhiteSpace(ext))
                    continue;

                var normalizedExt = ext.Trim();
                if (!normalizedExt.StartsWith("."))
                    normalizedExt = "." + normalizedExt;

                _normalizedExtensions.Add(normalizedExt);
            }
        }
    }

    /// <summary>
    /// Creates a new WebDataset from a single TAR file.
    /// </summary>
    public WebDataset(string tarPath, WebDatasetOptions? options = null)
        : this(new[] { tarPath }, options)
    {
    }

    /// <summary>
    /// Iterates through all samples across all TAR shards.
    /// </summary>
    /// <returns>An enumerable of samples, each a dictionary mapping file extension to file bytes.</returns>
    public IEnumerable<Dictionary<string, byte[]>> ReadSamples()
    {
        int samplesRead = 0;

        if (_options.Shuffle)
        {
            // Buffer-based shuffle: accumulate samples in a buffer, shuffle, yield, repeat
            var random = _options.Seed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
                : RandomHelper.CreateSecureRandom();

            var buffer = new List<Dictionary<string, byte[]>>();

            foreach (var sample in ReadSamplesRaw())
            {
                if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                    yield break;

                buffer.Add(sample);

                if (buffer.Count >= _options.ShuffleBufferSize)
                {
                    // Fisher-Yates shuffle
                    for (int i = buffer.Count - 1; i > 0; i--)
                    {
                        int j = random.Next(i + 1);
                        var temp = buffer[i];
                        buffer[i] = buffer[j];
                        buffer[j] = temp;
                    }

                    foreach (var item in buffer)
                    {
                        yield return item;
                        samplesRead++;
                        if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                            yield break;
                    }
                    buffer.Clear();
                }
            }

            // Flush remaining buffer
            if (buffer.Count > 0)
            {
                for (int i = buffer.Count - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    var temp = buffer[i];
                    buffer[i] = buffer[j];
                    buffer[j] = temp;
                }

                foreach (var item in buffer)
                {
                    if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                        yield break;
                    yield return item;
                    samplesRead++;
                }
            }
        }
        else
        {
            foreach (var sample in ReadSamplesRaw())
            {
                if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                    yield break;
                yield return sample;
                samplesRead++;
            }
        }
    }

    private IEnumerable<Dictionary<string, byte[]>> ReadSamplesRaw()
    {
        foreach (string tarPath in _tarPaths)
        {
            Stream tarStream;
            if (tarPath.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ||
                tarPath.EndsWith(".tgz", StringComparison.OrdinalIgnoreCase))
            {
                var fileStream = new FileStream(tarPath, FileMode.Open, FileAccess.Read, FileShare.Read);
                tarStream = new GZipStream(fileStream, CompressionMode.Decompress);
            }
            else
            {
                tarStream = new FileStream(tarPath, FileMode.Open, FileAccess.Read, FileShare.Read);
            }

            using (tarStream)
            {
                foreach (var sample in ReadTarSamples(tarStream))
                {
                    yield return sample;
                }
            }
        }
    }

    private IEnumerable<Dictionary<string, byte[]>> ReadTarSamples(Stream stream)
    {
        byte[] header = new byte[512];
        var currentSample = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
        string? currentBaseName = null;

        while (true)
        {
            int bytesRead = ReadFull(stream, header, 512);
            if (bytesRead == 0) break;
            if (bytesRead < 512)
            {
                throw new InvalidDataException("TAR header is truncated.");
            }

            // Check for zero block
            bool allZero = true;
            for (int i = 0; i < 512; i++)
            {
                if (header[i] != 0) { allZero = false; break; }
            }
            if (allZero) break;

            string name = ReadTarString(header, 0, 100);
            char typeFlag = (char)header[156];
            long size = ReadTarOctal(header, 124, 12);

            // UStar prefix
            string prefix = ReadTarString(header, 345, 155);
            if (prefix.Length > 0) name = prefix + "/" + name;

            if (typeFlag == '0' || typeFlag == '\0')
            {
                // Guard against TAR entries larger than int.MaxValue
                if (size > int.MaxValue)
                {
                    throw new InvalidOperationException(
                        $"TAR entry '{name}' has size {size} bytes which exceeds the maximum supported size of {int.MaxValue} bytes.");
                }

                // Regular file - read contents
                byte[] fileData = new byte[(int)size];
                int entryBytesRead = ReadFull(stream, fileData, (int)size);
                if (entryBytesRead < (int)size)
                {
                    throw new InvalidDataException(
                        $"TAR entry '{name}' is truncated: expected {size} bytes but only read {entryBytesRead}.");
                }

                // Skip padding
                long remainder = size % 512;
                if (remainder > 0)
                {
                    int padSize = (int)(512 - remainder);
                    byte[] pad = new byte[padSize];
                    int padRead = ReadFull(stream, pad, padSize);
                    if (padRead < padSize)
                    {
                        throw new InvalidDataException(
                            $"TAR entry '{name}' padding is truncated: expected {padSize} bytes but only read {padRead}.");
                    }
                }

                string ext = Path.GetExtension(name);
                // Use directory-qualified basename to avoid collisions across directories
                string nameWithoutExt = name.Length > ext.Length
                    ? name.Substring(0, name.Length - ext.Length)
                    : name;

                // Filter by extension if configured (case-insensitive via normalized set)
                if (_normalizedExtensions is not null &&
                    !_normalizedExtensions.Contains(ext))
                {
                    continue;
                }

                // Group files by directory-qualified basename into samples
                if (currentBaseName is not null && nameWithoutExt != currentBaseName)
                {
                    // New sample - yield the previous one
                    if (currentSample.Count > 0)
                    {
                        yield return currentSample;
                        currentSample = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
                    }
                }

                currentBaseName = nameWithoutExt;
                currentSample[ext] = fileData;
            }
            else if (typeFlag == '5' || name.EndsWith("/", StringComparison.Ordinal))
            {
                // Directory - skip
            }
            else
            {
                // Skip unknown entry types
                long paddedSize = ((size + 511) / 512) * 512;
                byte[] skip = new byte[Math.Min(4096, paddedSize)];
                long remaining = paddedSize;
                while (remaining > 0)
                {
                    int toRead = (int)Math.Min(skip.Length, remaining);
                    int read = stream.Read(skip, 0, toRead);
                    if (read == 0)
                    {
                        throw new InvalidDataException(
                            $"TAR entry '{name}' is truncated while skipping an unknown entry type.");
                    }
                    remaining -= read;
                }
            }
        }

        // Yield last sample
        if (currentSample.Count > 0)
        {
            yield return currentSample;
        }
    }

    private static int ReadFull(Stream stream, byte[] buffer, int count)
    {
        int totalRead = 0;
        while (totalRead < count)
        {
            int read = stream.Read(buffer, totalRead, count - totalRead);
            if (read == 0) break;
            totalRead += read;
        }
        return totalRead;
    }

    private static string ReadTarString(byte[] buffer, int offset, int length)
    {
        int end = offset;
        int limit = offset + length;
        while (end < limit && buffer[end] != 0) end++;
        return System.Text.Encoding.ASCII.GetString(buffer, offset, end - offset).Trim();
    }

    private static long ReadTarOctal(byte[] buffer, int offset, int length)
    {
        string s = ReadTarString(buffer, offset, length);
        if (s.Length == 0) return 0;
        return Convert.ToInt64(s, 8);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
