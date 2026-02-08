using System.IO;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Formats;

/// <summary>
/// Shard-based streaming dataset for deterministic, resumable, large-scale data loading.
/// </summary>
/// <remarks>
/// <para>
/// Inspired by Mosaic ML's Streaming library, this dataset reads from pre-sharded binary files
/// where each shard contains a sequence of length-prefixed byte records. This enables:
/// - Deterministic iteration order across restarts
/// - Efficient sequential I/O
/// - Shard-level parallelism
/// - Resumable training (restart from any shard + offset)
/// </para>
/// <para>
/// Each shard file has the format: [4-byte little-endian length][record bytes][4-byte LE length][record bytes]...
/// </para>
/// </remarks>
internal class ShardedStreamingDataset : IDisposable
{
    private readonly string[] _shardPaths;
    private readonly ShardedStreamingDatasetOptions _options;
    private bool _disposed;

    /// <summary>
    /// Gets the total number of shards.
    /// </summary>
    public int ShardCount => _shardPaths.Length;

    /// <summary>
    /// Creates a new sharded streaming dataset.
    /// </summary>
    /// <param name="shardPaths">Paths to the shard files.</param>
    /// <param name="options">Optional configuration.</param>
    public ShardedStreamingDataset(string[] shardPaths, ShardedStreamingDatasetOptions? options = null)
    {
        if (shardPaths is null || shardPaths.Length == 0)
            throw new ArgumentException("At least one shard path is required.", nameof(shardPaths));

        _shardPaths = shardPaths;
        _options = options ?? new ShardedStreamingDatasetOptions();
    }

    /// <summary>
    /// Iterates through all records across all shards.
    /// </summary>
    /// <param name="epoch">Current epoch number (used for deterministic shard shuffling).</param>
    /// <returns>An enumerable of raw byte records.</returns>
    public IEnumerable<byte[]> ReadRecords(int epoch = 0)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(ShardedStreamingDataset));

        string[] shardOrder;
        if (_options.ShuffleShards)
        {
            int seed = _options.Seed.HasValue ? _options.Seed.Value + epoch : epoch;
            var random = RandomHelper.CreateSeededRandom(seed);
            shardOrder = (string[])_shardPaths.Clone();
            for (int i = shardOrder.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (shardOrder[i], shardOrder[j]) = (shardOrder[j], shardOrder[i]);
            }
        }
        else
        {
            shardOrder = _shardPaths;
        }

        int samplesRead = 0;
        var shuffleBuffer = new List<byte[]>();
        var shuffleRandom = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value + epoch + 1000)
            : RandomHelper.CreateSecureRandom();

        foreach (string shardPath in shardOrder)
        {
            foreach (byte[] record in ReadShard(shardPath))
            {
                if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                    yield break;

                if (_options.ShuffleBufferSize > 1)
                {
                    shuffleBuffer.Add(record);
                    if (shuffleBuffer.Count >= _options.ShuffleBufferSize)
                    {
                        for (int i = shuffleBuffer.Count - 1; i > 0; i--)
                        {
                            int j = shuffleRandom.Next(i + 1);
                            var temp = shuffleBuffer[i];
                            shuffleBuffer[i] = shuffleBuffer[j];
                            shuffleBuffer[j] = temp;
                        }
                        foreach (var item in shuffleBuffer)
                        {
                            yield return item;
                            samplesRead++;
                            if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                                yield break;
                        }
                        shuffleBuffer.Clear();
                    }
                }
                else
                {
                    yield return record;
                    samplesRead++;
                }
            }
        }

        // Flush remaining buffer
        if (shuffleBuffer.Count > 0)
        {
            for (int i = shuffleBuffer.Count - 1; i > 0; i--)
            {
                int j = shuffleRandom.Next(i + 1);
                var temp = shuffleBuffer[i];
                shuffleBuffer[i] = shuffleBuffer[j];
                shuffleBuffer[j] = temp;
            }
            foreach (var item in shuffleBuffer)
            {
                if (_options.MaxSamples.HasValue && samplesRead >= _options.MaxSamples.Value)
                    yield break;
                yield return item;
                samplesRead++;
            }
        }
    }

    /// <summary>
    /// Reads all records from a single shard file.
    /// </summary>
    /// <param name="shardPath">Path to the shard file.</param>
    /// <returns>An enumerable of byte records.</returns>
    /// <summary>
    /// Maximum allowed record length (1 GB) to guard against corrupted files causing OOM.
    /// </summary>
    private const int MaxRecordLength = 1 << 30;

    public static IEnumerable<byte[]> ReadShard(string shardPath)
    {
        using var stream = new FileStream(shardPath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 65536);
        byte[] lengthBuf = new byte[4];

        while (true)
        {
            int read = ReadFull(stream, lengthBuf, 4);
            if (read < 4) yield break;

            int recordLength = BitConverter.ToInt32(lengthBuf, 0);
            if (recordLength <= 0 || recordLength > MaxRecordLength)
                throw new InvalidDataException(
                    $"Invalid record length {recordLength} in shard '{shardPath}'. " +
                    $"Must be between 1 and {MaxRecordLength} bytes. The shard file may be corrupted.");

            byte[] record = new byte[recordLength];
            read = ReadFull(stream, record, recordLength);
            if (read < recordLength)
                throw new InvalidDataException(
                    $"Truncated record in shard '{shardPath}': expected {recordLength} bytes but only read {read}. The shard file may be corrupted.");

            yield return record;
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

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}
