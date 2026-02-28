using System.Text;

namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Saves and restores training state mid-epoch for fault tolerance.
/// </summary>
/// <remarks>
/// <para>
/// Enables resuming training from the exact batch within an epoch after a failure.
/// Saves the current batch index, epoch number, and any custom state as a binary checkpoint.
/// Automatically rotates old checkpoints to limit disk usage.
/// </para>
/// </remarks>
public class MidEpochCheckpointer
{
    private readonly MidEpochCheckpointerOptions _options;
    private int _batchesSinceLastSave;

    /// <summary>
    /// Creates a new mid-epoch checkpointer.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public MidEpochCheckpointer(MidEpochCheckpointerOptions options)
    {
        _options = options;
        _batchesSinceLastSave = 0;

        if (!string.IsNullOrEmpty(_options.CheckpointDirectory))
            Directory.CreateDirectory(_options.CheckpointDirectory);
    }

    /// <summary>
    /// Called after each batch to potentially save a checkpoint.
    /// </summary>
    /// <param name="epoch">Current epoch number.</param>
    /// <param name="batchIndex">Current batch index within the epoch.</param>
    /// <param name="customState">Optional custom state to include in the checkpoint.</param>
    /// <returns>True if a checkpoint was saved.</returns>
    public bool OnBatchComplete(int epoch, int batchIndex, byte[]? customState = null)
    {
        _batchesSinceLastSave++;

        if (_batchesSinceLastSave >= _options.SaveEveryNBatches)
        {
            SaveCheckpoint(epoch, batchIndex, customState);
            _batchesSinceLastSave = 0;
            CleanOldCheckpoints();
            return true;
        }

        return false;
    }

    /// <summary>
    /// Saves a checkpoint with the current training state.
    /// </summary>
    /// <param name="epoch">Current epoch.</param>
    /// <param name="batchIndex">Current batch index.</param>
    /// <param name="customState">Optional custom binary state.</param>
    public void SaveCheckpoint(int epoch, int batchIndex, byte[]? customState = null)
    {
        string fileName = $"{_options.FilePrefix}_e{epoch}_b{batchIndex}.ckpt";
        string filePath = Path.Combine(_options.CheckpointDirectory, fileName);

        using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);

        // Write checkpoint header
        writer.Write(Encoding.ASCII.GetBytes("CKPT"));
        writer.Write(1); // version
        writer.Write(epoch);
        writer.Write(batchIndex);
        writer.Write(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());

        // Write custom state
        if (customState != null)
        {
            writer.Write(customState.Length);
            writer.Write(customState);
        }
        else
        {
            writer.Write(0);
        }
    }

    /// <summary>
    /// Loads the latest checkpoint from the checkpoint directory.
    /// </summary>
    /// <returns>Checkpoint data, or null if no checkpoint exists.</returns>
    public CheckpointData? LoadLatestCheckpoint()
    {
        if (!Directory.Exists(_options.CheckpointDirectory))
            return null;

        var checkpointFiles = Directory.GetFiles(_options.CheckpointDirectory, $"{_options.FilePrefix}_*.ckpt")
            .OrderByDescending(f => new FileInfo(f).LastWriteTimeUtc)
            .ToArray();

        if (checkpointFiles.Length == 0)
            return null;

        return LoadCheckpoint(checkpointFiles[0]);
    }

    /// <summary>
    /// Loads a specific checkpoint file.
    /// </summary>
    /// <param name="filePath">Path to the checkpoint file.</param>
    /// <returns>The checkpoint data.</returns>
    public static CheckpointData LoadCheckpoint(string filePath)
    {
        using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
        using var reader = new BinaryReader(stream);

        byte[] magic = reader.ReadBytes(4);
        if (Encoding.ASCII.GetString(magic) != "CKPT")
            throw new InvalidDataException("Not a valid checkpoint file.");

        int version = reader.ReadInt32();
        if (version != 1)
            throw new InvalidDataException($"Unsupported checkpoint version: {version}");

        int epoch = reader.ReadInt32();
        int batchIndex = reader.ReadInt32();
        long timestamp = reader.ReadInt64();

        int stateLength = reader.ReadInt32();
        byte[]? customState = stateLength > 0 ? reader.ReadBytes(stateLength) : null;

        return new CheckpointData
        {
            Epoch = epoch,
            BatchIndex = batchIndex,
            Timestamp = DateTimeOffset.FromUnixTimeMilliseconds(timestamp),
            CustomState = customState,
            FilePath = filePath
        };
    }

    private void CleanOldCheckpoints()
    {
        if (!Directory.Exists(_options.CheckpointDirectory))
            return;

        var checkpointFiles = Directory.GetFiles(_options.CheckpointDirectory, $"{_options.FilePrefix}_*.ckpt")
            .OrderByDescending(f => new FileInfo(f).LastWriteTimeUtc)
            .ToArray();

        for (int i = _options.MaxCheckpoints; i < checkpointFiles.Length; i++)
        {
            try { File.Delete(checkpointFiles[i]); }
            catch { /* Best effort cleanup */ }
        }
    }
}

/// <summary>
/// Data stored in a mid-epoch checkpoint.
/// </summary>
public sealed class CheckpointData
{
    /// <summary>The epoch number at checkpoint time.</summary>
    public int Epoch { get; init; }
    /// <summary>The batch index within the epoch.</summary>
    public int BatchIndex { get; init; }
    /// <summary>When the checkpoint was created.</summary>
    public DateTimeOffset Timestamp { get; init; }
    /// <summary>Optional custom state bytes.</summary>
    public byte[]? CustomState { get; init; }
    /// <summary>Path to the checkpoint file.</summary>
    public string FilePath { get; init; } = "";
}
