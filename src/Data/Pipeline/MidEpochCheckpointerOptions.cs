namespace AiDotNet.Data.Pipeline;

/// <summary>
/// Configuration options for mid-epoch checkpointing.
/// </summary>
public sealed class MidEpochCheckpointerOptions
{
    /// <summary>Directory to save checkpoint files. Required.</summary>
    public string CheckpointDirectory { get; set; } = "";
    /// <summary>Save a checkpoint every N batches. Default is 100.</summary>
    public int SaveEveryNBatches { get; set; } = 100;
    /// <summary>Maximum number of checkpoints to keep (oldest deleted first). Default is 3.</summary>
    public int MaxCheckpoints { get; set; } = 3;
    /// <summary>Prefix for checkpoint file names. Default is "checkpoint".</summary>
    public string FilePrefix { get; set; } = "checkpoint";

    /// <summary>Validates that all option values are within acceptable ranges.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any option is invalid.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(CheckpointDirectory)) throw new ArgumentException("CheckpointDirectory must not be empty.", nameof(CheckpointDirectory));
        if (SaveEveryNBatches <= 0) throw new ArgumentOutOfRangeException(nameof(SaveEveryNBatches), "SaveEveryNBatches must be positive.");
        if (MaxCheckpoints <= 0) throw new ArgumentOutOfRangeException(nameof(MaxCheckpoints), "MaxCheckpoints must be positive.");
        if (string.IsNullOrWhiteSpace(FilePrefix)) throw new ArgumentException("FilePrefix must not be empty or whitespace.", nameof(FilePrefix));
    }
}
