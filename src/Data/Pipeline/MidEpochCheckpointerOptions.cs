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
}
