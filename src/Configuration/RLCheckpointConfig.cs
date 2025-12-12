namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for checkpointing during RL training.
/// </summary>
public class RLCheckpointConfig
{
    /// <summary>
    /// Directory to save checkpoints.
    /// </summary>
    public string CheckpointDirectory { get; set; } = "./checkpoints";

    /// <summary>
    /// Save checkpoint every N episodes.
    /// </summary>
    public int SaveEveryEpisodes { get; set; } = 100;

    /// <summary>
    /// Keep only the best N checkpoints.
    /// </summary>
    public int KeepBestN { get; set; } = 3;

    /// <summary>
    /// Save checkpoint on best reward improvement.
    /// </summary>
    public bool SaveOnBestReward { get; set; } = true;
}
