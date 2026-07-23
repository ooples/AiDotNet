namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Configuration for distillation checkpoint management.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class controls when and how models are saved during
/// knowledge distillation training. Think of it like the "auto-save" settings in a video game.</para>
/// </remarks>
public class DistillationCheckpointConfig
{
    /// <summary>
    /// Directory where checkpoints will be saved.
    /// </summary>
    public string CheckpointDirectory { get; set; } = "./checkpoints";

    /// <summary>
    /// Save checkpoint every N epochs (0 = disabled).
    /// </summary>
    /// <remarks>
    /// <para>Example: SaveEveryEpochs = 5 means save after epochs 5, 10, 15, etc.</para>
    /// </remarks>
    public int SaveEveryEpochs { get; set; } = 10;

    /// <summary>
    /// Save checkpoint every N batches (0 = disabled).
    /// </summary>
    /// <remarks>
    /// <para>Useful for very long epochs. Example: SaveEveryBatches = 1000 means
    /// save after 1000 batches, 2000 batches, etc.</para>
    /// </remarks>
    public int SaveEveryBatches { get; set; } = 0;

    /// <summary>
    /// Keep only the best N checkpoints based on validation metric.
    /// </summary>
    /// <remarks>
    /// <para>Set to 0 to keep all checkpoints. Example: KeepBestN = 3 keeps only
    /// the 3 checkpoints with best validation performance.</para>
    /// </remarks>
    public int KeepBestN { get; set; } = 3;

    /// <summary>
    /// Save the teacher model checkpoint.
    /// </summary>
    /// <remarks>
    /// <para>Useful for multi-stage distillation where student becomes teacher.</para>
    /// </remarks>
    public bool SaveTeacher { get; set; } = false;

    /// <summary>
    /// Save the student model checkpoint.
    /// </summary>
    public bool SaveStudent { get; set; } = true;

    /// <summary>
    /// Metric to use for determining "best" checkpoint (e.g., "validation_loss", "accuracy").
    /// </summary>
    public string BestMetric { get; set; } = "validation_loss";

    /// <summary>
    /// Whether lower metric values are better (true for loss, false for accuracy).
    /// </summary>
    public bool LowerIsBetter { get; set; } = true;

    /// <summary>
    /// Save curriculum strategy state with checkpoint.
    /// </summary>
    /// <remarks>
    /// <para>Allows resuming with correct curriculum progress.</para>
    /// </remarks>
    public bool SaveCurriculumState { get; set; } = true;

    /// <summary>
    /// Save training metadata (epoch number, loss history, etc.).
    /// </summary>
    public bool SaveMetadata { get; set; } = true;

    /// <summary>
    /// Prefix for checkpoint filenames.
    /// </summary>
    public string CheckpointPrefix { get; set; } = "distillation";
}
