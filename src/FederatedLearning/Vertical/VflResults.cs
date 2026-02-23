using AiDotNet.FederatedLearning.PSI;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Contains summary statistics from the entity alignment phase of VFL.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Before VFL training starts, the parties must find which entities
/// they share. This summary tells you how many entities are shared, how much overlap there is,
/// and whether there's enough data for meaningful joint training.</para>
/// </remarks>
public class VflAlignmentSummary
{
    /// <summary>
    /// Gets or sets the total number of aligned entities across all parties.
    /// </summary>
    public int AlignedEntityCount { get; set; }

    /// <summary>
    /// Gets or sets the per-party entity counts before alignment.
    /// </summary>
    public IReadOnlyDictionary<string, int> PartyEntityCounts { get; set; } = new Dictionary<string, int>();

    /// <summary>
    /// Gets or sets the per-party overlap ratios (fraction of party's entities that are aligned).
    /// </summary>
    public IReadOnlyDictionary<string, double> PartyOverlapRatios { get; set; } = new Dictionary<string, double>();

    /// <summary>
    /// Gets or sets the underlying PSI result from the alignment protocol.
    /// </summary>
    public EntityAlignmentResult? AlignmentResult { get; set; }

    /// <summary>
    /// Gets or sets whether the alignment meets the minimum overlap threshold.
    /// </summary>
    public bool MeetsMinimumOverlap { get; set; }

    /// <summary>
    /// Gets or sets the time taken for the alignment phase.
    /// </summary>
    public TimeSpan AlignmentTime { get; set; }
}

/// <summary>
/// Contains metrics from a single VFL training epoch.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VflEpochResult<T>
{
    /// <summary>
    /// Gets or sets the epoch number (0-indexed).
    /// </summary>
    public int Epoch { get; set; }

    /// <summary>
    /// Gets or sets the average training loss for this epoch.
    /// </summary>
    public double AverageLoss { get; set; }

    /// <summary>
    /// Gets or sets the number of samples processed in this epoch.
    /// </summary>
    public int SamplesProcessed { get; set; }

    /// <summary>
    /// Gets or sets the number of batches processed in this epoch.
    /// </summary>
    public int BatchesProcessed { get; set; }

    /// <summary>
    /// Gets or sets the time taken for this epoch.
    /// </summary>
    public TimeSpan EpochTime { get; set; }

    /// <summary>
    /// Gets or sets the cumulative privacy budget spent (epsilon, delta) if label DP is enabled.
    /// </summary>
    public (double Epsilon, double Delta)? PrivacyBudgetSpent { get; set; }
}

/// <summary>
/// Contains the complete results from a VFL training run.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VflTrainingResult<T>
{
    /// <summary>
    /// Gets or sets the per-epoch training history.
    /// </summary>
    public IReadOnlyList<VflEpochResult<T>> EpochHistory { get; set; } = Array.Empty<VflEpochResult<T>>();

    /// <summary>
    /// Gets or sets the final training loss.
    /// </summary>
    public double FinalLoss { get; set; }

    /// <summary>
    /// Gets or sets the total training time across all epochs.
    /// </summary>
    public TimeSpan TotalTrainingTime { get; set; }

    /// <summary>
    /// Gets or sets the total number of epochs completed.
    /// </summary>
    public int EpochsCompleted { get; set; }

    /// <summary>
    /// Gets or sets the alignment summary from the entity alignment phase.
    /// </summary>
    public VflAlignmentSummary? AlignmentSummary { get; set; }

    /// <summary>
    /// Gets or sets whether training completed all requested epochs (vs. early stopping).
    /// </summary>
    public bool TrainingCompleted { get; set; }

    /// <summary>
    /// Gets or sets the number of parties that participated.
    /// </summary>
    public int NumberOfParties { get; set; }
}
