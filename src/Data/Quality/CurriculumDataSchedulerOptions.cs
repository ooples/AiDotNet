namespace AiDotNet.Data.Quality;

/// <summary>
/// Configuration options for curriculum data scheduling.
/// </summary>
/// <remarks>
/// Curriculum learning presents training samples in a meaningful order (easy to hard),
/// which can improve convergence speed and final model quality.
/// </remarks>
public sealed class CurriculumDataSchedulerOptions
{
    /// <summary>Curriculum ordering strategy. Default is EasyToHard.</summary>
    public CurriculumOrder Order { get; set; } = CurriculumOrder.EasyToHard;
    /// <summary>Pacing function controlling how fast harder samples are introduced. Default is Linear.</summary>
    public CurriculumPacing Pacing { get; set; } = CurriculumPacing.Linear;
    /// <summary>Initial fraction of the dataset available at the start. Default is 0.2 (20%).</summary>
    public double InitialFraction { get; set; } = 0.2;
    /// <summary>Epoch at which the full dataset becomes available. Default is 10.</summary>
    public int FullDataEpoch { get; set; } = 10;
}

/// <summary>
/// Order in which curriculum samples are presented.
/// </summary>
public enum CurriculumOrder
{
    /// <summary>Start with easy samples, gradually introduce harder ones.</summary>
    EasyToHard,
    /// <summary>Start with hard samples (anti-curriculum, sometimes effective for specific tasks).</summary>
    HardToEasy,
    /// <summary>Random order baseline (no curriculum).</summary>
    Random
}

/// <summary>
/// Pacing function controlling how fast the data pool grows.
/// </summary>
public enum CurriculumPacing
{
    /// <summary>Linear growth from initial fraction to 1.0.</summary>
    Linear,
    /// <summary>Exponential growth (slow start, rapid expansion).</summary>
    Exponential,
    /// <summary>Step function: discrete jumps in data pool size.</summary>
    Step
}
