namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a scheduling strategy for pipeline parallel training.
/// </summary>
/// <remarks>
/// <para>
/// Pipeline schedules determine the order in which forward and backward passes execute
/// across micro-batches and stages. Different schedules trade off memory usage, pipeline
/// bubble overhead, and implementation complexity.
/// </para>
/// <para><b>For Beginners:</b> In pipeline parallelism, multiple stages process data like an
/// assembly line. A "schedule" decides the order of operations to keep all stages as busy
/// as possible and minimize idle time ("pipeline bubbles").
///
/// Think of it like coordinating workers on an assembly line:
/// - GPipe: Worker 1 finishes ALL items, then Worker 2 starts ALL items (simple but slow)
/// - 1F1B: Workers alternate between forward and backward steps (more complex but faster)
/// </para>
/// </remarks>
public interface IPipelineSchedule
{
    /// <summary>
    /// Gets the name of the scheduling strategy for diagnostics.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Generates the sequence of operations for a given stage in the pipeline.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This returns a list of instructions for a specific stage,
    /// telling it when to do forward passes, backward passes, and which micro-batch to work on.</para>
    /// </remarks>
    /// <param name="stageId">The pipeline stage index (0-based).</param>
    /// <param name="numStages">Total number of pipeline stages.</param>
    /// <param name="numMicroBatches">Number of micro-batches per mini-batch.</param>
    /// <returns>Ordered sequence of pipeline operations for this stage.</returns>
    IReadOnlyList<PipelineOperation> GetSchedule(int stageId, int numStages, int numMicroBatches);

    /// <summary>
    /// Estimates the pipeline bubble fraction for this schedule.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The bubble fraction is the percentage of time that stages are idle
    /// (waiting for data). Lower is better. GPipe has ~(numStages-1)/numMicroBatches bubble.
    /// 1F1B reduces this significantly.</para>
    /// </remarks>
    /// <param name="numStages">Total number of pipeline stages.</param>
    /// <param name="numMicroBatches">Number of micro-batches per mini-batch.</param>
    /// <returns>Estimated fraction of total time spent in pipeline bubbles (0.0 to 1.0).</returns>
    double EstimateBubbleFraction(int numStages, int numMicroBatches);
}

/// <summary>
/// Represents a single operation in the pipeline schedule.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is one instruction in the schedule, like
/// "do forward pass on micro-batch #3" or "do backward pass on micro-batch #1".</para>
/// </remarks>
public class PipelineOperation
{
    /// <summary>
    /// Gets the type of pipeline operation (Forward or Backward).
    /// </summary>
    public PipelineOperationType Type { get; init; }

    /// <summary>
    /// Gets the micro-batch index this operation works on.
    /// </summary>
    public int MicroBatchIndex { get; init; }

    /// <summary>
    /// Gets whether this is a warmup operation (part of pipeline fill phase).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During warmup, the pipeline is "filling up" - not all stages
    /// are busy yet. After warmup, the pipeline runs at full utilization.</para>
    /// </remarks>
    public bool IsWarmup { get; init; }

    /// <summary>
    /// Gets whether this is a cooldown operation (part of pipeline drain phase).
    /// </summary>
    public bool IsCooldown { get; init; }
}

/// <summary>
/// Types of pipeline operations.
/// </summary>
public enum PipelineOperationType
{
    /// <summary>
    /// Forward pass through the stage's layers.
    /// </summary>
    Forward,

    /// <summary>
    /// Backward pass (gradient computation) through the stage's layers.
    /// </summary>
    Backward
}
