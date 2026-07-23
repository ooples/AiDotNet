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
/// <para>
/// Schedules fall into two categories:
/// - <b>Single-stage</b>: Each rank owns one contiguous model chunk (GPipe, 1F1B, ZB-H1, ZB-H2).
/// - <b>Multi-stage</b>: Each rank owns V non-contiguous chunks ("virtual stages")
///   (Interleaved 1F1B, Looped BFS, ZB-V).
/// </para>
/// <para><b>For Beginners:</b> In pipeline parallelism, multiple stages process data like an
/// assembly line. A "schedule" decides the order of operations to keep all stages as busy
/// as possible and minimize idle time ("pipeline bubbles").
///
/// Think of it like coordinating workers on an assembly line:
/// - GPipe: Worker 1 finishes ALL items, then Worker 2 starts ALL items (simple but slow)
/// - 1F1B: Workers alternate between forward and backward steps (more complex but faster)
/// - Zero Bubble: Workers split backward into two parts, using the flexible part to fill gaps
/// </para>
/// </remarks>
public interface IPipelineSchedule<T>
{
    /// <summary>
    /// Gets the name of the scheduling strategy for diagnostics.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the number of virtual stages (model chunks) each rank holds.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Most schedules assign one chunk of the model to each rank
    /// (VirtualStagesPerRank = 1). Advanced schedules like Interleaved 1F1B and ZB-V assign
    /// multiple non-contiguous chunks to each rank to reduce pipeline bubbles.</para>
    /// </remarks>
    int VirtualStagesPerRank { get; }

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
    /// 1F1B reduces this significantly. Zero Bubble schedules approach 0%.</para>
    /// </remarks>
    /// <param name="numStages">Total number of pipeline stages.</param>
    /// <param name="numMicroBatches">Number of micro-batches per mini-batch.</param>
    /// <returns>Estimated fraction of total time spent in pipeline bubbles (0.0 to 1.0).</returns>
    T EstimateBubbleFraction(int numStages, int numMicroBatches);
}

/// <summary>
/// Represents a single operation in the pipeline schedule.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is one instruction in the schedule, like
/// "do forward pass on micro-batch #3" or "do backward pass on micro-batch #1".</para>
/// <para>
/// Zero Bubble schedules split the backward pass into two operations:
/// BackwardInput (compute activation gradients, on the critical path) and
/// BackwardWeight (compute weight gradients, can fill bubbles). Traditional
/// schedules use the combined Backward type.
/// </para>
/// </remarks>
public class PipelineOperation
{
    /// <summary>
    /// Gets the type of pipeline operation (Forward, Backward, BackwardInput, or BackwardWeight).
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

    /// <summary>
    /// Gets the virtual stage index for multi-stage schedules (0-based within this rank).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In multi-stage schedules like Interleaved 1F1B, each rank
    /// holds multiple model chunks. This index tells which chunk to run this operation on.
    /// For single-stage schedules, this is always 0.</para>
    /// </remarks>
    public int VirtualStageIndex { get; init; }
}

/// <summary>
/// Types of pipeline operations.
/// </summary>
/// <remarks>
/// <para>
/// Traditional schedules (GPipe, 1F1B) use Forward and Backward.
/// Zero Bubble schedules decompose Backward into BackwardInput + BackwardWeight
/// to enable filling pipeline bubbles with weight gradient computation.
/// </para>
/// <para><b>Reference:</b> Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024.
/// https://arxiv.org/abs/2401.10241</para>
/// </remarks>
public enum PipelineOperationType
{
    /// <summary>
    /// Forward pass through the stage's layers.
    /// </summary>
    Forward,

    /// <summary>
    /// Combined backward pass (gradient computation) through the stage's layers.
    /// Used by traditional schedules (GPipe, 1F1B) that don't split the backward pass.
    /// </summary>
    Backward,

    /// <summary>
    /// Backward pass computing only activation gradients (dL/dInput).
    /// This is on the critical path - the upstream stage needs these gradients.
    /// Used by Zero Bubble schedules (ZB-H1, ZB-H2, ZB-V).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes how much the loss changes when the input
    /// to this stage changes. The previous stage needs this information to continue its
    /// own backward pass, so it must be done promptly.</para>
    /// </remarks>
    BackwardInput,

    /// <summary>
    /// Backward pass computing only weight gradients (dL/dWeights).
    /// This is NOT on the critical path - no other stage depends on it.
    /// Can be deferred to fill pipeline bubbles.
    /// Used by Zero Bubble schedules (ZB-H1, ZB-H2, ZB-V).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This computes how much the loss changes when the weights
    /// of this stage change. Since no other stage needs this information, it can be computed
    /// later to fill idle time (bubbles) in the pipeline.</para>
    /// </remarks>
    BackwardWeight
}
