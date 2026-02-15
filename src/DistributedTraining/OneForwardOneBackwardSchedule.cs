using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the 1F1B (One-Forward-One-Backward) pipeline schedule.
/// </summary>
/// <remarks>
/// <para>
/// The 1F1B schedule interleaves forward and backward passes to minimize pipeline bubble
/// and memory usage. It has three phases:
///
/// 1. <b>Warmup</b>: Each stage executes forward passes to fill the pipeline.
///    Stage i performs (numStages - 1 - i) forward passes before steady state.
///
/// 2. <b>Steady State</b>: Each stage alternates between one forward and one backward pass.
///    This keeps all stages busy and limits memory usage to at most (numStages) activations.
///
/// 3. <b>Cooldown</b>: Remaining backward passes drain the pipeline.
/// </para>
/// <para><b>For Beginners:</b> Instead of doing ALL forward passes then ALL backward passes (GPipe),
/// 1F1B interleaves them. This is like a factory where each worker handles their current item
/// and immediately starts the return processing, rather than waiting for all items to pass through.
///
/// Benefits:
/// - Reduces pipeline bubble from ~50% to ~12-15%
/// - Limits peak memory to (numStages) stored activations instead of (numMicroBatches)
/// - More efficient for large numbers of micro-batches
///
/// Example with 4 stages and 8 micro-batches:
/// <code>
/// Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
/// Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
/// Stage 2:       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
/// Stage 3:          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7
/// </code>
/// </para>
/// <para><b>Reference:</b> Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training", SOSP 2019.
/// https://arxiv.org/abs/1806.03377</para>
/// </remarks>
public class OneForwardOneBackwardSchedule : IPipelineSchedule
{
    /// <inheritdoc/>
    public string Name => "1F1B";

    /// <inheritdoc/>
    public int VirtualStagesPerRank => 1;

    /// <inheritdoc/>
    public IReadOnlyList<PipelineOperation> GetSchedule(int stageId, int numStages, int numMicroBatches)
    {
        if (numStages <= 0)
        {
            throw new ArgumentException("Number of stages must be positive.", nameof(numStages));
        }

        if (numMicroBatches <= 0)
        {
            throw new ArgumentException("Number of micro-batches must be positive.", nameof(numMicroBatches));
        }

        if (stageId < 0 || stageId >= numStages)
        {
            throw new ArgumentOutOfRangeException(nameof(stageId),
                $"Stage ID must be between 0 and {numStages - 1}.");
        }

        var ops = new List<PipelineOperation>();

        // Number of warmup forward passes for this stage
        // Earlier stages need more warmup to fill the pipeline
        int numWarmupForwards = Math.Min(numStages - 1 - stageId, numMicroBatches);

        // Number of steady-state 1F1B pairs
        int numSteadyState = Math.Max(0, numMicroBatches - numWarmupForwards);

        // Phase 1: Warmup - only forward passes
        int forwardIdx = 0;
        for (int i = 0; i < numWarmupForwards; i++)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = forwardIdx,
                IsWarmup = true,
                IsCooldown = false
            });
            forwardIdx++;
        }

        // Phase 2: Steady state - alternating 1F1B
        int backwardIdx = 0;
        for (int i = 0; i < numSteadyState; i++)
        {
            // One forward
            if (forwardIdx < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = forwardIdx,
                    IsWarmup = false,
                    IsCooldown = false
                });
                forwardIdx++;
            }

            // One backward
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Backward,
                MicroBatchIndex = backwardIdx,
                IsWarmup = false,
                IsCooldown = false
            });
            backwardIdx++;
        }

        // Phase 3: Cooldown - only backward passes
        while (backwardIdx < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Backward,
                MicroBatchIndex = backwardIdx,
                IsWarmup = false,
                IsCooldown = true
            });
            backwardIdx++;
        }

        return ops;
    }

    /// <inheritdoc/>
    public double EstimateBubbleFraction(int numStages, int numMicroBatches)
    {
        if (numStages <= 1 || numMicroBatches <= 0)
        {
            return 0.0;
        }

        // 1F1B bubble fraction: (P-1) / (2*M + P - 1) where P = stages, M = micro-batches
        // This is approximately half of GPipe's bubble for large M
        int p = numStages;
        int m = numMicroBatches;
        return (double)(p - 1) / (2L * m + p - 1);
    }
}
