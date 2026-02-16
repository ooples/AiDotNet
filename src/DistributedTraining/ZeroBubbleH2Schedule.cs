using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the Zero Bubble H2 (ZB-H2) pipeline schedule.
/// </summary>
/// <remarks>
/// <para>
/// ZB-H2 achieves true zero pipeline bubble by allowing more in-flight micro-batches
/// than 1F1B, trading peak memory for throughput. Like ZB-H1, it splits backward into
/// BackwardInput (B) and BackwardWeight (W), but schedules more aggressively.
/// </para>
/// <para><b>For Beginners:</b> ZB-H2 is the "maximum throughput" variant. It allows more
/// micro-batches to be in progress simultaneously (using more memory) to completely
/// eliminate idle time. If you have enough GPU memory, ZB-H2 gives the best possible
/// pipeline utilization.
///
/// The tradeoff:
/// - ZB-H1: Same memory as 1F1B, ~1/3 bubble
/// - ZB-H2: More memory than 1F1B, ~0% bubble (zero idle time)
/// </para>
/// <para><b>Reference:</b> Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
/// https://arxiv.org/abs/2401.10241</para>
/// </remarks>
public class ZeroBubbleH2Schedule : IPipelineSchedule
{
    /// <inheritdoc/>
    public string Name => "ZB-H2";

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

        int numWarmupForwards = Math.Min(numStages - stageId, numMicroBatches);

        int forwardIdx = EmitExtendedWarmup(ops, numWarmupForwards);
        var (backwardInputIdx, backwardWeightIdx) = EmitSteadyState(
            ops, numMicroBatches, numWarmupForwards, forwardIdx);
        EmitCooldown(ops, numMicroBatches, backwardInputIdx, backwardWeightIdx);

        return ops;
    }

    /// <summary>
    /// Phase 1: Extended warmup — more forward passes to fill pipeline completely.
    /// </summary>
    private static int EmitExtendedWarmup(List<PipelineOperation> ops, int numWarmupForwards)
    {
        for (int i = 0; i < numWarmupForwards; i++)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = i,
                IsWarmup = true,
                IsCooldown = false
            });
        }

        return numWarmupForwards;
    }

    /// <summary>
    /// Phase 2: Steady state — interleave B, F, W to maintain zero bubble.
    /// </summary>
    private static (int BackwardInputIdx, int BackwardWeightIdx) EmitSteadyState(
        List<PipelineOperation> ops, int numMicroBatches, int numWarmupForwards, int forwardIdx)
    {
        int backwardInputIdx = 0;
        int backwardWeightIdx = 0;
        int steadyStateCount = Math.Max(0, numMicroBatches - numWarmupForwards);

        for (int i = 0; i < steadyStateCount; i++)
        {
            // BackwardInput (B) first - critical path
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardInput,
                MicroBatchIndex = backwardInputIdx,
                IsWarmup = false,
                IsCooldown = false
            });
            backwardInputIdx++;

            // Forward for next micro-batch
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

            // BackwardWeight (W) - fills any remaining time
            if (backwardWeightIdx < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardWeight,
                    MicroBatchIndex = backwardWeightIdx,
                    IsWarmup = false,
                    IsCooldown = false
                });
                backwardWeightIdx++;
            }
        }

        return (backwardInputIdx, backwardWeightIdx);
    }

    /// <summary>
    /// Phase 3: Cooldown — drain remaining B and W passes.
    /// </summary>
    private static void EmitCooldown(
        List<PipelineOperation> ops, int numMicroBatches, int backwardInputIdx, int backwardWeightIdx)
    {
        while (backwardInputIdx < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardInput,
                MicroBatchIndex = backwardInputIdx,
                IsWarmup = false,
                IsCooldown = true
            });
            backwardInputIdx++;

            // Interleave W during cooldown
            if (backwardWeightIdx < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardWeight,
                    MicroBatchIndex = backwardWeightIdx,
                    IsWarmup = false,
                    IsCooldown = true
                });
                backwardWeightIdx++;
            }
        }

        // Final W drain
        while (backwardWeightIdx < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = backwardWeightIdx,
                IsWarmup = false,
                IsCooldown = true
            });
            backwardWeightIdx++;
        }
    }

    /// <inheritdoc/>
    public double EstimateBubbleFraction(int numStages, int numMicroBatches)
    {
        if (numStages <= 1 || numMicroBatches <= 0)
        {
            return 0.0;
        }

        // ZB-H2 achieves near-zero bubble when numMicroBatches >= numStages
        // For insufficient micro-batches, there's still some residual bubble
        if (numMicroBatches >= numStages)
        {
            return 0.0;
        }

        // Fallback estimate for small M
        double p = numStages;
        double m = numMicroBatches;
        return (p - m) / (3.0 * m + p);
    }
}
