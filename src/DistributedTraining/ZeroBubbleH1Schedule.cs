using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the Zero Bubble H1 (ZB-H1) pipeline schedule.
/// </summary>
/// <remarks>
/// <para>
/// ZB-H1 splits the backward pass into two independent computations:
/// - <b>B (BackwardInput)</b>: Computes activation gradients (dL/dInput) - on the critical path.
/// - <b>W (BackwardWeight)</b>: Computes weight gradients (dL/dWeights) - can be deferred.
///
/// By deferring W to fill pipeline bubbles, ZB-H1 reduces the bubble to approximately
/// one-third of 1F1B's bubble while maintaining the same peak memory footprint.
/// </para>
/// <para><b>For Beginners:</b> In standard 1F1B, the backward pass computes both activation and
/// weight gradients together. ZB-H1 splits this into two steps. The activation gradient (B)
/// must be done quickly (the previous stage is waiting), but the weight gradient (W) can wait.
/// By scheduling W during idle time, we reduce wasted time by ~67% compared to 1F1B.
///
/// Think of it like a car wash: the "rinse" (B) must happen right after soap, but "waxing" (W)
/// can be done whenever there's a free slot.
/// </para>
/// <para><b>Reference:</b> Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
/// https://arxiv.org/abs/2401.10241</para>
/// </remarks>
public class ZeroBubbleH1Schedule : IPipelineSchedule
{
    /// <inheritdoc/>
    public string Name => "ZB-H1";

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

        int numWarmupForwards = Math.Min(numStages - 1 - stageId, numMicroBatches);
        int numSteadyState = Math.Max(0, numMicroBatches - numWarmupForwards);

        int forwardIdx = EmitWarmupForwards(ops, numWarmupForwards);
        var (backwardInputIdx, backwardWeightIdx) = EmitSteadyState(
            ops, numSteadyState, numMicroBatches, forwardIdx);
        EmitCooldown(ops, numMicroBatches, backwardInputIdx, backwardWeightIdx);

        return ops;
    }

    /// <summary>
    /// Phase 1: Warmup — forward passes only (same as 1F1B).
    /// </summary>
    private static int EmitWarmupForwards(List<PipelineOperation> ops, int numWarmupForwards)
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
    /// Phase 2: Steady state — 1F-1B-1W pattern.
    /// For each step: one Forward, one BackwardInput, and one BackwardWeight (bubble fill).
    /// </summary>
    private static (int BackwardInputIdx, int BackwardWeightIdx) EmitSteadyState(
        List<PipelineOperation> ops, int numSteadyState, int numMicroBatches, int forwardIdx)
    {
        int backwardInputIdx = 0;
        int backwardWeightIdx = 0;

        for (int i = 0; i < numSteadyState; i++)
        {
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

            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardInput,
                MicroBatchIndex = backwardInputIdx,
                IsWarmup = false,
                IsCooldown = false
            });
            backwardInputIdx++;

            if (backwardWeightIdx < backwardInputIdx && backwardWeightIdx < numMicroBatches)
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
    /// Phase 3: Cooldown — drain remaining BackwardInput and BackwardWeight passes.
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
        }

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

        // ZB-H1 bubble is approximately 1/3 of 1F1B's bubble
        // 1F1B bubble: (P-1) / (2*M + P - 1)
        // ZB-H1 bubble: ~(P-1) / (3*M + P - 1)
        double p = numStages;
        double m = numMicroBatches;
        return (p - 1.0) / (3.0 * m + p - 1.0);
    }
}
