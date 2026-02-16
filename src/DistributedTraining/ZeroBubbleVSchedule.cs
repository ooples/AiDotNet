using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the Zero Bubble V (ZB-V) pipeline schedule with 2 virtual stages per rank.
/// </summary>
/// <remarks>
/// <para>
/// ZB-V combines the backward decomposition of ZB-H1/H2 with the virtual stage concept of
/// Interleaved 1F1B, using exactly V=2 virtual stages per rank. Each rank processes two
/// non-contiguous model chunks, creating a V-shaped execution pattern that achieves zero
/// pipeline bubble with the same peak memory as standard 1F1B.
/// </para>
/// <para>
/// The V-shape comes from the execution pattern on each rank:
/// - First half: Forward passes fill from top to bottom (forward through virtual stage 0)
/// - Middle: V-shaped transition from forward to backward
/// - Second half: Backward passes drain from bottom to top (backward through virtual stage 1)
/// </para>
/// <para><b>For Beginners:</b> ZB-V is the best of both worlds:
/// - Like Interleaved 1F1B: uses 2 model chunks per GPU to reduce bubble
/// - Like ZB-H1: splits backward into B (activation gradients) and W (weight gradients)
/// - Unlike ZB-H2: does NOT use extra memory (same as 1F1B)
///
/// The result is zero pipeline bubble with no extra memory cost. The tradeoff is slightly
/// more communication (each microbatch crosses each GPU twice) and implementation complexity.
///
/// Example with 4 GPUs (8 total virtual stages):
/// - GPU 0: virtual stages 0 and 4
/// - GPU 1: virtual stages 1 and 5
/// - GPU 2: virtual stages 2 and 6
/// - GPU 3: virtual stages 3 and 7
///
/// Each microbatch flows: 0->1->2->3->4->5->6->7 (visiting each GPU twice).
/// </para>
/// <para><b>Reference:</b> Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
/// https://arxiv.org/abs/2401.10241</para>
/// </remarks>
public class ZeroBubbleVSchedule : IPipelineSchedule
{
    /// <inheritdoc/>
    public string Name => "ZB-V";

    /// <inheritdoc/>
    public int VirtualStagesPerRank => 2;

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

        int warmupForwardsPerChunk = Math.Min(numStages - 1 - stageId, numMicroBatches);

        var state = new ScheduleState();
        EmitWarmup(ops, warmupForwardsPerChunk, numMicroBatches, state);
        EmitSteadyState(ops, numMicroBatches, state);
        EmitCooldown(ops, numMicroBatches, state);

        return ops;
    }

    /// <summary>
    /// Mutable counters tracking operation indices across schedule phases.
    /// </summary>
    private sealed class ScheduleState
    {
        public int ForwardCount0;
        public int ForwardCount1;
        public int BackwardInputCount0;
        public int BackwardInputCount1;
        public int BackwardWeightCount0;
        public int BackwardWeightCount1;
    }

    /// <summary>
    /// Phase 1: Warmup — interleaved forwards across both virtual stages (depth-first).
    /// </summary>
    private static void EmitWarmup(
        List<PipelineOperation> ops, int warmupForwardsPerChunk, int numMicroBatches, ScheduleState s)
    {
        for (int i = 0; i < warmupForwardsPerChunk && s.ForwardCount0 < numMicroBatches; i++)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = s.ForwardCount0,
                VirtualStageIndex = 0,
                IsWarmup = true,
                IsCooldown = false
            });
            s.ForwardCount0++;

            if (s.ForwardCount1 < s.ForwardCount0 && s.ForwardCount1 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = s.ForwardCount1,
                    VirtualStageIndex = 1,
                    IsWarmup = true,
                    IsCooldown = false
                });
                s.ForwardCount1++;
            }
        }
    }

    /// <summary>
    /// Phase 2: Steady state — F0, F1, B1, B0, W interleaving until all F/B complete.
    /// </summary>
    private static void EmitSteadyState(List<PipelineOperation> ops, int numMicroBatches, ScheduleState s)
    {
        while (s.ForwardCount0 < numMicroBatches ||
               s.ForwardCount1 < numMicroBatches ||
               s.BackwardInputCount0 < numMicroBatches ||
               s.BackwardInputCount1 < numMicroBatches)
        {
            bool isCooldown = s.ForwardCount0 >= numMicroBatches && s.ForwardCount1 >= numMicroBatches;

            EmitSteadyStateForwards(ops, numMicroBatches, s);
            EmitSteadyStateBackwardInputs(ops, numMicroBatches, isCooldown, s);
            EmitSteadyStateBackwardWeights(ops, numMicroBatches, isCooldown, s);
        }
    }

    private static void EmitSteadyStateForwards(List<PipelineOperation> ops, int numMicroBatches, ScheduleState s)
    {
        if (s.ForwardCount0 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = s.ForwardCount0,
                VirtualStageIndex = 0,
                IsWarmup = false,
                IsCooldown = false
            });
            s.ForwardCount0++;
        }

        if (s.ForwardCount1 < s.ForwardCount0 && s.ForwardCount1 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = s.ForwardCount1,
                VirtualStageIndex = 1,
                IsWarmup = false,
                IsCooldown = false
            });
            s.ForwardCount1++;
        }
    }

    private static void EmitSteadyStateBackwardInputs(
        List<PipelineOperation> ops, int numMicroBatches, bool isCooldown, ScheduleState s)
    {
        if (s.BackwardInputCount1 < s.ForwardCount1 && s.BackwardInputCount1 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardInput,
                MicroBatchIndex = s.BackwardInputCount1,
                VirtualStageIndex = 1,
                IsWarmup = false,
                IsCooldown = isCooldown
            });
            s.BackwardInputCount1++;
        }

        if (s.BackwardInputCount0 < s.BackwardInputCount1 && s.BackwardInputCount0 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardInput,
                MicroBatchIndex = s.BackwardInputCount0,
                VirtualStageIndex = 0,
                IsWarmup = false,
                IsCooldown = isCooldown
            });
            s.BackwardInputCount0++;
        }
    }

    private static void EmitSteadyStateBackwardWeights(
        List<PipelineOperation> ops, int numMicroBatches, bool isCooldown, ScheduleState s)
    {
        if (s.BackwardWeightCount1 < s.BackwardInputCount1 && s.BackwardWeightCount1 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = s.BackwardWeightCount1,
                VirtualStageIndex = 1,
                IsWarmup = false,
                IsCooldown = isCooldown
            });
            s.BackwardWeightCount1++;
        }

        if (s.BackwardWeightCount0 < s.BackwardInputCount0 && s.BackwardWeightCount0 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = s.BackwardWeightCount0,
                VirtualStageIndex = 0,
                IsWarmup = false,
                IsCooldown = isCooldown
            });
            s.BackwardWeightCount0++;
        }
    }

    /// <summary>
    /// Phase 3: Drain remaining BackwardWeight operations.
    /// </summary>
    private static void EmitCooldown(List<PipelineOperation> ops, int numMicroBatches, ScheduleState s)
    {
        while (s.BackwardWeightCount1 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = s.BackwardWeightCount1,
                VirtualStageIndex = 1,
                IsWarmup = false,
                IsCooldown = true
            });
            s.BackwardWeightCount1++;
        }

        while (s.BackwardWeightCount0 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = s.BackwardWeightCount0,
                VirtualStageIndex = 0,
                IsWarmup = false,
                IsCooldown = true
            });
            s.BackwardWeightCount0++;
        }
    }

    /// <inheritdoc/>
    public double EstimateBubbleFraction(int numStages, int numMicroBatches)
    {
        if (numStages <= 1 || numMicroBatches <= 0)
        {
            return 0.0;
        }

        // ZB-V achieves zero bubble when numMicroBatches >= numStages
        // Same as ZB-H2 but with 1F1B-equivalent memory
        if (numMicroBatches >= numStages)
        {
            return 0.0;
        }

        // For insufficient micro-batches, small residual bubble
        // With V=2 virtual stages, the bubble is reduced compared to ZB-H1
        double p = numStages;
        double m = numMicroBatches;
        return (p - m) / (6.0 * m + p);
    }
}
