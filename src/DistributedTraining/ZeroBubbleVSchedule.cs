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

        // ZB-V uses exactly 2 virtual stages per rank (V=2).
        // Virtual stage IDs for rank stageId: stageId (chunk 0) and stageId + numStages (chunk 1).
        //
        // The schedule interleaves F/B/W operations across both virtual stages:
        // - Forward on virtual stage 0 (chunk 0)
        // - Forward on virtual stage 1 (chunk 1)
        // - BackwardInput on virtual stage 1 (chunk 1, reverse order)
        // - BackwardInput on virtual stage 0 (chunk 0, reverse order)
        // - BackwardWeight fills any remaining gaps

        // Warmup: forwards across both virtual stages
        // Number of warmup forwards scales with position in pipeline
        int warmupForwardsPerChunk = Math.Min(numStages - 1 - stageId, numMicroBatches);

        int forwardCount0 = 0; // Forward count for virtual stage 0
        int forwardCount1 = 0; // Forward count for virtual stage 1
        int backwardInputCount0 = 0;
        int backwardInputCount1 = 0;
        int backwardWeightCount0 = 0;
        int backwardWeightCount1 = 0;

        // Phase 1: Warmup - interleaved forwards across both virtual stages
        // Depth-first: complete a microbatch through both chunks before starting next
        for (int i = 0; i < warmupForwardsPerChunk && forwardCount0 < numMicroBatches; i++)
        {
            // Forward on chunk 0
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = forwardCount0,
                VirtualStageIndex = 0,
                IsWarmup = true,
                IsCooldown = false
            });
            forwardCount0++;

            // Forward on chunk 1 for the same microbatch (if chunk 0 output is ready)
            if (forwardCount1 < forwardCount0 && forwardCount1 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = forwardCount1,
                    VirtualStageIndex = 1,
                    IsWarmup = true,
                    IsCooldown = false
                });
                forwardCount1++;
            }
        }

        // Phase 2: Steady state - F0, F1, B1, B0, W interleaving
        // Continue until all forwards and backwards are complete
        while (forwardCount0 < numMicroBatches ||
               forwardCount1 < numMicroBatches ||
               backwardInputCount0 < numMicroBatches ||
               backwardInputCount1 < numMicroBatches)
        {
            bool isCooldown = forwardCount0 >= numMicroBatches && forwardCount1 >= numMicroBatches;

            // Forward on chunk 0 (if available)
            if (forwardCount0 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = forwardCount0,
                    VirtualStageIndex = 0,
                    IsWarmup = false,
                    IsCooldown = false
                });
                forwardCount0++;
            }

            // Forward on chunk 1 (if chunk 0 has produced output for this microbatch)
            if (forwardCount1 < forwardCount0 && forwardCount1 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = forwardCount1,
                    VirtualStageIndex = 1,
                    IsWarmup = false,
                    IsCooldown = false
                });
                forwardCount1++;
            }

            // BackwardInput on chunk 1 (reverse order - B step, critical path)
            if (backwardInputCount1 < forwardCount1 && backwardInputCount1 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardInput,
                    MicroBatchIndex = backwardInputCount1,
                    VirtualStageIndex = 1,
                    IsWarmup = false,
                    IsCooldown = isCooldown
                });
                backwardInputCount1++;
            }

            // BackwardInput on chunk 0 (after chunk 1's B is done for this microbatch)
            if (backwardInputCount0 < backwardInputCount1 && backwardInputCount0 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardInput,
                    MicroBatchIndex = backwardInputCount0,
                    VirtualStageIndex = 0,
                    IsWarmup = false,
                    IsCooldown = isCooldown
                });
                backwardInputCount0++;
            }

            // BackwardWeight (W) - fills bubbles, process whichever chunk has pending W
            if (backwardWeightCount1 < backwardInputCount1 && backwardWeightCount1 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardWeight,
                    MicroBatchIndex = backwardWeightCount1,
                    VirtualStageIndex = 1,
                    IsWarmup = false,
                    IsCooldown = isCooldown
                });
                backwardWeightCount1++;
            }

            if (backwardWeightCount0 < backwardInputCount0 && backwardWeightCount0 < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.BackwardWeight,
                    MicroBatchIndex = backwardWeightCount0,
                    VirtualStageIndex = 0,
                    IsWarmup = false,
                    IsCooldown = isCooldown
                });
                backwardWeightCount0++;
            }
        }

        // Phase 3: Drain remaining BackwardWeight operations
        while (backwardWeightCount1 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = backwardWeightCount1,
                VirtualStageIndex = 1,
                IsWarmup = false,
                IsCooldown = true
            });
            backwardWeightCount1++;
        }

        while (backwardWeightCount0 < numMicroBatches)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.BackwardWeight,
                MicroBatchIndex = backwardWeightCount0,
                VirtualStageIndex = 0,
                IsWarmup = false,
                IsCooldown = true
            });
            backwardWeightCount0++;
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

        // ZB-V achieves zero bubble when numMicroBatches >= numStages
        // Same as ZB-H2 but with 1F1B-equivalent memory
        if (numMicroBatches >= numStages)
        {
            return 0.0;
        }

        // For insufficient micro-batches, small residual bubble
        // With V=2 virtual stages, the bubble is reduced compared to ZB-H1
        return (double)((long)numStages - numMicroBatches) / (6L * numMicroBatches + numStages);
    }
}
