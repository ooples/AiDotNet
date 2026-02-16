using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the Interleaved 1F1B pipeline schedule with multiple virtual stages per rank.
/// </summary>
/// <remarks>
/// <para>
/// Interleaved 1F1B assigns V non-contiguous model chunks ("virtual stages") to each rank.
/// Rank i holds chunks {i, i+P, i+2P, ...} where P is the number of physical ranks.
/// This reduces the pipeline bubble by a factor of V compared to standard 1F1B.
/// </para>
/// <para>
/// When a microbatch is ready for multiple local virtual stages, Interleaved 1F1B
/// prioritizes the <b>earlier microbatch</b> (depth-first ordering). This is in contrast
/// to Looped BFS which prioritizes the earlier stage.
/// </para>
/// <para><b>For Beginners:</b> Standard 1F1B gives each GPU one big chunk of the model.
/// Interleaved 1F1B gives each GPU V smaller, evenly-spaced chunks instead.
///
/// Example with 4 GPUs, V=2 (8 total chunks):
/// - GPU 0: chunks 0 and 4
/// - GPU 1: chunks 1 and 5
/// - GPU 2: chunks 2 and 6
/// - GPU 3: chunks 3 and 7
///
/// This means each microbatch visits each GPU twice (once for each chunk), creating more
/// opportunities to interleave work and reduce idle time. The bubble shrinks from
/// ~(P-1)/(2M+P-1) to ~(P-1)/(2MV+P-1).
///
/// Used in production by Megatron-LM v2 and NVIDIA NeMo.
/// </para>
/// <para><b>Reference:</b> Narayanan et al., "Efficient Large-Scale Language Model Training
/// on GPU Clusters Using Megatron-LM", SC 2021. https://arxiv.org/abs/2104.04473</para>
/// </remarks>
public class Interleaved1F1BSchedule : IPipelineSchedule
{
    private readonly int _virtualStagesPerRank;

    /// <summary>
    /// Creates a new Interleaved 1F1B schedule.
    /// </summary>
    /// <param name="virtualStagesPerRank">
    /// Number of model chunks per rank. Default is 2.
    /// Higher values reduce bubble but increase communication.
    /// Must be at least 2 (otherwise use standard 1F1B).
    /// </param>
    public Interleaved1F1BSchedule(int virtualStagesPerRank = 2)
    {
        if (virtualStagesPerRank < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(virtualStagesPerRank),
                "Interleaved schedule requires at least 2 virtual stages per rank. " +
                "Use OneForwardOneBackwardSchedule for single-stage scheduling.");
        }

        _virtualStagesPerRank = virtualStagesPerRank;
    }

    /// <inheritdoc/>
    public string Name => "Interleaved-1F1B";

    /// <inheritdoc/>
    public int VirtualStagesPerRank => _virtualStagesPerRank;

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

        // Use long arithmetic to prevent overflow when numStages * _virtualStagesPerRank
        // exceeds int.MaxValue, then validate the result fits in int.
        long totalVirtualStagesLong = (long)numStages * _virtualStagesPerRank;
        if (totalVirtualStagesLong > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Total virtual stages ({totalVirtualStagesLong}) exceeds int.MaxValue. " +
                "Reduce numStages or virtualStagesPerRank.");
        }
        int totalVirtualStages = (int)totalVirtualStagesLong;

        // Each rank handles V virtual stages. Virtual stage IDs for rank stageId:
        // stageId, stageId + numStages, stageId + 2*numStages, ...
        // In the interleaved schedule, microbatches flow through all virtual stages.

        // Warmup: number of forward passes before steady state begins
        // For interleaved, warmup is proportional to (totalVirtualStages - rank's first virtual stage - 1)
        long numWarmupForwardsLong = (long)numMicroBatches * _virtualStagesPerRank;
        int numWarmupForwards = Math.Min(
            totalVirtualStages - 1 - stageId,
            numWarmupForwardsLong > int.MaxValue ? int.MaxValue : (int)numWarmupForwardsLong);

        int totalForwards = numWarmupForwardsLong > int.MaxValue
            ? int.MaxValue
            : (int)numWarmupForwardsLong;
        int totalBackwards = totalForwards;
        int forwardsDone = 0;
        int backwardsDone = 0;

        // Phase 1: Warmup - forwards across virtual stages in depth-first order
        // (prioritize earlier microbatch over earlier virtual stage)
        for (int i = 0; i < numWarmupForwards && forwardsDone < totalForwards; i++)
        {
            // Depth-first: cycle through virtual stages for each microbatch
            int vStage = forwardsDone % _virtualStagesPerRank;
            int microBatch = forwardsDone / _virtualStagesPerRank;

            if (microBatch < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = microBatch,
                    VirtualStageIndex = vStage,
                    IsWarmup = true,
                    IsCooldown = false
                });
                forwardsDone++;
            }
        }

        // Phase 2: Steady state - alternating forward and backward
        while (forwardsDone < totalForwards || backwardsDone < totalBackwards)
        {
            // One forward (if available)
            if (forwardsDone < totalForwards)
            {
                int vStage = forwardsDone % _virtualStagesPerRank;
                int microBatch = forwardsDone / _virtualStagesPerRank;

                if (microBatch < numMicroBatches)
                {
                    ops.Add(new PipelineOperation
                    {
                        Type = PipelineOperationType.Forward,
                        MicroBatchIndex = microBatch,
                        VirtualStageIndex = vStage,
                        IsWarmup = false,
                        IsCooldown = false
                    });
                    forwardsDone++;
                }
            }

            // One backward (if available)
            if (backwardsDone < totalBackwards)
            {
                int vStage = backwardsDone % _virtualStagesPerRank;
                int microBatch = backwardsDone / _virtualStagesPerRank;

                if (microBatch < numMicroBatches)
                {
                    bool isCooldown = forwardsDone >= totalForwards;
                    ops.Add(new PipelineOperation
                    {
                        Type = PipelineOperationType.Backward,
                        MicroBatchIndex = microBatch,
                        VirtualStageIndex = _virtualStagesPerRank - 1 - vStage, // Backward visits in reverse
                        IsWarmup = false,
                        IsCooldown = isCooldown
                    });
                    backwardsDone++;
                }
            }
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

        // Interleaved 1F1B bubble: (P-1) / (2*M*V + P - 1)
        // V times smaller than standard 1F1B
        double p = numStages;
        double m = numMicroBatches;
        double v = _virtualStagesPerRank;
        return (p - 1) / (2.0 * m * v + p - 1.0);
    }
}
