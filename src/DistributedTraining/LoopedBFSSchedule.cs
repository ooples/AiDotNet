using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the Looped BFS (Breadth-First Schedule) pipeline schedule with multiple virtual stages per rank.
/// </summary>
/// <remarks>
/// <para>
/// Looped BFS, like Interleaved 1F1B, assigns V non-contiguous model chunks ("virtual stages")
/// to each rank. Rank i holds chunks {i, i+P, i+2P, ...} where P is the number of physical ranks.
/// </para>
/// <para>
/// The key difference from Interleaved 1F1B is the scheduling priority:
/// - <b>Interleaved 1F1B (Depth-First)</b>: Prioritizes the <b>earlier microbatch</b>. If microbatch 0
///   is ready for virtual stages 0 and 1, it runs stage 0 for microbatch 0 first.
/// - <b>Looped BFS (Breadth-First)</b>: Prioritizes the <b>earlier virtual stage</b>. If microbatches 0
///   and 1 are ready for virtual stage 0, it processes them both before moving to stage 1.
/// </para>
/// <para><b>For Beginners:</b> Imagine a factory with two assembly stations per worker (V=2).
/// Depth-first (Interleaved 1F1B) means: finish one product at both stations before starting the next.
/// Breadth-first (Looped BFS) means: run all products through station 1, then all through station 2.
///
/// Looped BFS tends to have slightly higher pipeline utilization in some configurations because
/// it minimizes the number of times data needs to cross between physical ranks. However, it
/// may have higher peak memory usage since more microbatches are in flight at each virtual stage.
///
/// Example with 4 GPUs, V=2 (8 total chunks):
/// - GPU 0: chunks 0 and 4
/// - GPU 1: chunks 1 and 5
/// - GPU 2: chunks 2 and 6
/// - GPU 3: chunks 3 and 7
///
/// Looped BFS processes ALL microbatches through chunks 0-3 first (loop 1),
/// then ALL microbatches through chunks 4-7 (loop 2).
/// </para>
/// <para><b>Reference:</b> Lamy-Poirier, "Breadth-First Pipeline Parallelism", 2022.
/// https://arxiv.org/abs/2211.05953</para>
/// </remarks>
public class LoopedBFSSchedule : IPipelineSchedule
{
    private readonly int _virtualStagesPerRank;

    /// <summary>
    /// Creates a new Looped BFS schedule.
    /// </summary>
    /// <param name="virtualStagesPerRank">
    /// Number of model chunks per rank. Default is 2.
    /// Higher values reduce bubble but increase communication.
    /// Must be at least 2 (otherwise use standard 1F1B).
    /// </param>
    public LoopedBFSSchedule(int virtualStagesPerRank = 2)
    {
        if (virtualStagesPerRank < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(virtualStagesPerRank),
                "Looped BFS requires at least 2 virtual stages per rank. " +
                "Use OneForwardOneBackwardSchedule for single-stage scheduling.");
        }

        _virtualStagesPerRank = virtualStagesPerRank;
    }

    /// <inheritdoc/>
    public string Name => "Looped-BFS";

    /// <inheritdoc/>
    public int VirtualStagesPerRank => _virtualStagesPerRank;

    /// <inheritdoc/>
    public IReadOnlyList<PipelineOperation> GetSchedule(int stageId, int numStages, int numMicroBatches)
    {
        if (stageId < 0 || stageId >= numStages)
        {
            throw new ArgumentOutOfRangeException(nameof(stageId),
                $"Stage ID must be between 0 and {numStages - 1}.");
        }

        if (numStages <= 0)
        {
            throw new ArgumentException("Number of stages must be positive.", nameof(numStages));
        }

        if (numMicroBatches <= 0)
        {
            throw new ArgumentException("Number of micro-batches must be positive.", nameof(numMicroBatches));
        }

        var ops = new List<PipelineOperation>();

        // Looped BFS: process all microbatches through each virtual stage loop before moving
        // to the next virtual stage. Within each loop, use 1F1B-style scheduling.
        //
        // Loop structure:
        //   for vStage in 0..V-1:
        //     warmup forwards for this vStage
        //     steady-state 1F1B for this vStage
        //     cooldown backwards for this vStage

        for (int vStage = 0; vStage < _virtualStagesPerRank; vStage++)
        {
            // Within each loop, apply 1F1B scheduling for this virtual stage
            int numWarmupForwards = Math.Min(numStages - 1 - stageId, numMicroBatches);
            int numSteadyState = Math.Max(0, numMicroBatches - numWarmupForwards);
            bool isFirstLoop = vStage == 0;
            bool isLastLoop = vStage == _virtualStagesPerRank - 1;

            // Phase 1: Warmup - forward passes only
            int forwardIdx = 0;
            for (int i = 0; i < numWarmupForwards; i++)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Forward,
                    MicroBatchIndex = forwardIdx,
                    VirtualStageIndex = vStage,
                    IsWarmup = true,
                    IsCooldown = false
                });
                forwardIdx++;
            }

            // Phase 2: Steady state - alternating 1F1B
            int backwardIdx = 0;
            for (int i = 0; i < numSteadyState; i++)
            {
                // Forward
                if (forwardIdx < numMicroBatches)
                {
                    ops.Add(new PipelineOperation
                    {
                        Type = PipelineOperationType.Forward,
                        MicroBatchIndex = forwardIdx,
                        VirtualStageIndex = vStage,
                        IsWarmup = false,
                        IsCooldown = false
                    });
                    forwardIdx++;
                }

                // Backward
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Backward,
                    MicroBatchIndex = backwardIdx,
                    VirtualStageIndex = vStage,
                    IsWarmup = false,
                    IsCooldown = false
                });
                backwardIdx++;
            }

            // Phase 3: Cooldown - remaining backward passes
            while (backwardIdx < numMicroBatches)
            {
                ops.Add(new PipelineOperation
                {
                    Type = PipelineOperationType.Backward,
                    MicroBatchIndex = backwardIdx,
                    VirtualStageIndex = vStage,
                    IsWarmup = false,
                    IsCooldown = true
                });
                backwardIdx++;
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

        // Looped BFS has approximately the same bubble as Interleaved 1F1B
        // but the communication pattern differs. The bubble is roughly:
        // (P-1) / (2*M*V + P - 1)
        // Same asymptotic behavior as Interleaved 1F1B.
        int p = numStages;
        int m = numMicroBatches;
        int v = _virtualStagesPerRank;
        return (double)(p - 1) / (2 * m * v + p - 1);
    }
}
