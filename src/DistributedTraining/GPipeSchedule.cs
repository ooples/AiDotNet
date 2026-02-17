using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements the GPipe scheduling strategy: all forward passes first, then all backward passes.
/// </summary>
/// <remarks>
/// <para>
/// GPipe is the simplest pipeline schedule. It executes all forward micro-batches sequentially
/// through the pipeline, storing all activations, then executes all backward micro-batches
/// in reverse order.
/// </para>
/// <para><b>For Beginners:</b> GPipe is the straightforward approach:
///
/// 1. Push ALL micro-batches through the forward pass (left to right through stages)
/// 2. Then push ALL micro-batches through the backward pass (right to left)
///
/// This creates a "bubble" where stages are idle during pipeline fill and drain.
/// With P stages and M micro-batches, the bubble fraction is approximately (P-1)/(P-1+M).
///
/// For 4 stages and 4 micro-batches:
/// <code>
/// Stage 0: F0 F1 F2 F3 __ __ __ B3 B2 B1 B0
/// Stage 1: __ F0 F1 F2 F3 __ B3 B2 B1 B0 __
/// Stage 2: __ __ F0 F1 F2 F3 B3 B2 B1 __ __
/// Stage 3: __ __ __ F0 F1 F2 B3 B2 __ __ __
/// </code>
///
/// The underscores represent idle time (bubble).
/// </para>
/// <para><b>Reference:</b> Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", 2019.
/// https://arxiv.org/abs/1811.06965</para>
/// </remarks>
public class GPipeSchedule<T> : IPipelineSchedule<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public string Name => "GPipe";

    /// <inheritdoc/>
    public int VirtualStagesPerRank => 1;

    /// <inheritdoc/>
    public IReadOnlyList<PipelineOperation> GetSchedule(int stageId, int numStages, int numMicroBatches)
    {
        ValidateScheduleParameters(stageId, numStages, numMicroBatches);

        var ops = new List<PipelineOperation>();

        // All forward passes
        for (int m = 0; m < numMicroBatches; m++)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Forward,
                MicroBatchIndex = m,
                IsWarmup = m < stageId,
                IsCooldown = false
            });
        }

        // All backward passes (in reverse micro-batch order)
        for (int m = numMicroBatches - 1; m >= 0; m--)
        {
            ops.Add(new PipelineOperation
            {
                Type = PipelineOperationType.Backward,
                MicroBatchIndex = m,
                IsWarmup = false,
                IsCooldown = m >= numMicroBatches - stageId
            });
        }

        return ops;
    }

    /// <inheritdoc/>
    public T EstimateBubbleFraction(int numStages, int numMicroBatches)
    {
        if (numStages <= 1 || numMicroBatches <= 0)
        {
            return NumOps.Zero;
        }

        // GPipe bubble fraction: (P-1) / (P-1+M) where P = stages, M = micro-batches
        T p = NumOps.FromDouble(numStages);
        T m = NumOps.FromDouble(numMicroBatches);
        T pMinusOne = NumOps.Subtract(p, NumOps.One);
        return NumOps.Divide(pMinusOne, NumOps.Add(pMinusOne, m));
    }

    private static void ValidateScheduleParameters(int stageId, int numStages, int numMicroBatches)
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
    }
}
