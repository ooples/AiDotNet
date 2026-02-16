using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Pipeline Parallel model wrapper - splits model into stages across ranks.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Pipeline Parallelism divides the model vertically into stages, with each process
/// owning specific layers. Input mini-batches are divided into micro-batches that flow through
/// the pipeline stages sequentially. This enables training models too large to fit on a single device
/// while maintaining good hardware utilization through micro-batch pipelining.
/// </para>
/// <para><b>For Beginners:</b>
/// Pipeline parallelism is like an assembly line for training. Imagine a deep neural network as
/// a tall building - instead of one person (GPU) handling all floors, we assign different floors
/// to different people. Process 0 handles layers 0-10, Process 1 handles layers 11-20, etc.
///
/// To keep everyone busy (avoid idle time), we split each batch into smaller "micro-batches" that
/// flow through the pipeline like cars on an assembly line. While Process 1 is working on micro-batch 1,
/// Process 0 can start on micro-batch 2.
/// </para>
/// <para><b>Supported Features (Issue #463):</b>
/// <list type="number">
/// <item><description>
/// <b>7 Pipeline Schedules</b>: GPipe, 1F1B, ZB-H1, ZB-H2, ZB-V, Interleaved 1F1B, Looped BFS.
/// Zero Bubble schedules decompose backward into BackwardInput + BackwardWeight for optimal throughput.
/// </description></item>
/// <item><description>
/// <b>Virtual Stages</b>: Multi-stage schedules (Interleaved 1F1B, Looped BFS, ZB-V) assign
/// multiple non-contiguous model chunks per rank, reducing pipeline bubble by factor V.
/// </description></item>
/// <item><description>
/// <b>Micro-Batch Slicing</b>: Input is automatically sliced into micro-batches that flow
/// through the pipeline independently.
/// </description></item>
/// <item><description>
/// <b>Backward Decomposition</b>: If the wrapped model implements <see cref="IPipelineDecomposableModel{T, TInput, TOutput}"/>,
/// BackwardInput and BackwardWeight are truly decomposed. Otherwise, a compatible emulation is used.
/// </description></item>
/// <item><description>
/// <b>Activation Checkpointing</b>: Trade compute for memory by recomputing activations from
/// checkpoints during the backward pass.
/// </description></item>
/// <item><description>
/// <b>Load-Balanced Partitioning</b>: Balance compute across stages via dynamic programming.
/// </description></item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class PipelineParallelModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private readonly int _microBatchSize;
    private readonly IPipelinePartitionStrategy<T>? _partitionStrategy;
    private readonly IPipelineSchedule _schedule;
    private readonly ActivationCheckpointConfig _checkpointConfig;
    private int _stageId;
    private int _numStages;
    private int _virtualStagesPerRank;

    // Total virtual stages across all ranks
    private int _totalVirtualStages;

    // Parameter ranges for each virtual stage this rank owns.
    // For single-stage schedules (V=1): one entry mapping to the full shard.
    // For multi-stage schedules (V>1): V entries for non-contiguous model chunks.
    // Key = local virtual stage index (0..V-1), Value = (StartIndex, Size) in full param vector.
    private readonly Dictionary<int, (int StartIndex, int Size)> _virtualStagePartitions = new();

    // Activation storage for checkpointing.
    // Key format: (microBatchIndex * _virtualStagesPerRank + virtualStageIndex) for uniqueness.
    private readonly Dictionary<int, Vector<T>> _checkpointedActivations = new();

    // Cached state from BackwardInput for later use by BackwardWeight (Zero Bubble B/W decomposition).
    // Key format: (microBatchIndex * _virtualStagesPerRank + virtualStageIndex).
    private readonly Dictionary<int, object?> _cachedBackwardState = new();

    // Cached weight gradients from BackwardInput for fallback accumulation when model
    // does not support IPipelineDecomposableModel (emulated B/W split).
    private readonly Dictionary<int, Vector<T>> _cachedWeightGradients = new();

    // Whether the wrapped model supports true B/W decomposition
    private bool _supportsDecomposedBackward;

    // Communication tag ranges to prevent collisions between forward activations,
    // backward gradients, and predict-time messages.
    private const int ActivationTagBase = 0;
    private const int GradientTagBase = 1_000_000;
    private const int PredictTagBase = 2_000_000;

    /// <summary>
    /// Gets the pipeline schedule used by this model.
    /// </summary>
    /// <remarks>
    /// This property is internal. Configure the schedule via <c>AiModelBuilder</c> methods
    /// (e.g., <c>ConfigurePipelineParallelism</c>) rather than accessing this directly.
    /// </remarks>
    internal IPipelineSchedule Schedule => _schedule;

    /// <summary>
    /// Gets the activation checkpoint configuration.
    /// </summary>
    /// <remarks>
    /// This property is internal. Configure checkpointing via <c>AiModelBuilder</c> methods
    /// rather than accessing this directly.
    /// </remarks>
    internal ActivationCheckpointConfig CheckpointConfig => _checkpointConfig;

    /// <summary>
    /// Gets the partition strategy, or null if using uniform partitioning.
    /// </summary>
    /// <remarks>
    /// This property is internal. Configure the partition strategy via <c>AiModelBuilder</c> methods
    /// rather than accessing this directly.
    /// </remarks>
    internal IPipelinePartitionStrategy<T>? PartitionStrategy => _partitionStrategy;

    /// <summary>
    /// Gets the estimated pipeline bubble fraction for the current configuration.
    /// </summary>
    public double EstimatedBubbleFraction
    {
        get
        {
            EnsureShardingInitialized();

            return _schedule.EstimateBubbleFraction(_numStages, _microBatchSize);
        }
    }

    /// <summary>
    /// Creates a new Pipeline Parallel model.
    /// </summary>
    /// <param name="wrappedModel">The model to split into pipeline stages.</param>
    /// <param name="config">Configuration for sharding and communication.</param>
    /// <param name="microBatchSize">Number of micro-batches to split the input into (default: 1).</param>
    /// <param name="partitionStrategy">
    /// Strategy for partitioning parameters across stages. If null, uses uniform partitioning.
    /// </param>
    /// <param name="schedule">
    /// Pipeline execution schedule. If null, uses <see cref="GPipeSchedule"/>.
    /// </param>
    /// <param name="checkpointConfig">
    /// Activation checkpointing configuration. If null, checkpointing is disabled.
    /// </param>
    public PipelineParallelModel(
        IFullModel<T, TInput, TOutput> wrappedModel,
        IShardingConfiguration<T> config,
        int microBatchSize = 1,
        IPipelinePartitionStrategy<T>? partitionStrategy = null,
        IPipelineSchedule? schedule = null,
        ActivationCheckpointConfig? checkpointConfig = null)
        : base(wrappedModel, config)
    {
        if (microBatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(microBatchSize),
                "Micro batch size must be at least 1.");
        }

        _microBatchSize = microBatchSize;
        _partitionStrategy = partitionStrategy;
        _schedule = schedule ?? new GPipeSchedule();
        _checkpointConfig = checkpointConfig ?? new ActivationCheckpointConfig();

        // Activation checkpointing recomputation strategies (Selective, Full) require
        // layer-level forward pass decomposition that is not yet implemented.
        // Only interval-based checkpoint storage is currently functional.
        if (_checkpointConfig.Enabled &&
            _checkpointConfig.RecomputeStrategy != RecomputeStrategy.None)
        {
            throw new NotImplementedException(
                $"Activation checkpointing with RecomputeStrategy.{_checkpointConfig.RecomputeStrategy} " +
                "is not yet implemented. Use RecomputeStrategy.None to enable checkpoint storage " +
                "without recomputation, or disable checkpointing entirely.");
        }
    }

    /// <summary>
    /// Called before InitializeSharding to set up derived class state.
    /// </summary>
    protected override void OnBeforeInitializeSharding()
    {
        _stageId = Config.CommunicationBackend.Rank;
        _numStages = Config.CommunicationBackend.WorldSize;
        _virtualStagesPerRank = _schedule.VirtualStagesPerRank;
        _totalVirtualStages = checked(_numStages * _virtualStagesPerRank);
        _supportsDecomposedBackward = WrappedModel is IPipelineDecomposableModel<T, TInput, TOutput>;
    }

    /// <summary>
    /// Initializes pipeline parallelism by partitioning parameters into stages,
    /// including virtual stage partitions for multi-stage schedules.
    /// </summary>
    /// <remarks>
    /// <para>When the wrapped model implements <see cref="ILayeredModel{T}"/>, this method
    /// uses layer-aware partitioning that respects layer boundaries and balances computational
    /// cost across stages using estimated FLOPs. This avoids splitting parameters in the
    /// middle of a layer, which would corrupt that layer's weights.</para>
    ///
    /// <para>When the wrapped model does not implement <see cref="ILayeredModel{T}"/>,
    /// falls back to simple parameter-count-based partitioning.</para>
    /// </remarks>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        _virtualStagePartitions.Clear();

        if (_virtualStagesPerRank > 1)
        {
            // Multi-stage schedule: partition into totalVirtualStages chunks,
            // then assign V non-contiguous chunks to this rank.
            // Rank i gets virtual stages: i, i+P, i+2P, ...
            (int StartIndex, int Size)[] vsPartitions;

            if (_partitionStrategy is not null)
            {
                // Use the configured partition strategy for load-balanced partitioning
                // across all virtual stages (not just physical stages)
                vsPartitions = _partitionStrategy.ComputePartition(totalParams, _totalVirtualStages);

                if (vsPartitions is null || vsPartitions.Length != _totalVirtualStages)
                {
                    throw new InvalidOperationException(
                        $"Partition strategy returned {(vsPartitions is null ? "null" : $"{vsPartitions.Length} partitions")} " +
                        $"but expected exactly {_totalVirtualStages} partitions for {_virtualStagesPerRank} virtual stages per rank.");
                }

                // Validate bounds for all virtual stage partitions
                for (int vs = 0; vs < _totalVirtualStages; vs++)
                {
                    var (start, size) = vsPartitions[vs];
                    if (start < 0 || size < 0 || start + size > totalParams)
                    {
                        throw new InvalidOperationException(
                            $"Partition strategy returned invalid partition for virtual stage {vs}: " +
                            $"StartIndex={start}, Size={size}, but total parameters is {totalParams}.");
                    }
                }
            }
            else
            {
                // Uniform partitioning
                vsPartitions = new (int StartIndex, int Size)[_totalVirtualStages];
                int baseChunkSize = totalParams / _totalVirtualStages;
                int remainder = totalParams % _totalVirtualStages;
                int offset = 0;
                for (int vs = 0; vs < _totalVirtualStages; vs++)
                {
                    int size = baseChunkSize + (vs < remainder ? 1 : 0);
                    vsPartitions[vs] = (offset, size);
                    offset += size;
                }
            }

            // Assign this rank's virtual stages
            int totalShardSize = 0;
            for (int v = 0; v < _virtualStagesPerRank; v++)
            {
                int globalVirtualStageId = _stageId + v * _numStages;
                if (globalVirtualStageId < _totalVirtualStages)
                {
                    var partition = vsPartitions[globalVirtualStageId];
                    _virtualStagePartitions[v] = partition;
                    totalShardSize += partition.Size;
                }
            }

            // The shard for base class is the union of all virtual stage parameters.
            // Use the first virtual stage's start as the shard start.
            if (_virtualStagePartitions.Count > 0)
            {
                ShardStartIndex = _virtualStagePartitions[0].StartIndex;
                ShardSize = totalShardSize;
            }
            else
            {
                ShardStartIndex = 0;
                ShardSize = 0;
            }
        }
        else if (WrappedModel is ILayeredModel<T> layeredModel && layeredModel.LayerCount > 0
                 && _partitionStrategy is null)
        {
            // Layer-aware partitioning: respects layer boundaries to avoid splitting
            // a layer's parameters across stages
            InitializeLayerAwareSharding(layeredModel, fullParameters);
            _virtualStagePartitions[0] = (ShardStartIndex, ShardSize);
        }
        else
        {
            // Single-stage schedule: standard partitioning
            if (_partitionStrategy is not null)
            {
                var partitions = _partitionStrategy.ComputePartition(totalParams, _numStages);

                if (partitions is null || partitions.Length != _numStages)
                {
                    throw new InvalidOperationException(
                        $"Partition strategy returned {(partitions is null ? "null" : $"{partitions.Length} partitions")} " +
                        $"but expected exactly {_numStages} partitions.");
                }

                var stagePartition = partitions[_stageId];
                if (stagePartition.StartIndex < 0 || stagePartition.Size < 0 ||
                    stagePartition.StartIndex + stagePartition.Size > totalParams)
                {
                    throw new InvalidOperationException(
                        $"Partition strategy returned invalid partition for stage {_stageId}: " +
                        $"StartIndex={stagePartition.StartIndex}, Size={stagePartition.Size}, " +
                        $"but total parameters is {totalParams}.");
                }

                ShardStartIndex = stagePartition.StartIndex;
                ShardSize = stagePartition.Size;
            }
            else
            {
                int baseShardSize = totalParams / _numStages;
                int leftover = totalParams % _numStages;

                ShardSize = baseShardSize + (_stageId < leftover ? 1 : 0);
                ShardStartIndex = _stageId * baseShardSize + Math.Min(_stageId, leftover);
            }

            _virtualStagePartitions[0] = (ShardStartIndex, ShardSize);
        }

        // Extract this stage's parameters (union of all virtual stage params)
        if (ShardSize > 0)
        {
            var shardData = new T[ShardSize];
            if (_virtualStagesPerRank > 1)
            {
                // For multi-stage: gather non-contiguous chunks
                int destOffset = 0;
                var paramArray = fullParameters.ToArray();
                for (int v = 0; v < _virtualStagesPerRank; v++)
                {
                    if (_virtualStagePartitions.TryGetValue(v, out var partition))
                    {
                        Array.Copy(paramArray, partition.StartIndex, shardData, destOffset, partition.Size);
                        destOffset += partition.Size;
                    }
                }
            }
            else
            {
                Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
            }
            LocalShard = new Vector<T>(shardData);
        }
        else
        {
            LocalShard = new Vector<T>(0);
        }

        CachedFullParameters = null;
    }

    /// <summary>
    /// Updates the local parameter shard from a full parameter vector, correctly handling
    /// non-contiguous virtual stage partitions for V&gt;1 schedules.
    /// </summary>
    /// <remarks>
    /// For single-stage schedules (V=1), parameters are contiguous and the base class
    /// <see cref="ShardedModelBase{T, TInput, TOutput}.UpdateLocalShardFromFull"/> works fine.
    /// For multi-stage schedules (V&gt;1), the local shard is a concatenation of non-contiguous
    /// chunks from different parts of the full parameter vector. This method mirrors the
    /// extraction logic in <see cref="InitializeSharding"/> to rebuild the shard correctly.
    /// </remarks>
    private void UpdateLocalShardFromFullParameters(Vector<T> fullParameters)
    {
        if (_virtualStagesPerRank <= 1)
        {
            // Single virtual stage: shard is contiguous, use base class
            UpdateLocalShardFromFull(fullParameters);
            return;
        }

        // Multi-stage: gather non-contiguous chunks in virtual-stage order
        var shardData = new T[ShardSize];
        int destOffset = 0;
        var paramArray = fullParameters.ToArray();

        for (int v = 0; v < _virtualStagesPerRank; v++)
        {
            if (_virtualStagePartitions.TryGetValue(v, out var partition))
            {
                Array.Copy(paramArray, partition.StartIndex, shardData, destOffset, partition.Size);
                destOffset += partition.Size;
            }
        }

        LocalShard = new Vector<T>(shardData);
        InvalidateCache();
    }

    /// <summary>
    /// Performs layer-aware partitioning that respects layer boundaries and balances
    /// computational cost across pipeline stages.
    /// </summary>
    /// <param name="layeredModel">The model with layer-level access.</param>
    /// <param name="fullParameters">The full parameter vector.</param>
    private void InitializeLayerAwareSharding(ILayeredModel<T> layeredModel, Vector<T> fullParameters)
    {
        var allLayerInfo = layeredModel.GetAllLayerInfo();
        int layerCount = allLayerInfo.Count;
        var paramArray = fullParameters.ToArray();

        if (layerCount == 0 || _numStages <= 1)
        {
            // Single stage or no layers: take all parameters
            ShardStartIndex = 0;
            ShardSize = fullParameters.Length;
            LocalShard = new Vector<T>(paramArray);
            return;
        }

        // Compute total FLOPs for balanced partitioning
        long totalFlops = 0;
        foreach (var info in allLayerInfo)
        {
            totalFlops += info.EstimatedFlops;
        }

        // Target FLOPs per stage for balanced distribution
        long targetFlopsPerStage = totalFlops / _numStages;

        // Greedily assign layers to stages, trying to balance FLOPs
        // stageStartLayer[s] = index of first layer in stage s
        var stageStartLayer = new int[_numStages + 1];
        stageStartLayer[0] = 0;

        int currentLayer = 0;
        for (int stage = 0; stage < _numStages - 1; stage++)
        {
            long stageFlops = 0;
            int layersInStage = 0;
            int candidateBoundary = -1;

            // Add layers until we exceed the target FLOPs or run out of layers
            // Always assign at least one layer per stage
            while (currentLayer < layerCount)
            {
                long layerFlops = allLayerInfo[currentLayer].EstimatedFlops;
                stageFlops += layerFlops;
                currentLayer++;
                layersInStage++;

                // Check if we've exceeded the target for this stage
                // But ensure remaining stages each get at least one layer
                int remainingLayers = layerCount - currentLayer;
                int remainingStages = _numStages - stage - 1;

                if (stageFlops >= targetFlopsPerStage && remainingLayers >= remainingStages)
                {
                    // Validate the partition point before accepting it
                    int partitionAfter = currentLayer - 1;
                    if (partitionAfter < layerCount - 1 &&
                        layeredModel.ValidatePartitionPoint(partitionAfter))
                    {
                        candidateBoundary = currentLayer;
                        break;
                    }

                    // Partition point not valid here; continue consuming layers
                    // to find the next valid boundary
                }
            }

            // If we found a valid boundary, it already equals currentLayer (set at line 214)
            if (candidateBoundary < 0)
            {
                // No valid forward boundary found; try backward search from currentLayer
                // to find the nearest valid partition point within this stage's range
                int searchStart = currentLayer - 1;
                int searchEnd = stageStartLayer[stage];
                bool foundBackward = false;

                for (int probe = searchStart; probe > searchEnd; probe--)
                {
                    if (layeredModel.ValidatePartitionPoint(probe - 1))
                    {
                        currentLayer = probe;
                        foundBackward = true;
                        break;
                    }
                }

                if (!foundBackward)
                {
                    if (layersInStage == 0 && currentLayer < layerCount)
                    {
                        // Ensure at least one layer per stage
                        System.Diagnostics.Debug.WriteLine(
                            $"[PipelineParallel] Stage {stage}: No valid partition point found, " +
                            $"forcing boundary at layer {currentLayer}.");
                        currentLayer++;
                    }
                    else if (currentLayer <= stageStartLayer[stage])
                    {
                        throw new InvalidOperationException(
                            $"Cannot find a valid partition point for stage {stage} " +
                            $"in layer range [{stageStartLayer[stage]}, {currentLayer}). " +
                            "The model's ValidatePartitionPoint rejected all candidates.");
                    }
                    else
                    {
                        System.Diagnostics.Debug.WriteLine(
                            $"[PipelineParallel] Stage {stage}: Backward search exhausted " +
                            $"at layer {currentLayer} with {layersInStage} layer(s) already assigned.");
                    }
                }
            }

            stageStartLayer[stage + 1] = currentLayer;
        }

        // Last stage gets all remaining layers
        stageStartLayer[_numStages] = layerCount;

        // Compute parameter offset and size for this stage
        int firstLayerInStage = stageStartLayer[_stageId];
        int lastLayerInStage = stageStartLayer[_stageId + 1] - 1;

        if (firstLayerInStage >= layerCount || lastLayerInStage < firstLayerInStage)
        {
            // This stage has no layers (more stages than layers)
            ShardStartIndex = 0;
            ShardSize = 0;
            LocalShard = new Vector<T>(0);
            return;
        }

        // Use LayerInfo parameter offsets for precise slicing
        int stageParamStart = allLayerInfo[firstLayerInStage].ParameterOffset;
        long stageParamEndLong = (long)allLayerInfo[lastLayerInStage].ParameterOffset +
                                 allLayerInfo[lastLayerInStage].ParameterCount;

        if (stageParamEndLong > fullParameters.Length)
        {
            throw new InvalidOperationException(
                $"Stage {_stageId} parameter range [{stageParamStart}, {stageParamEndLong}) exceeds " +
                $"total parameter count ({fullParameters.Length}). " +
                "LayerInfo metadata may be stale or inconsistent with the model parameters.");
        }

        if (stageParamEndLong > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Stage {_stageId} parameter end ({stageParamEndLong}) exceeds int.MaxValue. " +
                "Models with more than int.MaxValue parameters are not supported.");
        }

        int stageParamEnd = (int)stageParamEndLong;

        ShardStartIndex = stageParamStart;
        ShardSize = stageParamEnd - stageParamStart;

        if (ShardSize > 0)
        {
            var shardData = new T[ShardSize];
            Array.Copy(paramArray, ShardStartIndex, shardData, 0, ShardSize);
            LocalShard = new Vector<T>(shardData);
        }
        else
        {
            LocalShard = new Vector<T>(0);
        }
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Ensure sharding (and _numStages / _stageId) is initialized before using them
        EnsureShardingInitialized();

        // Pipeline parallel training using the configured schedule
        var scheduleOps = _schedule.GetSchedule(_stageId, _numStages, _microBatchSize);

        // Validate schedule output: externally injectable schedules may emit invalid indices
        foreach (var op in scheduleOps)
        {
            if (op.MicroBatchIndex < 0 || op.MicroBatchIndex >= _microBatchSize)
            {
                throw new InvalidOperationException(
                    $"Schedule '{_schedule.Name}' emitted MicroBatchIndex={op.MicroBatchIndex} " +
                    $"but valid range is [0, {_microBatchSize - 1}].");
            }

            if (op.VirtualStageIndex < 0 || op.VirtualStageIndex >= _virtualStagesPerRank)
            {
                throw new InvalidOperationException(
                    $"Schedule '{_schedule.Name}' emitted VirtualStageIndex={op.VirtualStageIndex} " +
                    $"but valid range is [0, {_virtualStagesPerRank - 1}].");
            }
        }

        // Gather full parameters before training
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Save parameters BEFORE training to compute gradients
        var parametersBefore = new Vector<T>(fullParams.ToArray());

        // Accumulated weight gradients across all micro-batches
        Vector<T>? accumulatedGradients = null;

        // Slice input and targets into micro-batches
        var microBatches = SliceInputIntoMicroBatches(input);
        var microBatchTargets = SliceTargetIntoMicroBatches(expectedOutput);

        // Track activations per (microBatch, virtualStage) for backward pass
        var forwardInputs = new Dictionary<int, TInput>();
        var forwardOutputs = new Dictionary<int, TOutput>();

        // Clear state from previous iteration
        _checkpointedActivations.Clear();
        _cachedBackwardState.Clear();
        _cachedWeightGradients.Clear();

        foreach (var op in scheduleOps)
        {
            int opKey = GetOperationKey(op.MicroBatchIndex, op.VirtualStageIndex);

            if (op.Type == PipelineOperationType.Forward)
            {
                ExecuteForward(op, microBatches, forwardInputs, forwardOutputs, opKey);
            }
            else if (op.Type == PipelineOperationType.Backward)
            {
                // Combined backward: compute all gradients and communicate in one step.
                // Used by traditional schedules (GPipe, 1F1B).
                var microBatchInput = RetrieveMicroBatchInput(opKey, forwardInputs, microBatches, op);
                var microBatchTarget = GetMicroBatchTarget(op.MicroBatchIndex, microBatchTargets, expectedOutput);

                var gradientVector = WrappedModel.ComputeGradients(microBatchInput, microBatchTarget);

                ReceiveAndAccumulateDownstreamGradients(gradientVector, op.MicroBatchIndex, op.VirtualStageIndex);
                SendGradientsUpstream(gradientVector, op.MicroBatchIndex, op.VirtualStageIndex);
                accumulatedGradients = AccumulateGradients(accumulatedGradients, gradientVector);

                FreeConsumedActivations(opKey, forwardInputs, forwardOutputs);
            }
            else if (op.Type == PipelineOperationType.BackwardInput)
            {
                // Zero Bubble B step: compute activation gradients (critical path).
                // Upstream stage is waiting for these gradients.
                var microBatchInput = RetrieveMicroBatchInput(opKey, forwardInputs, microBatches, op);
                var microBatchTarget = GetMicroBatchTarget(op.MicroBatchIndex, microBatchTargets, expectedOutput);

                if (_supportsDecomposedBackward)
                {
                    // True decomposition: compute only activation gradients
                    var decomposable = (IPipelineDecomposableModel<T, TInput, TOutput>)WrappedModel;
                    var (activationGrads, cachedState) = decomposable.ComputeActivationGradients(
                        microBatchInput, microBatchTarget);

                    ReceiveAndAccumulateDownstreamGradients(activationGrads, op.MicroBatchIndex, op.VirtualStageIndex);
                    SendGradientsUpstream(activationGrads, op.MicroBatchIndex, op.VirtualStageIndex);

                    // Cache state for BackwardWeight to avoid redundant computation
                    _cachedBackwardState[opKey] = cachedState;
                }
                else
                {
                    // Emulated decomposition: compute full gradients now, send activation grads upstream,
                    // cache weight gradients for BackwardWeight step to accumulate later.
                    var fullGradients = WrappedModel.ComputeGradients(microBatchInput, microBatchTarget);

                    ReceiveAndAccumulateDownstreamGradients(fullGradients, op.MicroBatchIndex, op.VirtualStageIndex);
                    SendGradientsUpstream(fullGradients, op.MicroBatchIndex, op.VirtualStageIndex);

                    // Cache the weight gradients for the W step
                    _cachedWeightGradients[opKey] = fullGradients;
                }
            }
            else if (op.Type == PipelineOperationType.BackwardWeight)
            {
                // Zero Bubble W step: compute weight gradients (fills bubbles).
                // No other stage depends on this - can be deferred.
                Vector<T> weightGradients;

                if (_supportsDecomposedBackward)
                {
                    // True decomposition: compute only weight gradients
                    var decomposable = (IPipelineDecomposableModel<T, TInput, TOutput>)WrappedModel;
                    var microBatchInput = RetrieveMicroBatchInput(opKey, forwardInputs, microBatches, op);
                    var microBatchTarget = GetMicroBatchTarget(op.MicroBatchIndex, microBatchTargets, expectedOutput);

                    _cachedBackwardState.TryGetValue(opKey, out var cachedState);
                    weightGradients = decomposable.ComputeWeightGradients(
                        microBatchInput, microBatchTarget, cachedState);
                    _cachedBackwardState.Remove(opKey);
                }
                else
                {
                    // Emulated: use cached gradients from BackwardInput step
                    if (_cachedWeightGradients.TryGetValue(opKey, out var cached))
                    {
                        weightGradients = cached;
                        _cachedWeightGradients.Remove(opKey);
                    }
                    else
                    {
                        // Fallback: recompute full gradients
                        var microBatchInput = RetrieveMicroBatchInput(opKey, forwardInputs, microBatches, op);
                        var microBatchTarget = GetMicroBatchTarget(op.MicroBatchIndex, microBatchTargets, expectedOutput);
                        weightGradients = WrappedModel.ComputeGradients(microBatchInput, microBatchTarget);
                    }
                }

                accumulatedGradients = AccumulateGradients(accumulatedGradients, weightGradients);
                FreeConsumedActivations(opKey, forwardInputs, forwardOutputs);
            }
        }

        // Apply accumulated gradients averaged across micro-batches
        if (accumulatedGradients is not null)
        {
            T microBatchCount = NumOps.FromDouble(_microBatchSize);
            for (int i = 0; i < accumulatedGradients.Length; i++)
            {
                accumulatedGradients[i] = NumOps.Divide(accumulatedGradients[i], microBatchCount);
            }

            WrappedModel.SetParameters(parametersBefore);
            WrappedModel.ApplyGradients(accumulatedGradients, Config.LearningRate);
        }

        // Extract this stage's parameter shard
        var updatedParams = WrappedModel.GetParameters();
        UpdateLocalShardFromFullParameters(updatedParams);
        InvalidateCache();

        // Clean up all activation/gradient storage
        _checkpointedActivations.Clear();
        _cachedBackwardState.Clear();
        _cachedWeightGradients.Clear();

        // Synchronize parameters across stages for consistency
        if (Config.AutoSyncGradients)
        {
            SynchronizeGradients();
        }
    }

    /// <summary>
    /// Executes a forward operation, handling virtual stage routing and activation checkpointing.
    /// </summary>
    private void ExecuteForward(
        PipelineOperation op,
        Dictionary<int, TInput> microBatches,
        Dictionary<int, TInput> forwardInputs,
        Dictionary<int, TOutput> forwardOutputs,
        int opKey)
    {
        var stageInput = GetStageInput(microBatches, op.MicroBatchIndex, op.VirtualStageIndex, forwardOutputs);

        // After consuming the previous virtual stage's output, free it to reclaim memory.
        // This handles the case where FreeNonCheckpointedActivations retained the output
        // for cross-virtual-stage dependencies (e.g., Looped BFS).
        if (op.VirtualStageIndex > 0
            && !_checkpointedActivations.ContainsKey(
                GetOperationKey(op.MicroBatchIndex, op.VirtualStageIndex - 1)))
        {
            forwardOutputs.Remove(GetOperationKey(op.MicroBatchIndex, op.VirtualStageIndex - 1));
        }

        // Checkpoint activation if configured
        if (ShouldCheckpointActivation(opKey))
        {
            var inputVector = ConversionsHelper.ConvertToVector<T, TInput>(stageInput);
            _checkpointedActivations[opKey] = inputVector;
        }

        forwardInputs[opKey] = stageInput;

        // Forward pass through the model
        var stageOutput = WrappedModel.Predict(stageInput);
        forwardOutputs[opKey] = stageOutput;

        // Send activations to the next stage in the pipeline
        SendActivationsForward(stageOutput, op.MicroBatchIndex, op.VirtualStageIndex);
    }

    /// <summary>
    /// Slices input into micro-batches by converting to a vector and dividing evenly.
    /// If the input cannot be sliced (e.g., single sample), all micro-batches use the same input.
    /// </summary>
    private Dictionary<int, TInput> SliceInputIntoMicroBatches(TInput fullData)
    {
        var slices = new Dictionary<int, TInput>();

        if (_microBatchSize <= 1)
        {
            slices[0] = fullData;
            return slices;
        }

        // Convert to vector for slicing
        Vector<T> fullVector;
        try
        {
            fullVector = ConversionsHelper.ConvertToVector<T, TInput>(fullData);
        }
        catch (InvalidOperationException ex)
        {
            throw new InvalidOperationException(
                $"Cannot slice input of type {typeof(TInput).Name} into micro-batches. " +
                "The input must be convertible to a vector for pipeline parallel training with micro-batches > 1.",
                ex);
        }

        int totalElements = fullVector.Length;
        int microBatchElements = totalElements / _microBatchSize;

        if (microBatchElements <= 0)
        {
            throw new InvalidOperationException(
                $"Cannot slice {totalElements} elements into {_microBatchSize} micro-batches. " +
                $"Reduce pipelineMicroBatchSize to at most {totalElements}.");
        }

        var fullArray = fullVector.ToArray();
        for (int i = 0; i < _microBatchSize; i++)
        {
            int startIdx = i * microBatchElements;
            int size = (i == _microBatchSize - 1)
                ? totalElements - startIdx  // Last slice gets remainder
                : microBatchElements;

            var sliceData = new T[size];
            Array.Copy(fullArray, startIdx, sliceData, 0, size);
            var sliceVector = new Vector<T>(sliceData);

            slices[i] = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(sliceVector);
        }

        return slices;
    }

    /// <summary>
    /// Slices target output into micro-batches by converting to a vector and dividing evenly.
    /// If the target cannot be sliced, all micro-batches use the same target.
    /// </summary>
    private Dictionary<int, TOutput> SliceTargetIntoMicroBatches(TOutput fullTarget)
    {
        var slices = new Dictionary<int, TOutput>();

        if (_microBatchSize <= 1)
        {
            slices[0] = fullTarget;
            return slices;
        }

        Vector<T> fullVector;
        try
        {
            fullVector = ConversionsHelper.ConvertToVector<T, TOutput>(fullTarget);
        }
        catch (InvalidOperationException ex)
        {
            throw new InvalidOperationException(
                $"Cannot slice target of type {typeof(TOutput).Name} into micro-batches. " +
                "The target must be convertible to a vector for pipeline parallel training with micro-batches > 1.",
                ex);
        }

        int totalElements = fullVector.Length;
        int microBatchElements = totalElements / _microBatchSize;

        if (microBatchElements <= 0)
        {
            throw new InvalidOperationException(
                $"Cannot slice {totalElements} target elements into {_microBatchSize} micro-batches. " +
                $"Reduce pipelineMicroBatchSize to at most {totalElements}.");
        }

        var fullArray = fullVector.ToArray();
        for (int i = 0; i < _microBatchSize; i++)
        {
            int startIdx = i * microBatchElements;
            int size = (i == _microBatchSize - 1)
                ? totalElements - startIdx
                : microBatchElements;

            var sliceData = new T[size];
            Array.Copy(fullArray, startIdx, sliceData, 0, size);
            var sliceVector = new Vector<T>(sliceData);

            // Convert back via input conversion (TOutput and TInput use the same underlying mechanism)
            slices[i] = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TOutput>(sliceVector);
        }

        return slices;
    }

    /// <summary>
    /// Gets a unique key for a (microBatchIndex, virtualStageIndex) combination.
    /// </summary>
    private int GetOperationKey(int microBatchIndex, int virtualStageIndex)
    {
        return checked(microBatchIndex * _virtualStagesPerRank + virtualStageIndex);
    }

    /// <summary>
    /// Gets the input for this stage, receiving from the previous global virtual stage.
    /// </summary>
    /// <remarks>
    /// <para>For V&gt;1 schedules (Interleaved 1F1B, Looped BFS, ZB-V), each rank holds non-contiguous
    /// model chunks. The global pipeline is: chunk 0 -> 1 -> 2 -> ... -> (V*P-1). Rank i holds chunks
    /// {i, i+P, i+2P, ...}. Each virtual stage on this rank receives input from the previous physical
    /// rank (which processed the previous global virtual stage), not from a local virtual stage.</para>
    /// <para>Special cases: global stage 0 uses original micro-batch input; single-rank with V&gt;1
    /// uses local forwarding (no inter-rank communication).</para>
    /// </remarks>
    private TInput GetStageInput(
        Dictionary<int, TInput> microBatches, int microBatchIndex, int virtualStageIndex,
        Dictionary<int, TOutput>? forwardOutputs = null)
    {
        int globalVStage = _stageId + virtualStageIndex * _numStages;

        if (globalVStage == 0)
        {
            // First global stage: use the micro-batch input directly
            if (microBatches.TryGetValue(microBatchIndex, out var microBatch))
            {
                return microBatch;
            }

            throw new InvalidOperationException(
                $"No micro-batch input found for micro-batch {microBatchIndex} at global virtual stage 0.");
        }

        if (_numStages == 1)
        {
            // Single rank with V>1: use the forward output from the previous local virtual stage
            if (forwardOutputs is not null
                && forwardOutputs.TryGetValue(GetOperationKey(microBatchIndex, virtualStageIndex - 1), out var prevOutput))
            {
                var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(prevOutput);
                return ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(outputVector);
            }

            throw new InvalidOperationException(
                $"No input available for micro-batch {microBatchIndex}, virtual stage {virtualStageIndex}. " +
                $"Forward output from virtual stage {virtualStageIndex - 1} was not found.");
        }

        // Multi-rank: receive from the rank that processed the previous global virtual stage
        int prevGlobalVStage = globalVStage - 1;
        int sourceRank = prevGlobalVStage % _numStages;
        int tag = ComputeForwardTag(microBatchIndex, prevGlobalVStage);

        Vector<T> sizeHeader = Config.CommunicationBackend.Receive(sourceRank, count: 1, tag: tag);
        int activationSize = NumOps.ToInt32(sizeHeader[0]);

        Vector<T> receivedActivations = Config.CommunicationBackend.Receive(
            sourceRank, activationSize, tag: tag);

        return ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
    }

    /// <summary>
    /// Gets the target for a specific micro-batch.
    /// </summary>
    private TOutput GetMicroBatchTarget(int microBatchIndex, Dictionary<int, TOutput> microBatchTargets, TOutput fullTarget)
    {
        if (microBatchTargets.TryGetValue(microBatchIndex, out var target))
        {
            return target;
        }
        return fullTarget;
    }

    /// <summary>
    /// Sends activations to the next stage in the global virtual pipeline.
    /// </summary>
    /// <remarks>
    /// For V&gt;1, every virtual stage (not just the last on this rank) sends to the next
    /// physical rank, because the next global virtual stage is always on rank (_stageId+1)%P.
    /// </remarks>
    private void SendActivationsForward(TOutput stageOutput, int microBatchIndex, int virtualStageIndex)
    {
        int globalVStage = _stageId + virtualStageIndex * _numStages;

        // Last global stage: nothing to send
        if (globalVStage >= _totalVirtualStages - 1)
        {
            return;
        }

        // Single rank with V>1: outputs stored locally in forwardOutputs, no communication
        if (_numStages == 1)
        {
            return;
        }

        // Send to rank responsible for next global virtual stage
        int destRank = (_stageId + 1) % _numStages;
        Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);
        int tag = ComputeForwardTag(microBatchIndex, globalVStage);

        var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
        Config.CommunicationBackend.Send(sizeHeader, destRank, tag: tag);
        Config.CommunicationBackend.Send(activationsToSend, destRank, tag: tag);
    }

    /// <summary>
    /// Computes a unique communication tag for forward pass activations.
    /// Tags are in the range [ActivationTagBase, GradientTagBase).
    /// </summary>
    /// <param name="microBatchIndex">The micro-batch index.</param>
    /// <param name="senderGlobalVStage">The sender's global virtual stage index.</param>
    private int ComputeForwardTag(int microBatchIndex, int senderGlobalVStage)
    {
        return checked(ActivationTagBase + microBatchIndex * _totalVirtualStages + senderGlobalVStage);
    }

    /// <summary>
    /// Computes a unique communication tag for backward pass gradients.
    /// Tags are in the range [GradientTagBase, PredictTagBase).
    /// </summary>
    /// <param name="microBatchIndex">The micro-batch index.</param>
    /// <param name="receiverGlobalVStage">The receiver's (upstream) global virtual stage index.</param>
    private int ComputeBackwardTag(int microBatchIndex, int receiverGlobalVStage)
    {
        return checked(GradientTagBase + microBatchIndex * _totalVirtualStages + receiverGlobalVStage);
    }

    /// <summary>
    /// Determines whether an activation should be checkpointed based on configuration.
    /// </summary>
    private bool ShouldCheckpointActivation(int opKey)
    {
        if (!_checkpointConfig.Enabled)
        {
            return false;
        }

        // MaxActivationsInMemory > 0 overrides interval-based checkpointing
        if (_checkpointConfig.MaxActivationsInMemory > 0)
        {
            return _checkpointedActivations.Count < _checkpointConfig.MaxActivationsInMemory;
        }

        // CheckpointFirstLayer: always checkpoint opKey 0 if enabled
        if (_checkpointConfig.CheckpointFirstLayer && opKey == 0)
        {
            return true;
        }

        // Interval-based checkpointing (CheckpointEveryNLayers validated >= 1 in setter)
        return _checkpointConfig.CheckpointEveryNLayers > 0
            && opKey % _checkpointConfig.CheckpointEveryNLayers == 0;
    }

    /// <summary>
    /// Retrieves the input for a micro-batch from cache, checkpoint, or recomputes it.
    /// Implements activation checkpointing recomputation when enabled.
    /// </summary>
    private TInput RetrieveMicroBatchInput(
        int opKey,
        Dictionary<int, TInput> forwardInputs,
        Dictionary<int, TInput> microBatches,
        PipelineOperation op)
    {
        // Check if input is still cached from forward pass
        if (forwardInputs.TryGetValue(opKey, out var cachedInput))
        {
            return cachedInput;
        }

        // Check activation checkpoints
        if (_checkpointConfig.Enabled && _checkpointedActivations.TryGetValue(opKey, out var checkpointedVector))
        {
            // Found a checkpoint - recompute from it if needed
            var recomputedInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(checkpointedVector);

            // If the checkpoint is for this exact operation, return directly
            return recomputedInput;
        }

        // Check if there's a nearby checkpoint to recompute from
        // NOTE: Currently unreachable because the constructor rejects RecomputeStrategy != None.
        // This is infrastructure for future recompute support (Selective/Full strategies).
        if (_checkpointConfig.Enabled && _checkpointConfig.RecomputeStrategy != RecomputeStrategy.None)
        {
            // Find the nearest earlier checkpoint within the SAME micro-batch.
            // opKey = microBatchIndex * _virtualStagesPerRank + virtualStageIndex,
            // so the current micro-batch's first key is microBatchIndex * _virtualStagesPerRank.
            int microBatchStartKey = op.MicroBatchIndex * _virtualStagesPerRank;
            int nearestCheckpointKey = -1;
            for (int searchKey = opKey - 1; searchKey >= microBatchStartKey; searchKey--)
            {
                if (_checkpointedActivations.ContainsKey(searchKey))
                {
                    nearestCheckpointKey = searchKey;
                    break;
                }
            }

            if (nearestCheckpointKey >= 0)
            {
                // Recompute forward from the nearest checkpoint to reconstruct the needed activation
                var checkpointVector = _checkpointedActivations[nearestCheckpointKey];
                var recomputeInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(checkpointVector);

                // Run forward passes from checkpoint to target, recomputing activations
                TInput currentInput = recomputeInput;
                for (int step = nearestCheckpointKey; step < opKey; step++)
                {
                    var stepOutput = WrappedModel.Predict(currentInput);
                    currentInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(
                        ConversionsHelper.ConvertToVector<T, TOutput>(stepOutput));
                }

                return currentInput;
            }
        }

        // Fallback: use the original micro-batch input
        return GetStageInput(microBatches, op.MicroBatchIndex, op.VirtualStageIndex);
    }

    /// <summary>
    /// Receives gradients from the downstream (next) stage and accumulates them.
    /// For multi-stage schedules, handles virtual stage routing.
    /// </summary>
    private void ReceiveAndAccumulateDownstreamGradients(
        Vector<T> gradientVector, int microBatchIndex, int virtualStageIndex)
    {
        int globalVStage = _stageId + virtualStageIndex * _numStages;

        // Last global stage has no downstream
        if (globalVStage >= _totalVirtualStages - 1)
        {
            return;
        }

        // Single rank: no communication
        if (_numStages == 1)
        {
            return;
        }

        // Receive from the rank that processed the next global virtual stage
        int sourceRank = (_stageId + 1) % _numStages;
        int tag = ComputeBackwardTag(microBatchIndex, globalVStage);
        Vector<T> nextStageGradients = Config.CommunicationBackend.Receive(
            sourceRank, gradientVector.Length, tag: tag);

        for (int i = 0; i < gradientVector.Length; i++)
        {
            gradientVector[i] = NumOps.Add(gradientVector[i], nextStageGradients[i]);
        }
    }

    /// <summary>
    /// Sends gradients to the upstream (previous) stage in the global virtual pipeline.
    /// </summary>
    private void SendGradientsUpstream(Vector<T> gradientVector, int microBatchIndex, int virtualStageIndex)
    {
        int globalVStage = _stageId + virtualStageIndex * _numStages;

        // First global stage has no upstream
        if (globalVStage == 0)
        {
            return;
        }

        // Single rank: no communication
        if (_numStages == 1)
        {
            return;
        }

        // Send to the rank that processed the previous global virtual stage
        int prevGlobalVStage = globalVStage - 1;
        int destRank = prevGlobalVStage % _numStages;
        int tag = ComputeBackwardTag(microBatchIndex, prevGlobalVStage);
        Config.CommunicationBackend.Send(gradientVector, destRank, tag: tag);
    }

    /// <summary>
    /// Accumulates gradients across micro-batches.
    /// </summary>
    private Vector<T> AccumulateGradients(Vector<T>? accumulated, Vector<T> newGradients)
    {
        if (accumulated is null)
        {
            // Clone to avoid mutating the original
            var copy = new T[newGradients.Length];
            for (int i = 0; i < newGradients.Length; i++)
            {
                copy[i] = newGradients[i];
            }
            return new Vector<T>(copy);
        }

        if (accumulated.Length != newGradients.Length)
        {
            throw new InvalidOperationException(
                $"Gradient length mismatch: accumulated has {accumulated.Length} elements " +
                $"but new gradients have {newGradients.Length} elements.");
        }

        for (int i = 0; i < accumulated.Length; i++)
        {
            accumulated[i] = NumOps.Add(accumulated[i], newGradients[i]);
        }

        return accumulated;
    }

    /// <summary>
    /// Frees activations after backward has consumed them to reduce memory usage.
    /// </summary>
    /// <remarks>
    /// <para>After backward completes for an operation, both the forward input and any checkpoint
    /// copy are no longer needed (with <see cref="RecomputeStrategy.None"/>). This method eagerly
    /// evicts both to minimize peak memory.</para>
    /// <para>For multi-stage schedules (V &gt; 1), forward outputs from non-last virtual stages are
    /// retained because the next virtual stage's forward pass needs them as input.
    /// These outputs are freed later when the next virtual stage's forward pass consumes them
    /// (via <see cref="ExecuteForward"/>).</para>
    /// </remarks>
    private void FreeConsumedActivations(
        int opKey, Dictionary<int, TInput> forwardInputs, Dictionary<int, TOutput> forwardOutputs)
    {
        // Backward has already consumed the input - free it unconditionally
        forwardInputs.Remove(opKey);

        // Only free forward outputs if this is the last virtual stage on this rank.
        // Non-last virtual stages' outputs are needed by the next virtual stage's
        // forward pass (e.g., Looped BFS completes vStage 0's 1F1B before starting
        // vStage 1, which needs vStage 0's forward outputs as input).
        int virtualStageIndex = opKey % _virtualStagesPerRank;
        if (virtualStageIndex >= _virtualStagesPerRank - 1)
        {
            forwardOutputs.Remove(opKey);
        }

        // With RecomputeStrategy.None, checkpointed activations are not needed after
        // backward since we cannot recompute from them. Evict to save memory.
        // Future recompute strategies (Selective/Full) would keep these as recovery points.
        if (_checkpointConfig.RecomputeStrategy == RecomputeStrategy.None)
        {
            _checkpointedActivations.Remove(opKey);
        }
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        TInput stageInput = input;

        if (_stageId > 0)
        {
            int tag = PredictTagBase;
            Vector<T> sizeHeader = Config.CommunicationBackend.Receive(_stageId - 1, count: 1, tag: tag);
            int activationSize = NumOps.ToInt32(sizeHeader[0]);

            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: tag);
            stageInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
        }

        TOutput stageOutput = WrappedModel.Predict(stageInput);

        if (_stageId < _numStages - 1)
        {
            int tag = PredictTagBase;
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);

            var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
            Config.CommunicationBackend.Send(sizeHeader, _stageId + 1, tag: tag);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: tag);
        }

        return stageOutput;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        EnsureShardingInitialized();
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "PipelineParallel");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("StageId", _stageId);
        metadata.SetProperty("NumStages", _numStages);
        metadata.SetProperty("MicroBatchSize", _microBatchSize);
        metadata.SetProperty("Schedule", _schedule.Name);
        metadata.SetProperty("VirtualStagesPerRank", _virtualStagesPerRank);
        metadata.SetProperty("EstimatedBubbleFraction", EstimatedBubbleFraction);
        metadata.SetProperty("ActivationCheckpointing", _checkpointConfig.Enabled);
        metadata.SetProperty("PartitionStrategy", _partitionStrategy?.GetType().Name ?? "Uniform");
        metadata.SetProperty("SupportsDecomposedBackward", _supportsDecomposedBackward);
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new PipelineParallelModel<T, TInput, TOutput>(
            WrappedModel.WithParameters(parameters), Config, _microBatchSize,
            _partitionStrategy, _schedule, _checkpointConfig);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        EnsureShardingInitialized();
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(_microBatchSize);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);
        writer.Write(_schedule.Name);
        writer.Write(_checkpointConfig.Enabled);
        writer.Write(_checkpointConfig.CheckpointEveryNLayers);
        writer.Write(_virtualStagesPerRank);
        var modelData = WrappedModel.Serialize();
        writer.Write(modelData.Length);
        writer.Write(modelData);
        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var reader = new BinaryReader(ms);
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        int savedMicroBatchSize = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression
        reader.ReadString(); // Schedule name (informational)
        reader.ReadBoolean(); // Checkpointing enabled
        reader.ReadInt32(); // CheckpointEveryNLayers
        reader.ReadInt32(); // VirtualStagesPerRank (informational)

        if (savedWorldSize != WorldSize)
            throw new InvalidOperationException($"World size mismatch: {savedWorldSize} vs {WorldSize}");
        if (savedRank != Rank)
            throw new InvalidOperationException($"Rank mismatch: {savedRank} vs {Rank}");
        if (savedMicroBatchSize != _microBatchSize)
            throw new InvalidOperationException($"Micro batch size mismatch: saved model was trained with {savedMicroBatchSize}, but current instance configured with {_microBatchSize}");

        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);
        WrappedModel.Deserialize(modelData);

        // EnsureShardingInitialized calls OnBeforeInitializeSharding (which sets _numStages
        // and other derived state) before InitializeSharding. Calling InitializeSharding
        // directly would skip that setup and cause divide-by-zero.
        EnsureShardingInitialized();
    }

    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
        Config.CommunicationBackend.Barrier();
        try
        {
            if (Rank == 0)
                File.WriteAllBytes(filePath, Serialize());
        }
        finally
        {
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
    {
        Config.CommunicationBackend.Barrier();
        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        finally
        {
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> Clone()
    {
        return new PipelineParallelModel<T, TInput, TOutput>(
            WrappedModel.Clone(), Config, _microBatchSize,
            _partitionStrategy, _schedule, _checkpointConfig);
    }
}
