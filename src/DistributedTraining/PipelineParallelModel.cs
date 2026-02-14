using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;


namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Pipeline Parallel model wrapper - splits model into stages across ranks.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Pipeline Parallelism (GPipe-style) divides the model vertically into stages, with each process
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
/// <para><b>Use Cases:</b>
/// - Very deep models that don't fit on a single GPU
/// - When model depth (layers) >> width (parameters per layer)
/// - Transformer models with many layers
/// - Complementary to data parallelism (can combine them)
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent for deep models - each rank stores only its layers
/// - Communication: Low - only activations passed between adjacent stages
/// - Complexity: High - requires micro-batching, careful scheduling, pipeline bubble overhead
/// - Best for: Very deep models, limited per-device memory
/// - Limitation: Pipeline "bubble" (idle time) reduces efficiency
/// </para>
/// <para><b>Production Optimizations (Issue #463):</b>
/// This implementation supports three production optimizations:
///
/// 1. <b>Custom Partition Strategies</b>: Balance compute load across stages using
///    <see cref="IPipelinePartitionStrategy{T}"/> (default: uniform).
///
/// 2. <b>Pipeline Schedules</b>: Choose between GPipe (simple) and 1F1B (efficient)
///    via <see cref="IPipelineSchedule"/> to reduce pipeline bubble overhead.
///
/// 3. <b>Activation Checkpointing</b>: Trade compute for memory via
///    <see cref="ActivationCheckpointConfig"/> to train deeper models.
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new DeepNeuralNetwork&lt;double&gt;(...); // 100 layers
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// // Basic usage (uniform partition, GPipe schedule)
/// var pipelineModel = new PipelineParallelModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config, microBatchSize: 4);
///
/// // Advanced usage (load-balanced partition, 1F1B schedule, checkpointing)
/// var pipelineModel = new PipelineParallelModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config, microBatchSize: 8,
///     partitionStrategy: new LoadBalancedPartitionStrategy&lt;double&gt;(estimatedLayerSize: 1024),
///     schedule: new OneForwardOneBackwardSchedule(),
///     checkpointConfig: new ActivationCheckpointConfig { Enabled = true, CheckpointEveryNLayers = 10 });
/// </code>
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

    // Activation storage for checkpointing
    private readonly Dictionary<int, Vector<T>> _checkpointedActivations = new();

    /// <summary>
    /// Gets the pipeline schedule used by this model.
    /// </summary>
    public IPipelineSchedule Schedule => _schedule;

    /// <summary>
    /// Gets the activation checkpoint configuration.
    /// </summary>
    public ActivationCheckpointConfig CheckpointConfig => _checkpointConfig;

    /// <summary>
    /// Gets the partition strategy, or null if using uniform partitioning.
    /// </summary>
    public IPipelinePartitionStrategy<T>? PartitionStrategy => _partitionStrategy;

    /// <summary>
    /// Gets the estimated pipeline bubble fraction for the current configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the percentage of time that stages are idle.
    /// Lower is better. Values closer to 0.0 mean the pipeline is being used efficiently.</para>
    /// </remarks>
    public double EstimatedBubbleFraction => _schedule.EstimateBubbleFraction(_numStages, _microBatchSize);

    /// <summary>
    /// Creates a new Pipeline Parallel model.
    /// </summary>
    /// <param name="wrappedModel">The model to split into pipeline stages.</param>
    /// <param name="config">Configuration for sharding and communication.</param>
    /// <param name="microBatchSize">Size of micro-batches for pipeline execution (default: 1).</param>
    /// <param name="partitionStrategy">
    /// Strategy for partitioning parameters across stages. If null, uses uniform partitioning.
    /// <para><b>For Beginners:</b> This decides how to split the model across devices.
    /// The default splits evenly, but you can use <see cref="LoadBalancedPartitionStrategy{T}"/>
    /// to balance computational load.</para>
    /// </param>
    /// <param name="schedule">
    /// Pipeline execution schedule. If null, uses <see cref="GPipeSchedule"/>.
    /// <para><b>For Beginners:</b> This decides the order of forward/backward passes.
    /// Use <see cref="OneForwardOneBackwardSchedule"/> for better efficiency.</para>
    /// </param>
    /// <param name="checkpointConfig">
    /// Activation checkpointing configuration. If null, checkpointing is disabled.
    /// <para><b>For Beginners:</b> Enable this to reduce memory usage at the cost of
    /// additional computation during the backward pass.</para>
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
    }

    /// <summary>
    /// Called before InitializeSharding to set up derived class state.
    /// </summary>
    protected override void OnBeforeInitializeSharding()
    {
        _stageId = Config.CommunicationBackend.Rank;
        _numStages = Config.CommunicationBackend.WorldSize;
    }

    /// <summary>
    /// Initializes pipeline parallelism by partitioning parameters into stages.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        if (_partitionStrategy is not null)
        {
            // Use custom partition strategy
            var partitions = _partitionStrategy.ComputePartition(totalParams, _numStages);
            ShardStartIndex = partitions[_stageId].StartIndex;
            ShardSize = partitions[_stageId].Size;
        }
        else
        {
            // Default: uniform partitioning
            int baseShardSize = totalParams / _numStages;
            int remainder = totalParams % _numStages;

            ShardSize = baseShardSize + (_stageId < remainder ? 1 : 0);
            ShardStartIndex = _stageId * baseShardSize + Math.Min(_stageId, remainder);
        }

        // Extract this stage's parameters
        if (ShardSize > 0)
        {
            var shardData = new T[ShardSize];
            Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
            LocalShard = new Vector<T>(shardData);
        }
        else
        {
            LocalShard = new Vector<T>(0);
        }

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Pipeline parallel training using the configured schedule
        var scheduleOps = _schedule.GetSchedule(_stageId, _numStages, _microBatchSize);

        // Gather full parameters before training
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Save parameters BEFORE training to compute gradients
        var parametersBefore = new Vector<T>(fullParams.ToArray());

        // Accumulated gradients across all micro-batches
        Vector<T>? accumulatedGradients = null;

        // Track activations per micro-batch for backward pass
        var microBatchInputs = new Dictionary<int, TInput>();
        var microBatchOutputs = new Dictionary<int, TOutput>();

        // Clear checkpointed activations from previous iteration
        _checkpointedActivations.Clear();

        foreach (var op in scheduleOps)
        {
            if (op.Type == PipelineOperationType.Forward)
            {
                var stageInput = GetStageInput(input, op.MicroBatchIndex);

                // Store input for backward pass (with checkpointing awareness)
                if (ShouldCheckpointActivation(op.MicroBatchIndex))
                {
                    var inputVector = ConversionsHelper.ConvertToVector<T, TInput>(stageInput);
                    _checkpointedActivations[op.MicroBatchIndex] = inputVector;
                }

                microBatchInputs[op.MicroBatchIndex] = stageInput;

                // Predict stage output
                var stageOutput = WrappedModel.Predict(stageInput);
                microBatchOutputs[op.MicroBatchIndex] = stageOutput;

                // Send activations to next stage
                SendActivationsForward(stageOutput, tag: op.MicroBatchIndex * 10);
            }
            else // Backward
            {
                // Get the input for this micro-batch (from cache or recompute from checkpoint)
                TInput microBatchInput;
                if (microBatchInputs.TryGetValue(op.MicroBatchIndex, out var cachedInput))
                {
                    microBatchInput = cachedInput;
                }
                else if (_checkpointConfig.Enabled && _checkpointedActivations.TryGetValue(op.MicroBatchIndex, out var checkpointedVector))
                {
                    microBatchInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(checkpointedVector);
                }
                else
                {
                    microBatchInput = GetStageInput(input, op.MicroBatchIndex);
                }

                // Compute gradients for this micro-batch
                var gradientVector = WrappedModel.ComputeGradients(microBatchInput, expectedOutput);

                // Receive and accumulate gradients from next stage
                if (_stageId < _numStages - 1)
                {
                    Vector<T> nextStageGradients = Config.CommunicationBackend.Receive(
                        _stageId + 1, gradientVector.Length, tag: 1000 + op.MicroBatchIndex);

                    for (int i = 0; i < gradientVector.Length; i++)
                    {
                        gradientVector[i] = NumOps.Add(gradientVector[i], nextStageGradients[i]);
                    }
                }

                // Send gradients to previous stage
                if (_stageId > 0)
                {
                    Config.CommunicationBackend.Send(gradientVector, _stageId - 1, tag: 1000 + op.MicroBatchIndex);
                }

                // Accumulate gradients across micro-batches
                if (accumulatedGradients is null)
                {
                    accumulatedGradients = gradientVector;
                }
                else
                {
                    for (int i = 0; i < accumulatedGradients.Length; i++)
                    {
                        accumulatedGradients[i] = NumOps.Add(accumulatedGradients[i], gradientVector[i]);
                    }
                }

                // Free non-checkpointed activations to save memory
                if (!ShouldCheckpointActivation(op.MicroBatchIndex))
                {
                    microBatchInputs.Remove(op.MicroBatchIndex);
                    microBatchOutputs.Remove(op.MicroBatchIndex);
                }
            }
        }

        // Apply accumulated gradients
        if (accumulatedGradients is not null)
        {
            // Average gradients across micro-batches
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
        UpdateLocalShardFromFull(updatedParams);
        InvalidateCache();

        // Clean up activation storage
        _checkpointedActivations.Clear();

        // Synchronize parameters across stages for consistency
        if (Config.AutoSyncGradients)
        {
            SynchronizeGradients();
        }
    }

    /// <summary>
    /// Gets the input for this stage, receiving from previous stage if needed.
    /// </summary>
    private TInput GetStageInput(TInput originalInput, int microBatchIndex)
    {
        if (_stageId > 0)
        {
            // Receive activations from previous stage
            Vector<T> sizeHeader = Config.CommunicationBackend.Receive(
                _stageId - 1, count: 1, tag: microBatchIndex * 10);
            int activationSize = NumOps.ToInt32(sizeHeader[0]);

            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(
                _stageId - 1, activationSize, tag: microBatchIndex * 10);

            return ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
        }

        return originalInput;
    }

    /// <summary>
    /// Sends activations to the next stage in the pipeline.
    /// </summary>
    private void SendActivationsForward(TOutput stageOutput, int tag)
    {
        if (_stageId < _numStages - 1)
        {
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);

            var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
            Config.CommunicationBackend.Send(sizeHeader, _stageId + 1, tag: tag);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: tag);
        }
    }

    /// <summary>
    /// Determines whether an activation for the given micro-batch should be checkpointed.
    /// </summary>
    private bool ShouldCheckpointActivation(int microBatchIndex)
    {
        if (!_checkpointConfig.Enabled)
        {
            return false;
        }

        if (_checkpointConfig.MaxActivationsInMemory > 0)
        {
            // Limit-based checkpointing: keep the most recent N activations
            return _checkpointedActivations.Count < _checkpointConfig.MaxActivationsInMemory;
        }

        // Interval-based checkpointing
        return microBatchIndex % _checkpointConfig.CheckpointEveryNLayers == 0;
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        // Pipeline forward pass for inference
        // Activations flow through stages sequentially

        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Determine actual input for this stage
        TInput stageInput = input;

        // FORWARD PASS: Receive activations from previous stage
        if (_stageId > 0)
        {
            Vector<T> sizeHeader = Config.CommunicationBackend.Receive(_stageId - 1, count: 1, tag: 10);
            int activationSize = NumOps.ToInt32(sizeHeader[0]);

            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: 10);
            stageInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
        }

        // Process through this stage's layers
        TOutput stageOutput = WrappedModel.Predict(stageInput);

        // FORWARD PASS: Send activations to next stage
        if (_stageId < _numStages - 1)
        {
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);

            var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
            Config.CommunicationBackend.Send(sizeHeader, _stageId + 1, tag: 10);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: 10);

            return stageOutput;
        }

        // Last stage returns the final prediction
        return stageOutput;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "PipelineParallel");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("StageId", _stageId);
        metadata.SetProperty("NumStages", _numStages);
        metadata.SetProperty("MicroBatchSize", _microBatchSize);
        metadata.SetProperty("Schedule", _schedule.Name);
        metadata.SetProperty("EstimatedBubbleFraction", EstimatedBubbleFraction);
        metadata.SetProperty("ActivationCheckpointing", _checkpointConfig.Enabled);
        metadata.SetProperty("PartitionStrategy", _partitionStrategy?.GetType().Name ?? "Uniform");
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

        if (savedWorldSize != WorldSize)
            throw new InvalidOperationException($"World size mismatch: {savedWorldSize} vs {WorldSize}");
        if (savedRank != Rank)
            throw new InvalidOperationException($"Rank mismatch: {savedRank} vs {Rank}");
        if (savedMicroBatchSize != _microBatchSize)
            throw new InvalidOperationException($"Micro batch size mismatch: saved model was trained with {savedMicroBatchSize}, but current instance configured with {_microBatchSize}");

        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);
        WrappedModel.Deserialize(modelData);
        InitializeSharding();
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
