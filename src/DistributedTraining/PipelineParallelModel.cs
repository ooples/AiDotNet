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
/// - Limitation: Pipeline "bubble" (idle time) reduces efficiency, typically ~12-25% for GPipe
/// </para>
/// <para><b>Implementation Note:</b>
/// This implementation provides GPipe-style pipeline parallelism with gradient-based backward pass.
/// The forward pass sends activations between adjacent stages, and the backward pass communicates
/// gradients in the reverse direction. Gradients are accumulated across stages and applied to
/// parameters after the backward pass completes.
///
/// Gradient Approximation: Since IFullModel.Train() combines gradient computation and parameter
/// updates into a single operation, gradients are approximated as parameter differences
/// (params_before - params_after). This captures the complete parameter update including learning
/// rate and optimizer state. For access to raw gradients before optimizer application, extend
/// this class or use an optimizer that exposes gradients via IGradientBasedOptimizer.
///
/// For production use with specific models, consider:
/// 1. Model-specific layer partitioning strategies (e.g., balance compute load across stages)
/// 2. Micro-batch scheduling to reduce pipeline bubbles
/// 3. Activation checkpointing to reduce memory usage
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new DeepNeuralNetwork&lt;double&gt;(...); // 100 layers
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// // Rank 0: layers 0-24, Rank 1: layers 25-49, Rank 2: layers 50-74, Rank 3: layers 75-99
/// var pipelineModel = new PipelineParallelModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config, microBatchSize: 4);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class PipelineParallelModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private readonly int _microBatchSize;
    private int _stageId;
    private int _numStages;

    /// <summary>
    /// Creates a new Pipeline Parallel model.
    /// </summary>
    /// <param name="wrappedModel">The model to split into pipeline stages</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="microBatchSize">Size of micro-batches for pipeline execution (default: 1)</param>
    public PipelineParallelModel(
        IFullModel<T, TInput, TOutput> wrappedModel,
        IShardingConfiguration<T> config,
        int microBatchSize = 1)
        : base(wrappedModel, config)
    {
        if (microBatchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(microBatchSize),
                "Micro batch size must be at least 1.");
        }

        _microBatchSize = microBatchSize;
        // Note: _stageId and _numStages are set in OnBeforeInitializeSharding which is called by lazy initialization
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

        // Try layer-aware partitioning first
        if (WrappedModel is ILayeredModel<T> layeredModel && layeredModel.LayerCount > 0)
        {
            InitializeLayerAwareSharding(layeredModel, fullParameters);
        }
        else
        {
            // Fallback: divide parameters into pipeline stages by count
            int baseShardSize = totalParams / _numStages;
            int remainder = totalParams % _numStages;

            ShardSize = baseShardSize + (_stageId < remainder ? 1 : 0);
            ShardStartIndex = _stageId * baseShardSize + Math.Min(_stageId, remainder);

            var shardData = new T[ShardSize];
            Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
            LocalShard = new Vector<T>(shardData);
        }

        CachedFullParameters = null;
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

        if (layerCount == 0 || _numStages <= 1)
        {
            // Single stage or no layers: take all parameters
            ShardStartIndex = 0;
            ShardSize = fullParameters.Length;
            LocalShard = new Vector<T>(fullParameters.ToArray());
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
                    break;
                }
            }

            // Ensure at least one layer per stage
            if (layersInStage == 0 && currentLayer < layerCount)
            {
                currentLayer++;
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
        int stageParamEnd = allLayerInfo[lastLayerInStage].ParameterOffset +
                            allLayerInfo[lastLayerInStage].ParameterCount;

        ShardStartIndex = stageParamStart;
        ShardSize = stageParamEnd - stageParamStart;

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
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // GPipe-style pipeline parallel training with gradient-based backward pass
        // Strategy: Forward pass sends activations, backward pass sends gradients

        // Gather full parameters before training
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Save parameters BEFORE training to compute gradients
        var parametersBefore = new Vector<T>(fullParams.ToArray());

        // Determine actual input for this stage
        TInput stageInput = input;

        // FORWARD PASS: Receive activations from previous stage
        if (_stageId > 0)
        {
            // Protocol: First receive 1-element size header, then receive activations
            // This prevents size mismatches when stage output size differs from input size
            Vector<T> sizeHeader = Config.CommunicationBackend.Receive(_stageId - 1, count: 1, tag: 0);
            int activationSize = NumOps.ToInt32(sizeHeader[0]);

            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: 0);

            // For intermediate stages, convert received activations to TInput type WITHOUT using
            // the original input as reference (which would have the wrong shape for non-first stages).
            // Use ConversionsHelper to centralize conversion logic and avoid code duplication.
            stageInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
        }

        // Compute true gradients using the model's gradient computation
        // This provides accurate gradients before optimizer updates are applied
        var gradientVector = WrappedModel.ComputeGradients(stageInput, expectedOutput);

        // Predict stage output for forward pass communication
        var stageOutput = WrappedModel.Predict(stageInput);

        // FORWARD PASS: Send activations to next stage
        if (_stageId < _numStages - 1)
        {
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);

            // Protocol: First send 1-element size header, then send activations
            // This allows receiver to know the exact size of incoming activations
            var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
            Config.CommunicationBackend.Send(sizeHeader, _stageId + 1, tag: 0);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: 0);
        }

        // BACKWARD PASS: Gradient communication
        // Gradients flow backward through the pipeline (opposite direction of activations)
        if (_stageId < _numStages - 1)
        {
            // Non-last stages receive gradient contributions from next stage
            Vector<T> nextStageGradients = Config.CommunicationBackend.Receive(_stageId + 1, gradientVector.Length, tag: 1);

            // Accumulate gradients: local gradients + gradients from downstream stages
            for (int i = 0; i < gradientVector.Length; i++)
            {
                gradientVector[i] = NumOps.Add(gradientVector[i], nextStageGradients[i]);
            }
        }

        if (_stageId > 0)
        {
            // Non-first stages send accumulated gradients to previous stage
            Config.CommunicationBackend.Send(gradientVector, _stageId - 1, tag: 1);
        }

        // Apply accumulated gradients to parameters using the configured learning rate
        // In pipeline parallelism, we use a simple SGD-style update: θ = θ - lr * gradients
        // For more sophisticated optimization, wrap this model with a gradient-based optimizer
        WrappedModel.SetParameters(parametersBefore);
        WrappedModel.ApplyGradients(gradientVector, Config.LearningRate);

        // Extract this stage's parameter shard
        var updatedParams = WrappedModel.GetParameters();
        UpdateLocalShardFromFull(updatedParams);
        InvalidateCache();

        // Synchronize parameters across stages for consistency
        if (Config.AutoSyncGradients)
        {
            SynchronizeGradients();
        }
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
            // Protocol: First receive 1-element size header, then receive activations
            // This prevents size mismatches when stage output size differs from input size
            Vector<T> sizeHeader = Config.CommunicationBackend.Receive(_stageId - 1, count: 1, tag: 10);
            int activationSize = NumOps.ToInt32(sizeHeader[0]);

            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: 10);

            // For intermediate stages, convert received activations to TInput type WITHOUT using
            // the original input as reference (which would have the wrong shape for non-first stages).
            // Use ConversionsHelper to centralize conversion logic and avoid code duplication.
            stageInput = ConversionsHelper.ConvertVectorToInputWithoutReference<T, TInput>(receivedActivations);
        }

        // Process through this stage's layers
        TOutput stageOutput = WrappedModel.Predict(stageInput);

        // FORWARD PASS: Send activations to next stage
        if (_stageId < _numStages - 1)
        {
            // Non-last stages send their output to next stage
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);

            // Protocol: First send 1-element size header, then send activations
            // This allows receiver to know the exact size of incoming activations
            var sizeHeader = new Vector<T>(new[] { NumOps.FromDouble(activationsToSend.Length) });
            Config.CommunicationBackend.Send(sizeHeader, _stageId + 1, tag: 10);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: 10);

            // Intermediate stages must still return a value
            // Return the stage output (caller should only use output from last stage)
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
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new PipelineParallelModel<T, TInput, TOutput>(
            WrappedModel.WithParameters(parameters), Config, _microBatchSize);
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
        reader.ReadBoolean();
        reader.ReadInt32();
        reader.ReadBoolean();

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
        return new PipelineParallelModel<T, TInput, TOutput>(WrappedModel.Clone(), Config, _microBatchSize);
    }
}
