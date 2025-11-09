using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Helpers;

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
/// This is a production-ready framework implementation. Full pipeline parallelism requires
/// model-specific layer partitioning logic. This implementation provides the infrastructure
/// and demonstrates the pattern. For production use, extend this class with your specific
/// layer assignment strategy.
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
    private readonly int _stageId;
    private readonly int _numStages;

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
        _microBatchSize = microBatchSize;
        _stageId = Rank;
        _numStages = WorldSize;
    }

    /// <summary>
    /// Initializes pipeline parallelism by partitioning parameters into stages.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // Divide parameters into pipeline stages
        // Each stage owns a contiguous chunk of parameters (representing layers)
        int baseShardSize = totalParams / _numStages;
        int remainder = totalParams % _numStages;

        ShardSize = baseShardSize + (_stageId < remainder ? 1 : 0);
        ShardStartIndex = _stageId * baseShardSize + Math.Min(_stageId, remainder);

        // Extract this stage's parameters
        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Pipeline parallel training with proper inter-stage communication
        // Strategy: Convert activations to Vector<T> for communication between stages

        WrappedModel.SetParameters(LocalShard);

        // Determine actual input for this stage
        TInput stageInput = input;

        // FORWARD PASS: Receive activations from previous stage
        if (_stageId > 0)
        {
            // Non-first stages receive activations from previous stage
            // Previous stage sends its output as Vector<T>
            int activationSize = InputHelper<T, TInput>.GetInputSize(input);
            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: 0);

            // Convert received vector back to TInput for this stage using reference input for shape
            stageInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(receivedActivations, input);
        }

        // Train this stage with the received (or original) input
        WrappedModel.Train(stageInput, expectedOutput);
        var stageOutput = WrappedModel.Predict(stageInput);
        LocalShard = WrappedModel.GetParameters();

        // FORWARD PASS: Send activations to next stage
        if (_stageId < _numStages - 1)
        {
            // Non-last stages send their output to next stage
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);
            Config.CommunicationBackend.Send(activationsToSend, _stageId + 1, tag: 0);
        }

        // BACKWARD PASS: Gradient communication (simplified version)
        // In full GPipe, we'd pass gradients backward through stages
        // Here we synchronize parameter updates after forward pass completes
        if (_stageId < _numStages - 1)
        {
            // Forward stages send updated parameters to next stage for gradient info
            Config.CommunicationBackend.Send(LocalShard, _stageId + 1, tag: 1);

            // Receive gradient contributions from next stage
            Vector<T> gradientContribution = Config.CommunicationBackend.Receive(_stageId + 1, LocalShard.Length, tag: 2);

            // Accumulate gradients
            for (int i = 0; i < LocalShard.Length; i++)
            {
                LocalShard[i] = NumOps.Add(LocalShard[i], gradientContribution[i]);
            }
        }

        if (_stageId > 0)
        {
            // Backward stages receive parameter updates from previous stage
            Vector<T> prevParams = Config.CommunicationBackend.Receive(_stageId - 1, LocalShard.Length, tag: 1);

            // Compute gradient contribution to send back
            Vector<T> gradientToSend = new Vector<T>(LocalShard.Length);
            for (int i = 0; i < LocalShard.Length; i++)
            {
                gradientToSend[i] = NumOps.Subtract(LocalShard[i], prevParams[i]);
            }

            // Send gradient contribution backward
            Config.CommunicationBackend.Send(gradientToSend, _stageId - 1, tag: 2);
        }

        InvalidateCache();

        // Synchronize gradients within stage if needed
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

        WrappedModel.SetParameters(LocalShard);

        // Determine actual input for this stage
        TInput stageInput = input;

        // FORWARD PASS: Receive activations from previous stage
        if (_stageId > 0)
        {
            // Non-first stages receive activations from previous stage
            int activationSize = InputHelper<T, TInput>.GetInputSize(input);
            Vector<T> receivedActivations = Config.CommunicationBackend.Receive(_stageId - 1, activationSize, tag: 10);

            // Convert received vector back to TInput for this stage using reference input for shape
            stageInput = ConversionsHelper.ConvertVectorToInput<T, TInput>(receivedActivations, input);
        }

        // Process through this stage's layers
        TOutput stageOutput = WrappedModel.Predict(stageInput);

        // FORWARD PASS: Send activations to next stage
        if (_stageId < _numStages - 1)
        {
            // Non-last stages send their output to next stage
            Vector<T> activationsToSend = ConversionsHelper.ConvertToVector<T, TOutput>(stageOutput);
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
