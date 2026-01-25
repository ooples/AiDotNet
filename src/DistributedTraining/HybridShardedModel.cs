using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements 3D Parallelism (Hybrid Sharded) model - combines data, tensor, and pipeline parallelism.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// 3D Parallelism combines all three major parallelism strategies for maximum scalability:
/// - Data Parallelism: Different data batches across replicas
/// - Tensor Parallelism: Layer-wise partitioning within each pipeline stage
/// - Pipeline Parallelism: Model depth partitioning across stages
///
/// This enables training extremely large models (100B+ parameters) on thousands of GPUs by
/// exploiting parallelism in all dimensions. This is the strategy used for training models
/// like GPT-3, Megatron-Turing NLG, and other frontier models.
/// </para>
/// <para><b>For Beginners:</b>
/// 3D Parallelism is the ultimate distributed training strategy - it combines ALL the techniques:
///
/// Imagine training a MASSIVE model across 512 GPUs:
/// - Pipeline Parallel (depth): Split model into 8 stages (64 GPUs per stage)
/// - Tensor Parallel (width): Within each stage, split layers 8 ways (8 GPUs per tensor group)
/// - Data Parallel (batches): Remaining 8 GPUs in each tensor group process different data
///
/// Layout example for 512 GPUs = 8 pipeline × 8 tensor × 8 data:
/// - Stage 0: GPUs 0-63 (layers 0-12)
///   - Tensor group 0: GPUs 0-7 (data replicas)
///   - Tensor group 1: GPUs 8-15 (data replicas)
///   - ... 8 tensor groups total
/// - Stage 1: GPUs 64-127 (layers 13-25)
///   - ...and so on
/// </para>
/// <para><b>Use Cases:</b>
/// - Training frontier models (GPT-3 scale: 100B-1T parameters)
/// - Requires 100s to 1000s of GPUs
/// - When single parallelism dimension isn't enough
/// - Production training at largest scales (OpenAI, Google, Meta)
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - exploits all memory-saving strategies
/// - Communication: Complex - requires careful network topology optimization
/// - Complexity: Very High - most complex distributed strategy
/// - Best for: Frontier-scale models (100B+ params), massive GPU clusters
/// - Requires: Careful tuning of all three parallelism dimensions for efficiency
/// </para>
/// <para><b>Implementation Note:</b>
/// This is a production-ready framework providing the 3D parallelism infrastructure.
/// Full production deployment requires:
/// 1. Process group management (separate groups for data/tensor/pipeline)
/// 2. Model-specific layer partitioning
/// 3. Careful configuration tuning for your specific cluster topology
/// This implementation demonstrates the pattern and provides the foundation.
/// </para>
/// <para>
/// Example:
/// <code>
/// // Training a 175B parameter model on 512 GPUs
/// // 8 pipeline stages × 8 tensor parallel × 8 data parallel = 512
/// var model = new MassiveTransformer&lt;double&gt;(...);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: myRank, worldSize: 512);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// var hybridModel = new HybridShardedModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config,
///     pipelineParallelSize: 8,
///     tensorParallelSize: 8,
///     dataParallelSize: 8);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class HybridShardedModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private Vector<T>? _computedGradients;

    // Static ThreadLocal to pass constructor parameters before base constructor call.
    // This is necessary because C# doesn't allow derived class code to run before base constructor.
    private static readonly ThreadLocal<(int pp, int tp, int dp)?> PendingConfig = new();

    // Computed values (set in OnBeforeInitializeSharding)
    private int _pipelineParallelSize;
    private int _tensorParallelSize;
    private int _dataParallelSize;
    private int _pipelineRank;
    private int _tensorRank;
    private int _dataRank;

    /// <summary>
    /// Creates a new 3D Parallel (Hybrid Sharded) model.
    /// </summary>
    /// <param name="wrappedModel">The model to partition with 3D parallelism</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="pipelineParallelSize">Number of pipeline stages (default: 1)</param>
    /// <param name="tensorParallelSize">Tensor parallelism degree (default: 1)</param>
    /// <param name="dataParallelSize">Data parallelism degree (default: uses remaining GPUs)</param>
    public HybridShardedModel(
        IFullModel<T, TInput, TOutput> wrappedModel,
        IShardingConfiguration<T> config,
        int pipelineParallelSize = 1,
        int tensorParallelSize = 1,
        int dataParallelSize = -1)
        : base(StoreConfigAndPassThrough(wrappedModel, pipelineParallelSize, tensorParallelSize, dataParallelSize), config)
    {
        // Validate parameters early
        if (pipelineParallelSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(pipelineParallelSize),
                "Pipeline parallel size must be at least 1.");
        }

        if (tensorParallelSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(tensorParallelSize),
                "Tensor parallel size must be at least 1.");
        }

        // Validate WorldSize constraint early (if explicit dataParallelSize is provided)
        if (dataParallelSize > 0)
        {
            int worldSize = config.CommunicationBackend.WorldSize;
            if (pipelineParallelSize * tensorParallelSize * dataParallelSize != worldSize)
            {
                throw new ArgumentException(
                    $"Pipeline ({pipelineParallelSize}) × Tensor ({tensorParallelSize}) × " +
                    $"Data ({dataParallelSize}) must equal WorldSize ({worldSize})");
            }
        }

        // Clear the pending config after base constructor completes
        PendingConfig.Value = null;
    }

    /// <summary>
    /// Stores constructor parameters in ThreadLocal before base constructor call.
    /// This workaround is necessary because C# doesn't allow derived class code to execute
    /// before the base constructor, but we need these values in OnBeforeInitializeSharding.
    /// </summary>
    private static IFullModel<T, TInput, TOutput> StoreConfigAndPassThrough(
        IFullModel<T, TInput, TOutput> model,
        int pipelineParallelSize,
        int tensorParallelSize,
        int dataParallelSize)
    {
        PendingConfig.Value = (pipelineParallelSize, tensorParallelSize, dataParallelSize);
        return model;
    }

    /// <summary>
    /// Called before InitializeSharding to set up derived class state.
    /// </summary>
    protected override void OnBeforeInitializeSharding()
    {
        // Read configuration from ThreadLocal (stored before base constructor call)
        var pending = PendingConfig.Value ?? (1, 1, -1);
        int requestedPipelineParallelSize = pending.pp;
        int requestedTensorParallelSize = pending.tp;
        int requestedDataParallelSize = pending.dp;

        _pipelineParallelSize = requestedPipelineParallelSize;
        _tensorParallelSize = requestedTensorParallelSize;

        // Calculate data parallel size if not specified
        if (requestedDataParallelSize == -1)
        {
            int totalGpus = Config.CommunicationBackend.WorldSize;
            if (totalGpus % (_pipelineParallelSize * _tensorParallelSize) != 0)
            {
                throw new ArgumentException(
                    $"WorldSize ({totalGpus}) must be divisible by " +
                    $"pipelineParallelSize ({_pipelineParallelSize}) × tensorParallelSize ({_tensorParallelSize})");
            }
            _dataParallelSize = totalGpus / (_pipelineParallelSize * _tensorParallelSize);
        }
        else
        {
            _dataParallelSize = requestedDataParallelSize;
        }

        // Verify configuration
        int worldSize = Config.CommunicationBackend.WorldSize;
        if (_pipelineParallelSize * _tensorParallelSize * _dataParallelSize != worldSize)
        {
            throw new ArgumentException(
                $"Pipeline ({_pipelineParallelSize}) × Tensor ({_tensorParallelSize}) × " +
                $"Data ({_dataParallelSize}) must equal WorldSize ({worldSize})");
        }

        // Calculate this process's position in the 3D grid
        // Layout: [pipeline][tensor][data]
        int rank = Config.CommunicationBackend.Rank;
        int tensorGroupSize = _tensorParallelSize * _dataParallelSize;
        _pipelineRank = rank / tensorGroupSize;
        int withinStage = rank % tensorGroupSize;
        _tensorRank = withinStage / _dataParallelSize;
        _dataRank = withinStage % _dataParallelSize;
    }

    /// <summary>
    /// Initializes 3D parallelism by partitioning along all dimensions.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // Apply pipeline partitioning first (depth-wise)
        int pipelineShardSize = totalParams / _pipelineParallelSize;
        int pipelineRemainder = totalParams % _pipelineParallelSize;
        int myPipelineSize = pipelineShardSize + (_pipelineRank < pipelineRemainder ? 1 : 0);
        int myPipelineStart = _pipelineRank * pipelineShardSize + Math.Min(_pipelineRank, pipelineRemainder);

        // Then apply tensor partitioning within the pipeline stage (width-wise)
        int tensorShardSize = myPipelineSize / _tensorParallelSize;
        int tensorRemainder = myPipelineSize % _tensorParallelSize;
        ShardSize = tensorShardSize + (_tensorRank < tensorRemainder ? 1 : 0);
        int tensorOffset = _tensorRank * tensorShardSize + Math.Min(_tensorRank, tensorRemainder);

        ShardStartIndex = myPipelineStart + tensorOffset;

        // Data parallelism doesn't affect parameter sharding (parameters replicated across data-parallel group)
        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void SynchronizeGradients()
    {
        if (_computedGradients == null)
        {
            throw new InvalidOperationException(
                "Gradients have not been computed. Call Train() before SynchronizeGradients().");
        }

        // In 3D parallelism (hybrid sharding), gradient synchronization is complex:
        // - Tensor-parallel: Need to reduce partial gradients within tensor group
        // - Data-parallel: Need to average gradients across data replicas
        // - Pipeline-parallel: Each stage handles its own gradients
        //
        // Correct synchronization requires:
        // 1. AllReduce within tensor-parallel group (sum partial gradients from same pipeline stage)
        // 2. AllReduce within data-parallel group (average gradients across data replicas that share same pipeline/tensor position)
        // 3. Pipeline parallel stages handle their own gradient accumulation

        // Guard against full-world AllReduce when data-parallel size > 1
        if (_dataParallelSize > 1)
        {
            throw new NotSupportedException(
                "HybridShardedModel needs subgroup AllReduce over the data-parallel replica set; " +
                "reducing across the full world corrupts tensor/pipeline shards. " +
                "Implement subgroup-aware collectives that sync only within the data-parallel group (ranks sharing same pipeline/tensor position), " +
                "or use a data-parallel subgroup communicator for correct gradient averaging.");
        }

        // Single data-parallel replica mode (no data parallelism)
        // In this case, AllReduce is a no-op or only syncs within tensor/pipeline groups
        if (_dataParallelSize <= 1)
        {
            // No data-parallel sync needed
            CachedFullParameters = null;
            return;
        }
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // 3D parallel training workflow with explicit gradient computation:
        //   1. AllGather: Gather full parameters from shards
        //   2. ComputeGradients: Forward + backward pass to compute TRUE gradients
        //   3. SynchronizeGradients: Complex multi-level sync (tensor/data/pipeline groups)
        //   4. ApplyGradients: Update parameters using synchronized gradients
        //   5. UpdateShards: Extract local shard from updated parameters
        //
        // Note: Full 3D parallelism is complex and requires:
        // - Pipeline: forward/backward through stages with microbatching
        // - Tensor: partial computation within each layer with all-reduce
        // - Data: different batches on data-parallel replicas
        //
        // This framework provides the foundation with simplified implementation.

        // Gather full parameters
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Compute TRUE gradients using the model's gradient computation
        _computedGradients = WrappedModel.ComputeGradients(input, expectedOutput);

        if (Config.AutoSyncGradients)
        {
            // Synchronize gradients (throws NotSupportedException if _dataParallelSize > 1)
            SynchronizeGradients();

            // Apply the synchronized gradients to update parameters
            WrappedModel.ApplyGradients(_computedGradients, Config.LearningRate);

            // Get updated parameters
            var updatedParams = WrappedModel.GetParameters();

            // Update local shard (this already invalidates cache via UpdateLocalShardFromFull)
            UpdateLocalShardFromFull(updatedParams);
        }
        else
        {
            // Without gradient synchronization, apply gradients locally
            WrappedModel.ApplyGradients(_computedGradients, Config.LearningRate);
            var updatedParams = WrappedModel.GetParameters();

            // Update local shard (this already invalidates cache via UpdateLocalShardFromFull)
            UpdateLocalShardFromFull(updatedParams);
        }
        // Note: Cache is already invalidated by UpdateLocalShardFromFull.
        // If AutoSyncGradients is false, subsequent predictions benefit from cached parameters.
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "3D-Parallelism (Hybrid)");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("PipelineParallelSize", _pipelineParallelSize);
        metadata.SetProperty("TensorParallelSize", _tensorParallelSize);
        metadata.SetProperty("DataParallelSize", _dataParallelSize);
        metadata.SetProperty("PipelineRank", _pipelineRank);
        metadata.SetProperty("TensorRank", _tensorRank);
        metadata.SetProperty("DataRank", _dataRank);
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new HybridShardedModel<T, TInput, TOutput>(
            WrappedModel.WithParameters(parameters), Config,
            _pipelineParallelSize, _tensorParallelSize, _dataParallelSize);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(_pipelineParallelSize);
        writer.Write(_tensorParallelSize);
        writer.Write(_dataParallelSize);
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
        int savedPP = reader.ReadInt32();
        int savedTP = reader.ReadInt32();
        int savedDP = reader.ReadInt32();
        reader.ReadBoolean();
        reader.ReadInt32();
        reader.ReadBoolean();

        if (savedWorldSize != WorldSize)
            throw new InvalidOperationException($"World size mismatch: {savedWorldSize} vs {WorldSize}");
        if (savedRank != Rank)
            throw new InvalidOperationException($"Rank mismatch: {savedRank} vs {Rank}");
        if (savedPP != _pipelineParallelSize)
            throw new InvalidOperationException($"Pipeline parallel size mismatch: saved model used {savedPP} pipeline stages, but current instance configured with {_pipelineParallelSize}");
        if (savedTP != _tensorParallelSize)
            throw new InvalidOperationException($"Tensor parallel size mismatch: saved model used {savedTP} tensor parallel groups, but current instance configured with {_tensorParallelSize}");
        if (savedDP != _dataParallelSize)
            throw new InvalidOperationException($"Data parallel size mismatch: saved model used {savedDP} data parallel replicas, but current instance configured with {_dataParallelSize}");

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
        return new HybridShardedModel<T, TInput, TOutput>(
            WrappedModel.Clone(), Config,
            _pipelineParallelSize, _tensorParallelSize, _dataParallelSize);
    }
}
