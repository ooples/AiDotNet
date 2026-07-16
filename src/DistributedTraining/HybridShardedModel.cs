using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ResearchPaper("PyTorch FSDP: Scaling Fully Sharded Data Parallel", "https://arxiv.org/abs/2304.11277")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
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

        // PendingConfig is cleared in OnBeforeInitializeSharding after consumption
        // (not here, because lazy init means OnBeforeInitializeSharding may not have run yet)
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

        // Clear the pending config now that we've consumed it
        PendingConfig.Value = null;
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
        var fullParameters = InterfaceGuard.Parameterizable(WrappedModel).GetParameters();
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

        // Synchronization scope for THIS (parameter-sharding) hybrid model: this wrapper gathers the
        // full parameters and computes the FULL-model gradient on every rank (the wrapped model's
        // compute is a black box that cannot be layer-partitioned). Therefore:
        // - Tensor- and pipeline-parallel neighbours own DIFFERENT parameter shards of that full
        //   gradient — there are NO partial-sum contributions to reduce across them (partial-sum
        //   tensor-parallel gradients arise only in COMPUTE-partitioned Megatron TP, which is provided
        //   separately by ColumnParallelLinear/RowParallelLinear whose ḡ operator carries the in-layer
        //   AllReduce). So no cross-tensor/pipeline gradient reduction is required or correct here.
        // - Data-parallel replicas share the same parameter shard but may process different data, so
        //   THEIR gradients must be averaged. That data-parallel subgroup average is the complete and
        //   correct gradient synchronization for this model; it is performed below.

        // Average gradients across the DATA-PARALLEL replica group ONLY — the ranks sharing this
        // rank's pipeline+tensor position, which hold REPLICATED parameters but processed different
        // data batches (so their gradients genuinely differ and must be averaged). The tensor- and
        // pipeline-parallel neighbours own DIFFERENT parameter shards, so a full-world AllReduce would
        // sum unrelated gradients and corrupt those shards; we reduce strictly within the data-parallel
        // subgroup [groupStart, groupStart + _dataParallelSize).
        if (_dataParallelSize > 1)
        {
            _computedGradients = AverageGradientsAcrossDataParallelGroup(_computedGradients);
        }
        // With a single data-parallel replica there is nothing to average across; each tensor/pipeline
        // position keeps its own gradients (handled by its own shard update).
        CachedFullParameters = null;
    }

    /// <summary>
    /// Averages a gradient vector across this rank's data-parallel replica group (a subgroup AllReduce
    /// with Average). The 3D layout is [pipeline][tensor][data], so the data-parallel replicas of a
    /// given (pipeline, tensor) position are the <c>_dataParallelSize</c> CONSECUTIVE ranks
    /// [groupStart, groupStart + _dataParallelSize), with
    /// <c>groupStart = pipelineRank·(tensor·data) + tensorRank·data</c>. The communication backend
    /// exposes only world-wide collectives, so the subgroup reduction is built from point-to-point
    /// Send/Receive: every non-leader sends its gradient to the group leader (the lowest rank in the
    /// group), the leader sums all contributions, divides by the group size, and sends the average
    /// back. This reduces ONLY within the data-parallel group and never touches the tensor/pipeline
    /// neighbours' distinct parameter shards.
    /// </summary>
    private Vector<T> AverageGradientsAcrossDataParallelGroup(Vector<T> gradients)
    {
        var backend = Config.CommunicationBackend;
        int rank = backend.Rank;
        int tensorGroupSize = _tensorParallelSize * _dataParallelSize;
        int groupStart = (rank / tensorGroupSize) * tensorGroupSize + _tensorRank * _dataParallelSize;
        int leader = groupStart;
        int n = gradients.Length;
        const int TagToLeader = 0x5D0;
        const int TagFromLeader = 0x5D1;

        if (rank == leader)
        {
            var sum = gradients.ToArray(); // leader's own contribution
            for (int d = 1; d < _dataParallelSize; d++)
            {
                var recv = backend.Receive(groupStart + d, n, TagToLeader);
                for (int i = 0; i < n; i++) sum[i] = NumOps.Add(sum[i], recv[i]);
            }
            var invCount = NumOps.FromDouble(1.0 / _dataParallelSize);
            for (int i = 0; i < n; i++) sum[i] = NumOps.Multiply(sum[i], invCount);
            var averaged = new Vector<T>(sum);
            for (int d = 1; d < _dataParallelSize; d++)
                backend.Send(averaged, groupStart + d, TagFromLeader);
            return averaged;
        }

        backend.Send(gradients, leader, TagToLeader);
        return backend.Receive(leader, n, TagFromLeader);
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
        // Synchronization model for THIS parameter-sharded hybrid wrapper (see SynchronizeGradients for
        // the full rationale): the wrapped model is a black box whose compute cannot be layer-partitioned,
        // so every rank gathers the full parameters and computes the FULL-model gradient. Data-parallel
        // replicas therefore hold replicated parameters but process different batches and MUST be averaged
        // (done in SynchronizeGradients); tensor/pipeline neighbours own DISJOINT parameter shards of that
        // full gradient, so there is nothing to reduce across them. Compute-partitioned tensor parallelism
        // (partial per-layer matmuls + in-layer all-reduce) is provided separately by the Megatron layer
        // primitives (ColumnParallelLinear/RowParallelLinear); a model built from those does not need this
        // black-box wrapper.

        // Gather full parameters
        var fullParams = GatherFullParameters();
        InterfaceGuard.Parameterizable(WrappedModel).SetParameters(fullParams);

        // Compute TRUE gradients using the model's gradient computation
        _computedGradients = InterfaceGuard.GradientComputable(WrappedModel).ComputeGradients(input, expectedOutput);

        if (Config.AutoSyncGradients)
        {
            // ZeRO Stage-2 offload: bring gradients to CPU and drop the GPU
            // cache entry before any subgroup reduction runs inside
            // SynchronizeGradients. No-op when CpuOffloadGradients is off.
            OffloadGradientsToCpu(_computedGradients);
            // Average gradients within the data-parallel replica subgroup (no-op at _dataParallelSize == 1).
            SynchronizeGradients();

            // Apply the synchronized gradients to update parameters
            InterfaceGuard.GradientComputable(WrappedModel).ApplyGradients(_computedGradients, Config.LearningRate);

            // Get updated parameters
            var updatedParams = InterfaceGuard.Parameterizable(WrappedModel).GetParameters();

            // Update local shard (this already invalidates cache via UpdateLocalShardFromFull)
            UpdateLocalShardFromFull(updatedParams);
        }
        else
        {
            // Without gradient synchronization, apply gradients locally
            InterfaceGuard.GradientComputable(WrappedModel).ApplyGradients(_computedGradients, Config.LearningRate);
            var updatedParams = InterfaceGuard.Parameterizable(WrappedModel).GetParameters();

            // Update local shard (this already invalidates cache via UpdateLocalShardFromFull)
            UpdateLocalShardFromFull(updatedParams);
        }
        // Note: Cache is already invalidated by UpdateLocalShardFromFull.
        // If AutoSyncGradients is false, subsequent predictions benefit from cached parameters.

        // ZeRO Stage-3 param offload: drop GPU-cached params so the next
        // forward re-uploads from the just-updated CPU-resident values.
        // No-op when CpuOffloadParams is off.
        OffloadParamsToCpu();
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var fullParams = GatherFullParameters();
        InterfaceGuard.Parameterizable(WrappedModel).SetParameters(fullParams);
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
            InterfaceGuard.Parameterizable(WrappedModel).WithParameters(parameters), Config,
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
        // Synchronize ranks, then delegate to base which handles AIMF envelope + per-rank paths
        Config.CommunicationBackend.Barrier();
        try
        {
            base.SaveModel(filePath);
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
            base.LoadModel(filePath);
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
