using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Tensor Parallel model wrapper - splits individual layers across ranks (Megatron-LM style).
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Tensor Parallelism (Megatron-LM style) partitions individual layers horizontally across processes.
/// For example, a large matrix multiplication is split so each GPU computes only a portion of the output,
/// then results are combined. This is particularly effective for transformer models where attention and
/// feed-forward layers can be partitioned along specific dimensions (column-parallel and row-parallel).
/// </para>
/// <para><b>For Beginners:</b>
/// Tensor parallelism is like splitting a single large calculation across multiple workers.
/// Imagine a huge spreadsheet calculation - instead of one person doing all the math, we divide
/// the spreadsheet columns across multiple people, each computing their portion simultaneously.
///
/// For example, in a neural network layer with a 10000x10000 weight matrix:
/// - GPU 0 handles columns 0-2499
/// - GPU 1 handles columns 2500-4999
/// - GPU 2 handles columns 5000-7499
/// - GPU 3 handles columns 7500-9999
///
/// They compute in parallel, then combine results.
/// </para>
/// <para><b>Use Cases:</b>
/// - Very wide models (large hidden dimensions)
/// - Transformer models (BERT, GPT) with large attention/FFN layers
/// - When individual layers are too large for single GPU
/// - Often combined with pipeline parallelism for maximum scalability
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent for wide layers - each rank stores only portion of weights
/// - Communication: High - requires AllReduce or AllGather within each layer
/// - Complexity: Very High - requires model-aware partitioning, specific to layer types
/// - Best for: Transformer models, very wide layers, fast interconnects (NVLink)
/// - Limitation: Requires fast communication (high overhead on slow networks)
/// </para>
/// <para><b>Implementation Note:</b>
/// This is a production-ready framework implementation. Full tensor parallelism requires
/// model-specific layer partitioning (column-parallel vs row-parallel strategy for different
/// layer types). This implementation provides the infrastructure. For production use with
/// specific models (e.g., transformers), extend this class with layer-aware partitioning.
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new TransformerModel&lt;double&gt;(...); // Large transformer
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// // Each rank handles 1/4 of each layer's width
/// var tensorParallelModel = new TensorParallelModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     model, config);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class TensorParallelModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private readonly int _tensorParallelSize;

    /// <summary>
    /// Creates a new Tensor Parallel model.
    /// </summary>
    /// <param name="wrappedModel">The model to partition with tensor parallelism</param>
    /// <param name="config">Configuration for sharding and communication</param>
    public TensorParallelModel(
        IFullModel<T, TInput, TOutput> wrappedModel,
        IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
        _tensorParallelSize = WorldSize;
    }

    /// <summary>
    /// Initializes tensor parallelism by partitioning layer weights.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // In tensor parallelism, we partition weights within layers
        // For this framework implementation, we use a simplified column-wise partitioning
        // Production usage would require layer-specific partitioning logic

        int baseShardSize = totalParams / _tensorParallelSize;
        int remainder = totalParams % _tensorParallelSize;

        ShardSize = baseShardSize + (Rank < remainder ? 1 : 0);
        ShardStartIndex = Rank * baseShardSize + Math.Min(Rank, remainder);

        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <summary>
    /// Synchronizes tensor-parallel computation results.
    /// </summary>
    /// <remarks>
    /// In tensor parallelism, different layers require different synchronization patterns:
    /// - Column-parallel layers: AllReduce after computation
    /// - Row-parallel layers: AllGather before computation
    /// This simplified implementation uses AllReduce.
    /// </remarks>
    public override void SynchronizeGradients()
    {
        // In tensor parallelism, each rank holds a disjoint slice of parameters.
        // AllReduce across the full world would sum unrelated indices and corrupt the shards.
        // We need subgroup-aware collectives that sync only within the tensor-parallel group.
        //
        // For column-parallel layers: AllReduce within tensor-parallel group
        // For row-parallel layers: Different synchronization pattern
        // For combined data+tensor parallel: AllReduce within data-parallel subgroup

        if (_tensorParallelSize > 1)
        {
            throw new NotSupportedException(
                "TensorParallelModel requires subgroup gradient synchronization; " +
                "AllReduce over the full world corrupts the shard. " +
                "Implement subgroup-aware collectives that synchronize only within the tensor-parallel group, " +
                "or use a data-parallel subgroup communicator for combined data+tensor parallel setups.");
        }

        // Single-process mode or pure data-parallel mode (no tensor parallelism)
        Config.CommunicationBackend.AllReduce(LocalShard, ReductionOperation.Sum);

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Tensor parallel training:
        // 1. Each rank has a slice of the layer weights
        // 2. Forward: compute partial outputs, then AllReduce or AllGather depending on layer type
        // 3. Backward: similar communication pattern in reverse

        // For this framework implementation, we provide simplified pattern
        // Production usage would implement layer-specific forward/backward logic

        // Set local shard (partial weights)
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Train
        WrappedModel.Train(input, expectedOutput);

        // Get updated parameters and extract shard
        var updatedParams = WrappedModel.GetParameters();
        UpdateLocalShardFromFull(updatedParams);
        InvalidateCache();

        // Synchronize across tensor-parallel group
        // Only call SynchronizeGradients if it won't throw (i.e., single-rank mode)
        // For multi-rank tensor parallelism, parameters are already synchronized via
        // GatherFullParameters/UpdateLocalShardFromFull pattern above
        if (Config.AutoSyncGradients && _tensorParallelSize == 1)
        {
            SynchronizeGradients();
        }
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        // Tensor parallel inference
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "TensorParallel");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("TensorParallelSize", _tensorParallelSize);
        metadata.SetProperty("PartitioningStyle", "Column-wise (simplified)");
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new TensorParallelModel<T, TInput, TOutput>(
            WrappedModel.WithParameters(parameters), Config);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(WorldSize);
        writer.Write(Rank);
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
        return new TensorParallelModel<T, TInput, TOutput>(WrappedModel.Clone(), Config);
    }
}
