using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 2 model wrapper - shards optimizer states and gradients.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO Stage 2 builds on ZeRO-1 by additionally sharding gradients across processes.
/// Parameters are still replicated for the forward pass, but gradients are reduced and scattered
/// (ReduceScatter) so each process only stores a portion. This saves significant memory compared
/// to ZeRO-1, especially for large models.
/// </para>
/// <para><b>For Beginners:</b>
/// This implements ZeRO Stage 2, which saves even more memory than ZeRO-1. The model parameters
/// are still fully replicated (like DDP and ZeRO-1), but now both the optimizer state AND the
/// gradients are split across processes. After computing gradients, they're immediately reduced
/// and scattered so each process only keeps its portion.
/// </para>
/// <para>
/// Think of it like a team where everyone has the full playbook (parameters), but when taking
/// notes during practice (gradients), they divide up the note-taking so each person is responsible
/// for recording only certain plays. This saves everyone from having to write everything down.
/// </para>
/// <para><b>Use Cases:</b>
/// - Larger models where gradient memory becomes significant
/// - Want substantial memory savings with moderate communication cost
/// - Preparing for ZeRO-3/FSDP migration
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Very Good - saves both optimizer states and gradients
/// - Communication: Moderate - uses ReduceScatter instead of AllReduce
/// - Complexity: Moderate - gradient sharding adds some complexity
/// - Best for: Large models where gradient memory is significant
/// </para>
/// <para>
/// Example:
/// <code>
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var zero2Model = new ZeRO2Model&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Use with ZeRO2Optimizer for full ZeRO-2 benefits
/// var zero2Optimizer = new ZeRO2Optimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(optimizer, config);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO2Model<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private Vector<T>? _gradientShard;

    public ZeRO2Model(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
    }

    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();

        // Parameters still replicated (like ZeRO-1)
        ShardStartIndex = 0;
        ShardSize = fullParameters.Length;
        LocalShard = new Vector<T>(fullParameters.ToArray());

        // Calculate gradient shard size
        int totalParams = fullParameters.Length;
        int baseShardSize = totalParams / WorldSize;
        int remainder = totalParams % WorldSize;
        int gradShardSize = baseShardSize + (Rank < remainder ? 1 : 0);

        _gradientShard = new Vector<T>(new T[gradShardSize]);
        CachedFullParameters = null;
    }

    /// <summary>
    /// Synchronizes gradients using ReduceScatter - each process gets its shard of reduced gradients.
    /// </summary>
    public override void SynchronizeGradients()
    {
        var totalParams = LocalShard.Length;
        var remainder = totalParams % WorldSize;

        // Pad to satisfy ReduceScatter's divisibility requirement
        Vector<T> reduceInput = LocalShard;
        if (remainder != 0)
        {
            var paddedLength = totalParams + (WorldSize - remainder);
            var padded = new T[paddedLength];
            Array.Copy(LocalShard.ToArray(), padded, totalParams);
            // Padding elements remain at default(T) which is typically 0
            reduceInput = new Vector<T>(padded);
        }

        // Perform ReduceScatter on padded data
        var reducedChunk = Config.CommunicationBackend.ReduceScatter(reduceInput, ReductionOperation.Average);

        // Trim padding so each rank keeps only its logical shard
        // Distribute remainder elements to first 'remainder' ranks
        var shardLength = totalParams / WorldSize + (Rank < remainder ? 1 : 0);
        var shardData = new T[shardLength];
        Array.Copy(reducedChunk.ToArray(), 0, shardData, 0, shardLength);
        _gradientShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Full parameters available locally
        WrappedModel.SetParameters(LocalShard);
        WrappedModel.Train(input, expectedOutput);
        LocalShard = WrappedModel.GetParameters();

        if (Config.AutoSyncGradients)
        {
            SynchronizeGradients();
            // After ReduceScatter, we have sharded gradients in _gradientShard
            // For next forward pass, we need to AllGather parameters
            // (This is simplified - full implementation would handle this more carefully)
            // Cache invalidated by SynchronizeGradients
        }
        // Note: Cache not invalidated if AutoSyncGradients is false,
        // allowing multiple predictions to benefit from cached full parameters
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        WrappedModel.SetParameters(LocalShard);
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "ZeRO-2");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("OptimizerStateSharded", true);
        metadata.SetProperty("GradientsSharded", true);
        metadata.SetProperty("ParametersReplicated", true);
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        return new ZeRO2Model<T, TInput, TOutput>(WrappedModel.WithParameters(parameters), Config);
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
        return new ZeRO2Model<T, TInput, TOutput>(WrappedModel.Clone(), Config);
    }
}
