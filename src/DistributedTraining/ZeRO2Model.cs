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
    private Vector<T>? _parameterDeltaShard;
    private Vector<T>? _computedGradients;
    private Vector<T>? _gradientShard;

    /// <summary>
    /// Gets the local parameter delta shard for this rank after synchronization.
    /// </summary>
    /// <remarks>
    /// <para><b>DEPRECATED:</b> This property is no longer used. ZeRO2Model now uses true gradient
    /// semantics via IFullModel.ComputeGradients() which properly separates gradient computation from
    /// parameter updates. The implementation stores true gradients (not parameter deltas) in the
    /// internal _gradientShard field.</para>
    /// <para>
    /// In the current implementation:
    /// 1. ComputeGradients() computes true gradients via backpropagation without modifying parameters
    /// 2. Gradients are sharded via ReduceScatter so each rank stores only its portion
    /// 3. Each rank updates only its parameter shard using the gradient shard
    /// 4. All ranks perform AllGather to reconstruct the full updated parameter vector
    /// </para>
    /// <para>
    /// This property always returns null in the current implementation. For gradient access,
    /// use the Train() workflow which internally manages _gradientShard.
    /// </para>
    /// </remarks>
    public Vector<T>? ParameterDeltaShard => _parameterDeltaShard;

    public ZeRO2Model(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
        // Must call InitializeSharding() after base constructor to ensure proper initialization
        InitializeSharding();
    }

    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();

        // Parameters still replicated (like ZeRO-1)
        ShardStartIndex = 0;
        ShardSize = fullParameters.Length;
        LocalShard = new Vector<T>(fullParameters.ToArray());

        // Calculate parameter delta shard size to align with ReduceScatter chunk boundaries
        // Using ceiling division ensures chunks align: (34, 34, 32) instead of (34, 33, 33)
        // This prevents misalignment where ReduceScatter chunks don't match logical shard boundaries
        int totalParams = fullParameters.Length;
        int chunkSize = (totalParams + WorldSize - 1) / WorldSize;  // Ceiling division
        int shardStart = Rank * chunkSize;
        int shardEnd = Math.Min((Rank + 1) * chunkSize, totalParams);
        int deltaShardSize = shardEnd - shardStart;

        // Initialize to null - will be populated by SynchronizeGradients()
        _parameterDeltaShard = null;
        CachedFullParameters = null;
    }

    /// <summary>
    /// Synchronizes gradients using ReduceScatter - each process gets its shard of reduced gradients.
    /// </summary>
    public override void SynchronizeGradients()
    {
        if (_computedGradients == null)
        {
            throw new InvalidOperationException(
                "Gradients have not been computed. Call Train() before SynchronizeGradients().");
        }

        var totalParams = _computedGradients.Length;

        // Calculate chunk size using ceiling division to align with ReduceScatter boundaries
        int chunkSize = (totalParams + WorldSize - 1) / WorldSize;
        var paddedLength = chunkSize * WorldSize;

        // Pad to satisfy ReduceScatter's divisibility requirement
        Vector<T> reduceInput = _computedGradients;
        if (paddedLength > totalParams)
        {
            var padded = new T[paddedLength];
            Array.Copy(_computedGradients.ToArray(), padded, totalParams);
            // Padding elements remain at default(T) which is typically 0
            reduceInput = new Vector<T>(padded);
        }

        // Perform ReduceScatter on padded gradient data
        var reducedChunk = Config.CommunicationBackend.ReduceScatter(reduceInput, ReductionOperation.Average);

        // Calculate this rank's shard boundaries in the original (non-padded) parameter space
        // This ensures proper alignment: each rank gets chunkSize elements except the last rank
        // which gets whatever remains. For example, with 100 params and 3 ranks:
        // Rank 0: [0:34) = 34 elements, Rank 1: [34:68) = 34 elements, Rank 2: [68:100) = 32 elements
        int shardStart = Rank * chunkSize;
        int shardEnd = Math.Min((Rank + 1) * chunkSize, totalParams);
        int shardLength = shardEnd - shardStart;

        // Extract the logical shard from the received chunk (trimming any padding)
        var shardData = new T[shardLength];
        Array.Copy(reducedChunk.ToArray(), 0, shardData, 0, shardLength);
        _gradientShard = new Vector<T>(shardData);

        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // ZeRO-2 workflow with explicit gradient computation (IFullModel.IGradientComputable support):
        //   1. ComputeGradients: Forward + backward pass to compute TRUE gradients
        //   2. ReduceScatter gradients (each rank gets a shard)
        //   3. ApplyGradients: Update ONLY local parameter shard using gradient shard
        //   4. AllGather updated shards to reconstruct full parameters
        //
        // Now that IFullModel extends IGradientComputable, we can compute true gradients
        // instead of parameter deltas, enabling proper ZeRO-2 semantics!

        // Set full parameters for gradient computation
        WrappedModel.SetParameters(LocalShard);

        // Compute TRUE gradients using the model's gradient computation
        // This calls the model's backpropagation without updating parameters
        _computedGradients = WrappedModel.ComputeGradients(input, expectedOutput);

        if (Config.AutoSyncGradients)
        {
            // Synchronize gradients via ReduceScatter
            // Each rank receives only its shard of the averaged gradients
            SynchronizeGradients();

            // Apply the gradient shard to update only this rank's parameter shard
            // Note: Learning rate comes from Config. For adaptive optimizers,
            // use ZeRO2Optimizer which properly handles optimizer state.
            if (_gradientShard is null)
                throw new InvalidOperationException("Gradient shard is null after synchronization.");

            var learningRate = Config.LearningRate;
            var updatedShard = new T[_gradientShard.Length];

            int shardStart = Rank * ((LocalShard.Length + WorldSize - 1) / WorldSize);

            // Update only our shard: params[shard] = params[shard] - lr * gradients[shard]
            // Create mutable copy of LocalShard to apply updates
            var updatedParams = LocalShard.ToArray();
            for (int i = 0; i < _gradientShard.Length && (shardStart + i) < LocalShard.Length; i++)
            {
                int globalIndex = shardStart + i;
                updatedParams[globalIndex] = NumOps.Subtract(
                    LocalShard[globalIndex],
                    NumOps.Multiply(learningRate, _gradientShard[i]));
            }

            // Update LocalShard with the modified parameters
            LocalShard = new Vector<T>(updatedParams);

            // AllGather parameter shards to reconstruct full parameters
            LocalShard = AllGatherParameterShards();
            CachedFullParameters = null;
        }
        else
        {
            // Without gradient synchronization, apply gradients locally
            // This is equivalent to non-distributed training
            WrappedModel.ApplyGradients(_computedGradients, Config.LearningRate);
            LocalShard = WrappedModel.GetParameters();
        }
    }

    /// <summary>
    /// Reconstructs full parameters by gathering parameter shards from all ranks.
    /// </summary>
    /// <remarks>
    /// In ZeRO-2, after the optimizer updates each rank's parameter shard,
    /// we need to AllGather all shards to reconstruct the full parameter vector
    /// for the next forward pass. This ensures all ranks have identical synchronized
    /// parameters.
    ///
    /// This method is primarily used when integrating with ZeRO2Optimizer, where each rank
    /// updates only its portion of the parameter vector. Proper AllGather collects these
    /// disjoint updated shards and concatenates them to form the complete parameter vector.
    /// </remarks>
    /// <returns>Full parameter vector reconstructed from all ranks' shards</returns>
    public Vector<T> AllGatherParameterShards()
    {
        // Calculate this rank's shard boundaries (must match SynchronizeGradients logic)
        int totalParams = LocalShard.Length;
        int chunkSize = (totalParams + WorldSize - 1) / WorldSize;
        int shardStart = Rank * chunkSize;
        int shardEnd = Math.Min((Rank + 1) * chunkSize, totalParams);
        int shardLength = shardEnd - shardStart;

        // Extract and pad this rank's parameter shard to uniform chunkSize
        // Padding is required because AllGather expects same-length contributions from all ranks
        var paddedShard = new T[chunkSize];
        Array.Copy(LocalShard.ToArray(), shardStart, paddedShard, 0, shardLength);
        // Padding elements remain at default(T) which is typically 0

        // AllGather: Each rank contributes its padded shard
        var gathered = Config.CommunicationBackend.AllGather(new Vector<T>(paddedShard));

        // Trim padding back to the logical parameter length
        var allData = gathered.ToArray();
        if (allData.Length < totalParams)
        {
            throw new InvalidOperationException(
                $"Expected at least {totalParams} parameters after AllGather, received {allData.Length}.");
        }

        var trimmed = new T[totalParams];
        Array.Copy(allData, 0, trimmed, 0, totalParams);
        return new Vector<T>(trimmed);
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
        bool savedAutoSyncGradients = reader.ReadBoolean();
        int savedMinimumParameterGroupSize = reader.ReadInt32();
        bool savedEnableGradientCompression = reader.ReadBoolean();

        if (savedWorldSize != WorldSize)
            throw new InvalidOperationException($"World size mismatch: {savedWorldSize} vs {WorldSize}");
        if (savedRank != Rank)
            throw new InvalidOperationException($"Rank mismatch: {savedRank} vs {Rank}");

        // Validate configuration compatibility
        if (savedAutoSyncGradients != Config.AutoSyncGradients)
            throw new InvalidOperationException($"AutoSyncGradients mismatch: saved={savedAutoSyncGradients}, current={Config.AutoSyncGradients}");
        if (savedMinimumParameterGroupSize != Config.MinimumParameterGroupSize)
            throw new InvalidOperationException($"MinimumParameterGroupSize mismatch: saved={savedMinimumParameterGroupSize}, current={Config.MinimumParameterGroupSize}");
        if (savedEnableGradientCompression != Config.EnableGradientCompression)
            throw new InvalidOperationException($"EnableGradientCompression mismatch: saved={savedEnableGradientCompression}, current={Config.EnableGradientCompression}");

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
