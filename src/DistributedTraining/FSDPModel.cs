using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements FSDP (Fully Sharded Data Parallel) model wrapper that shards parameters across multiple processes.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// FSDP (Fully Sharded Data Parallel) is PyTorch's implementation of the ZeRO-3 optimization strategy.
/// It shards model parameters, gradients, and optimizer states across all processes, achieving maximum
/// memory efficiency. Parameters are gathered just-in-time for forward/backward passes and then released.
/// </para>
/// <para><b>For Beginners:</b>
/// This class implements FSDP (Fully Sharded Data Parallel), which makes any model work across multiple GPUs or machines
/// with maximum memory efficiency. It automatically handles:
/// - Splitting ALL model components (parameters, gradients, optimizer states) across processes
/// - Gathering parameters only when needed for forward/backward pass
/// - Releasing parameters immediately after use to save memory
/// - Averaging gradients across all processes during training
/// </para>
/// <para>
/// Think of it like a team project where each person holds part of the solution, but unlike DDP,
/// FSDP only shares the full model temporarily when absolutely needed, then immediately goes back
/// to holding just their piece. This saves a lot of memory!
/// </para>
/// <para><b>Use Cases:</b>
/// - Training very large models that don't fit in a single GPU's memory
/// - Maximizing memory efficiency for multi-GPU training
/// - Scaling to hundreds or thousands of GPUs
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Excellent - shards everything (parameters + gradients + optimizer states)
/// - Communication: Higher - requires AllGather for each forward/backward pass
/// - Complexity: Moderate - automatic just-in-time parameter gathering
/// - Best for: Very large models, memory-constrained scenarios
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Wrap it for FSDP distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var fsdpModel = new FSDPModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Now train as usual - FSDP magic happens automatically!
/// fsdpModel.Train(inputs, outputs);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class FSDPModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new FSDP model wrapping an existing model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing model and makes it distributed using FSDP strategy.
    /// You provide:
    /// 1. The model you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The constructor automatically:
    /// - Splits the model's parameters across all processes (sharding)
    /// - Sets up communication channels
    /// - Prepares everything for FSDP distributed training
    /// </para>
    /// </remarks>
    /// <param name="wrappedModel">The model to wrap with FSDP capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public FSDPModel(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Gather full parameters for training
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Train the wrapped model
        WrappedModel.Train(input, expectedOutput);

        // Get updated parameters
        var updatedParams = WrappedModel.GetParameters();

        // Update local shard
        UpdateLocalShardFromFull(updatedParams);

        // Invalidate cache immediately after local shard changes
        InvalidateCache();

        // Synchronize gradients if auto-sync is enabled
        if (Config.AutoSyncGradients)
        {
            SynchronizeGradients();

            // Apply synchronized parameters back to the model
            fullParams = GatherFullParameters();
            WrappedModel.SetParameters(fullParams);
        }
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        // Gather full parameters for prediction
        var fullParams = GatherFullParameters();
        WrappedModel.SetParameters(fullParams);

        // Use wrapped model for prediction
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>FSDP Gradient Synchronization:</b>
    /// Unlike the base implementation which does AllReduce on LocalShard (which would mix disjoint
    /// parameter indices from different ranks and corrupt weights), FSDP gradient synchronization works by:
    /// 1. Gathering full parameters from all ranks (allGather operation)
    /// 2. Performing AllReduce on the full parameter vector (synchronizing matching indices)
    /// 3. Scattering the result back to local shards (each rank gets its portion)
    ///
    /// This ensures we reduce matching data across ranks rather than mixing unrelated parameter indices.
    /// </para>
    /// </remarks>
    public override void SynchronizeGradients()
    {
        // Gather full parameters from all shards
        // This ensures we have the complete parameter vector on each rank
        var fullParams = GatherFullParameters();

        // Perform AllReduce on the full parameter vector
        // This synchronizes matching parameter indices across all ranks
        Config.CommunicationBackend.AllReduce(fullParams, ReductionOperation.Average);

        // Update local shard from the synchronized full parameters
        // Each rank extracts its portion of the reduced parameters
        UpdateLocalShardFromFull(fullParams);

        // Invalidate cached full parameters to force re-gather on next access
        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();

        // Add distributed training info
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "FSDP");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("ShardSize", ShardSize);

        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = WrappedModel.WithParameters(parameters);
        return new FSDPModel<T, TInput, TOutput>(newModel, Config);
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(Config.AutoSyncGradients);
        writer.Write(Config.MinimumParameterGroupSize);
        writer.Write(Config.EnableGradientCompression);

        // Serialize wrapped model
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

        // Read sharding configuration (for validation)
        int savedWorldSize = reader.ReadInt32();
        int savedRank = reader.ReadInt32();
        reader.ReadBoolean(); // AutoSyncGradients
        reader.ReadInt32(); // MinimumParameterGroupSize
        reader.ReadBoolean(); // EnableGradientCompression

        if (savedWorldSize != WorldSize)
        {
            throw new InvalidOperationException(
                $"World size mismatch. Model was trained with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        // Validate rank matches - different rank could indicate configuration mismatch
        if (savedRank != Rank)
        {
            throw new InvalidOperationException(
                $"Rank mismatch. Model was saved on rank {savedRank}, " +
                $"but is being loaded on rank {Rank}. This could indicate a configuration error.");
        }

        // Read wrapped model
        int modelDataLength = reader.ReadInt32();
        byte[] modelData = reader.ReadBytes(modelDataLength);
        WrappedModel.Deserialize(modelData);

        // Re-initialize sharding
        InitializeSharding();
    }

    /// <inheritdoc/>
    public override void SaveModel(string filePath)
    {
        // Barrier before rank check to prevent deadlock if rank 0 fails
        Config.CommunicationBackend.Barrier();

        try
        {
            // Only rank 0 saves to avoid file write conflicts
            if (Rank == 0)
            {
                var data = Serialize();
                File.WriteAllBytes(filePath, data);
            }
        }
        finally
        {
            // Ensure all processes reach this barrier even if rank 0 fails
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override void LoadModel(string filePath)
    {
        // Barrier before loading to ensure all processes start together
        Config.CommunicationBackend.Barrier();

        try
        {
            // All processes read the same file (read-only, no conflicts)
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        finally
        {
            // Ensure all processes finish loading before proceeding
            Config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> Clone()
    {
        var clonedWrappedModel = WrappedModel.Clone();
        return new FSDPModel<T, TInput, TOutput>(clonedWrappedModel, Config);
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        return WrappedModel.GetFeatureImportance();
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var deepCopiedWrappedModel = WrappedModel.DeepCopy();
        return new FSDPModel<T, TInput, TOutput>(deepCopiedWrappedModel, Config);
    }

    /// <inheritdoc/>
    public override IEnumerable<int> GetActiveFeatureIndices()
    {
        return WrappedModel.GetActiveFeatureIndices();
    }

    /// <inheritdoc/>
    public override void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        WrappedModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <inheritdoc/>
    public override bool IsFeatureUsed(int featureIndex)
    {
        return WrappedModel.IsFeatureUsed(featureIndex);
    }
}
