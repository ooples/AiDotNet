using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements a distributed model wrapper that shards parameters across multiple processes.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// This class wraps any existing model and makes it work across multiple GPUs or machines.
/// It automatically handles:
/// - Splitting parameters across processes (sharding)
/// - Gathering parameters when needed for forward pass
/// - Averaging gradients across all processes during training
/// </para>
/// <para>
/// Think of it like a team project where each person holds part of the solution.
/// When you need the full solution, everyone shares their part (AllGather).
/// When everyone learns something new, they share and average their learnings (AllReduce).
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Wrap it for distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var distributedModel = new ShardedModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Now train as usual - distributed magic happens automatically!
/// distributedModel.Train(inputs, outputs);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ShardedModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a new sharded model wrapping an existing model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing model and makes it distributed.
    /// You provide:
    /// 1. The model you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The constructor automatically:
    /// - Splits the model's parameters across all processes
    /// - Sets up communication channels
    /// - Prepares everything for distributed training
    /// </para>
    /// </remarks>
    /// <param name="wrappedModel">The model to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public ShardedModel(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
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
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();

        // Add distributed training info
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("ShardSize", ShardSize);

        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = WrappedModel.WithParameters(parameters);
        return new ShardedModel<T, TInput, TOutput>(newModel, Config);
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
        return new ShardedModel<T, TInput, TOutput>(clonedWrappedModel, Config);
    }

    /// <inheritdoc/>
    public Dictionary<string, T> GetFeatureImportance()
    {
        return WrappedModel.GetFeatureImportance();
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var deepCopiedWrappedModel = WrappedModel.DeepCopy();
        return new ShardedModel<T, TInput, TOutput>(deepCopiedWrappedModel, Config);
    }

    /// <inheritdoc/>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return WrappedModel.GetActiveFeatureIndices();
    }

    /// <inheritdoc/>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        WrappedModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <inheritdoc/>
    public bool IsFeatureUsed(int featureIndex)
    {
        return WrappedModel.IsFeatureUsed(featureIndex);
    }
}
