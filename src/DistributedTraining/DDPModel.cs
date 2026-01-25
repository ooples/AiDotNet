using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements DDP (Distributed Data Parallel) model wrapper for distributed training.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// DDP (Distributed Data Parallel) is the most common and straightforward distributed training strategy.
/// Each process maintains a full replica of the model. During training, gradients are synchronized
/// across all processes using AllReduce, ensuring all replicas stay identical. This is PyTorch's
/// default distributed training strategy.
/// </para>
/// <para><b>For Beginners:</b>
/// This class implements DDP (Distributed Data Parallel), the simplest and most popular way to train
/// models across multiple GPUs or machines. Unlike FSDP which shards parameters, DDP keeps a complete
/// copy of the model on each process. It automatically handles:
/// - Keeping full model parameters on each process (no sharding)
/// - Averaging gradients across all processes after backward pass
/// - Ensuring all model replicas stay synchronized
/// </para>
/// <para>
/// Think of it like multiple chefs each making the full recipe. After each step, they compare notes
/// and average their learnings, so everyone stays on the same page. This is simpler than FSDP where
/// each person only knows part of the recipe.
/// </para>
/// <para><b>Use Cases:</b>
/// - Standard multi-GPU training where model fits in single GPU memory
/// - When communication is fast (NVLink, InfiniBand)
/// - Simpler debugging than FSDP (full model on each process)
/// - Default choice for most distributed training scenarios
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Moderate - each process stores full model (parameters replicated)
/// - Communication: Low - only gradients synchronized (AllReduce after backward)
/// - Complexity: Low - simplest distributed strategy
/// - Best for: Models that fit in single GPU memory, fast interconnects
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Wrap it for DDP distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var ddpModel = new DDPModel&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Now train as usual - DDP magic happens automatically!
/// ddpModel.Train(inputs, outputs);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class DDPModel<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private Vector<T>? _computedGradients;

    /// <summary>
    /// Creates a new DDP model wrapping an existing model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This constructor takes your existing model and makes it distributed using DDP strategy.
    /// You provide:
    /// 1. The model you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    /// </para>
    /// <para>
    /// The constructor automatically:
    /// - Ensures each process has a full copy of the model
    /// - Sets up communication channels for gradient synchronization
    /// - Prepares everything for DDP distributed training
    /// </para>
    /// </remarks>
    /// <param name="wrappedModel">The model to wrap with DDP capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public DDPModel(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
        // Must call InitializeSharding() after base constructor to ensure proper initialization
        InitializeSharding();
    }

    /// <summary>
    /// Initializes DDP - no actual parameter sharding, each process keeps full parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Unlike FSDP which splits parameters, DDP keeps the full model on each process.
    /// This method sets up the local shard to actually be the full parameter set.
    /// </para>
    /// </remarks>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();

        // In DDP, each process holds ALL parameters (no actual sharding)
        ShardStartIndex = 0;
        ShardSize = fullParameters.Length;
        LocalShard = new Vector<T>(fullParameters.ToArray());

        // Invalidate cache
        CachedFullParameters = null;
    }

    /// <summary>
    /// Synchronizes gradients across all processes using AllReduce.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// After training on local data, each process has computed gradients based on its batch.
    /// This method averages those gradients across all processes so everyone has the same update.
    /// This is the core of DDP - gradient averaging via AllReduce.
    /// </para>
    /// </remarks>
    public override void SynchronizeGradients()
    {
        if (_computedGradients == null)
        {
            throw new InvalidOperationException(
                "Gradients have not been computed. Call Train() before SynchronizeGradients().");
        }

        // In DDP, we AllReduce the full gradient vector
        // This averages gradients across all processes
        Config.CommunicationBackend.AllReduce(_computedGradients, ReductionOperation.Average);

        // Invalidate cached full parameters
        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // DDP workflow with explicit gradient computation:
        //   1. ComputeGradients: Forward + backward pass to compute TRUE gradients
        //   2. AllReduce gradients (average across all processes)
        //   3. ApplyGradients: Update local parameters using averaged gradients
        //
        // Since all processes have full parameters and use the same averaged gradients,
        // they all end up with identical parameters (DDP's key property).

        // Set full parameters for gradient computation
        WrappedModel.SetParameters(LocalShard);

        // Compute TRUE gradients using the model's gradient computation
        // This calls the model's backpropagation without updating parameters
        _computedGradients = WrappedModel.ComputeGradients(input, expectedOutput);

        if (Config.AutoSyncGradients)
        {
            // Synchronize gradients via AllReduce
            // All processes receive the averaged gradients
            SynchronizeGradients();

            // Apply the averaged gradients to update parameters
            // Note: For adaptive optimizers, use DDPOptimizer which properly handles optimizer state.
            WrappedModel.ApplyGradients(_computedGradients, Config.LearningRate);

            // Get updated parameters back to LocalShard
            LocalShard = WrappedModel.GetParameters();

            // Invalidate cache after synchronization since parameters changed
            InvalidateCache();
        }
        else
        {
            // Without gradient synchronization, apply gradients locally
            // This is equivalent to non-distributed training
            WrappedModel.ApplyGradients(_computedGradients, Config.LearningRate);
            LocalShard = WrappedModel.GetParameters();
        }
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        // No need to gather - we already have full parameters locally
        WrappedModel.SetParameters(LocalShard);

        // Use wrapped model for prediction
        return WrappedModel.Predict(input);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = WrappedModel.GetModelMetadata();

        // Add distributed training info
        metadata.SetProperty("IsDistributed", true);
        metadata.SetProperty("Strategy", "DDP");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("ParametersReplicated", true);

        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = WrappedModel.WithParameters(parameters);
        return new DDPModel<T, TInput, TOutput>(newModel, Config);
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

        // Validate rank matches
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

        // Re-initialize (will set full parameters, not sharded)
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
        return new DDPModel<T, TInput, TOutput>(clonedWrappedModel, Config);
    }
}
