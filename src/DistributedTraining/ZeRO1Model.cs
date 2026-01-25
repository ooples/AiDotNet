using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 1 model wrapper - shards optimizer states only.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO Stage 1 (Zero Redundancy Optimizer) shards only the optimizer states (momentum, variance buffers)
/// across processes while keeping parameters and gradients replicated. This provides memory savings
/// for the optimizer state without changing the training communication pattern significantly.
/// </para>
/// <para><b>For Beginners:</b>
/// This class implements ZeRO Stage 1, which is a middle ground between DDP and full sharding.
/// The model parameters and gradients are replicated (like DDP), but the optimizer's internal state
/// (like momentum buffers in Adam) is split across processes to save memory.
/// </para>
/// <para>
/// Think of it like a team where everyone has the full playbook (parameters) and shares all their
/// notes (gradients), but each person keeps their own personal training journal (optimizer state)
/// rather than everyone keeping a copy of everyone's journals.
/// </para>
/// <para><b>Use Cases:</b>
/// - When optimizer state is large (Adam, RMSprop) but model fits in memory
/// - Want some memory savings without full FSDP complexity
/// - Gradual migration path from DDP to ZeRO-3/FSDP
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Good - saves optimizer state memory (4x for Adam: fp32 params + momentum + variance + gradients)
/// - Communication: Low - same as DDP (AllReduce gradients)
/// - Complexity: Low - similar to DDP, works with ZeRO1Optimizer
/// - Best for: Large models with stateful optimizers, transitioning from DDP
/// </para>
/// <para>
/// Example:
/// <code>
/// // Original model
/// var model = new NeuralNetworkModel&lt;double&gt;(...);
///
/// // Wrap it for ZeRO-1 distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var zero1Model = new ZeRO1Model&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, config);
///
/// // Use with ZeRO1Optimizer for full ZeRO-1 benefits
/// var zero1Optimizer = new ZeRO1Optimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(optimizer, config);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO1Model<T, TInput, TOutput> : ShardedModelBase<T, TInput, TOutput>
{
    private Vector<T>? _computedGradients;

    /// <summary>
    /// Creates a new ZeRO-1 model wrapping an existing model.
    /// </summary>
    /// <param name="wrappedModel">The model to wrap with ZeRO-1 capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    public ZeRO1Model(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
        : base(wrappedModel, config)
    {
        // Must call InitializeSharding() after base constructor to ensure proper initialization
        InitializeSharding();
    }

    /// <summary>
    /// Initializes ZeRO-1 - no parameter sharding, keeps full parameters like DDP.
    /// </summary>
    protected override void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();

        // In ZeRO-1, parameters are replicated (no sharding at model level)
        // Optimizer state sharding is handled by ZeRO1Optimizer
        ShardStartIndex = 0;
        ShardSize = fullParameters.Length;
        LocalShard = new Vector<T>(fullParameters.ToArray());

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

        // Same as DDP - AllReduce all gradients
        Config.CommunicationBackend.AllReduce(_computedGradients, ReductionOperation.Average);
        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // ZeRO-1 workflow with explicit gradient computation:
        //   1. ComputeGradients: Forward + backward pass to compute TRUE gradients
        //   2. AllReduce gradients (average across all processes - same as DDP)
        //   3. ApplyGradients: Update local parameters using averaged gradients
        //
        // ZeRO-1 is like DDP for the model - full parameters and gradients on each process.
        // The difference is that ZeRO-1 pairs with ZeRO1Optimizer which shards optimizer states.

        // Set full parameters for gradient computation
        WrappedModel.SetParameters(LocalShard);

        // Compute TRUE gradients using the model's gradient computation
        _computedGradients = WrappedModel.ComputeGradients(input, expectedOutput);

        if (Config.AutoSyncGradients)
        {
            // Synchronize gradients via AllReduce
            SynchronizeGradients();

            // Apply the averaged gradients to update parameters
            // Note: We use a fixed learning rate here. For adaptive optimizers,
            // use ZeRO1Optimizer which properly handles optimizer state sharding.
            WrappedModel.ApplyGradients(_computedGradients, NumOps.FromDouble(0.01));

            // Get updated parameters back to LocalShard
            LocalShard = WrappedModel.GetParameters();
            WrappedModel.SetParameters(LocalShard);
            // Cache invalidated by SynchronizeGradients
        }
        else
        {
            // Without gradient synchronization, apply gradients locally
            WrappedModel.ApplyGradients(_computedGradients, NumOps.FromDouble(0.01));
            LocalShard = WrappedModel.GetParameters();
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
        metadata.SetProperty("Strategy", "ZeRO-1");
        metadata.SetProperty("WorldSize", WorldSize);
        metadata.SetProperty("Rank", Rank);
        metadata.SetProperty("OptimizerStateSharded", true);
        metadata.SetProperty("ParametersReplicated", true);
        metadata.SetProperty("GradientsReplicated", true);
        return metadata;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = WrappedModel.WithParameters(parameters);
        return new ZeRO1Model<T, TInput, TOutput>(newModel, Config);
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
        return new ZeRO1Model<T, TInput, TOutput>(WrappedModel.Clone(), Config);
    }
}
