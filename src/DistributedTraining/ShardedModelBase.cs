using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;


namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides base implementation for distributed models with parameter sharding.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for all sharded models,
/// including parameter management, sharding logic, gradient synchronization, and
/// integration with the model serialization system. Derived classes can customize
/// the sharding strategy, communication pattern, or add optimization-specific features.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all distributed models build upon.
///
/// Think of this as a template for splitting a big model across multiple computers or GPUs.
/// It handles common tasks like:
/// - Dividing model parameters into chunks (sharding)
/// - Collecting all chunks when needed (gathering)
/// - Sharing learning updates across all processes (gradient sync)
/// - Saving and loading distributed models
///
/// Specific types of distributed models (like fully sharded or hybrid sharded) inherit
/// from this and add their own strategies. This prevents code duplication and ensures
/// all distributed models work consistently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public abstract class ShardedModelBase<T, TInput, TOutput> : IShardedModel<T, TInput, TOutput>
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The wrapped model that this sharded model delegates to.
    /// </summary>
    private readonly IFullModel<T, TInput, TOutput> _wrappedModel;

    /// <summary>
    /// The sharding configuration containing communication backend and settings.
    /// </summary>
    protected readonly IShardingConfiguration<T> Config;

    /// <summary>
    /// The local parameter shard owned by this process.
    /// </summary>
    protected Vector<T> LocalShard;

    /// <summary>
    /// Cached full parameters to avoid repeated gathering.
    /// </summary>
    protected Vector<T>? CachedFullParameters;

    /// <summary>
    /// Starting index of this process's shard in the full parameter vector.
    /// </summary>
    protected int ShardStartIndex;

    /// <summary>
    /// Size of this process's parameter shard.
    /// </summary>
    protected int ShardSize;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> WrappedModel => _wrappedModel;

    /// <summary>
    /// Protected access to wrapped model for derived classes.
    /// </summary>
    protected IFullModel<T, TInput, TOutput> WrappedModelInternal => _wrappedModel;

    /// <inheritdoc/>
    public int Rank => Config.CommunicationBackend.Rank;

    /// <inheritdoc/>
    public int WorldSize => Config.CommunicationBackend.WorldSize;

    /// <inheritdoc/>
    public Vector<T> LocalParameterShard => LocalShard;

    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration => Config;

    /// <inheritdoc/>
    public int ParameterCount => WrappedModel.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the ShardedModelBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor wraps an existing model with distributed training capabilities.
    /// It initializes the communication backend if needed and sets up parameter sharding.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor takes your regular model and makes it distributed.
    ///
    /// You provide:
    /// 1. The model you want to distribute
    /// 2. Configuration that tells us how to distribute it
    ///
    /// The constructor automatically:
    /// - Sets up communication if not already done
    /// - Splits the model's parameters across processes
    /// - Prepares everything for distributed training
    /// </para>
    /// </remarks>
    /// <param name="wrappedModel">The model to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if model or config is null</exception>
    protected ShardedModelBase(IFullModel<T, TInput, TOutput> wrappedModel, IShardingConfiguration<T> config)
    {
        _wrappedModel = wrappedModel ?? throw new ArgumentNullException(nameof(wrappedModel));
        Config = config ?? throw new ArgumentNullException(nameof(config));
        NumOps = MathHelper.GetNumericOperations<T>();

        // Initialize communication backend if needed
        if (!Config.CommunicationBackend.IsInitialized)
        {
            Config.CommunicationBackend.Initialize();
        }

        // Initialize sharding
        LocalShard = new Vector<T>(Array.Empty<T>());
        ShardStartIndex = 0;
        ShardSize = 0;
        CachedFullParameters = null;

        // Allow derived classes to set up state before sharding
        OnBeforeInitializeSharding();
        InitializeSharding();
    }

    /// <summary>
    /// Called before InitializeSharding to allow derived classes to set up state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Override this method in derived classes to initialize fields that are needed
    /// by InitializeSharding but cannot be set before the base constructor call.
    /// </para>
    /// </remarks>
    protected virtual void OnBeforeInitializeSharding()
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Initializes parameter sharding by dividing parameters across processes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method calculates how to distribute parameters evenly across all processes,
    /// with remainder parameters distributed to the first few processes.
    /// Derived classes can override this to implement different sharding strategies.
    /// </para>
    /// <para><b>For Beginners:</b> This splits the model's parameters across all processes.
    ///
    /// Think of it like dividing a deck of cards among players. If you have 10 parameters
    /// and 3 processes:
    /// - Process 0 gets parameters 0-3 (4 parameters)
    /// - Process 1 gets parameters 4-6 (3 parameters)
    /// - Process 2 gets parameters 7-9 (3 parameters)
    ///
    /// We try to split evenly, but if there's a remainder, the first processes get
    /// one extra parameter each.
    /// </para>
    /// </remarks>
    protected virtual void InitializeSharding()
    {
        var fullParameters = WrappedModel.GetParameters();
        int totalParams = fullParameters.Length;

        // Calculate shard size for this process
        int baseShardSize = totalParams / WorldSize;
        int remainder = totalParams % WorldSize;

        // Distribute remainder among first 'remainder' processes
        ShardSize = baseShardSize + (Rank < remainder ? 1 : 0);
        ShardStartIndex = Rank * baseShardSize + Math.Min(Rank, remainder);

        // Extract local shard
        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);

        // Invalidate cache
        CachedFullParameters = null;
    }

    /// <inheritdoc/>
    public virtual Vector<T> GatherFullParameters()
    {
        // Use cached version if available
        if (CachedFullParameters != null)
        {
            return CachedFullParameters;
        }

        // Gather parameters from all processes
        var gathered = Config.CommunicationBackend.AllGather(LocalShard);
        CachedFullParameters = gathered;
        return gathered;
    }

    /// <inheritdoc/>
    public virtual void SynchronizeGradients()
    {
        // Perform AllReduce with average operation
        Config.CommunicationBackend.AllReduce(LocalShard, ReductionOperation.Average);

        // Invalidate cached full parameters
        CachedFullParameters = null;
    }

    /// <summary>
    /// Invalidates the cached full parameters, forcing a re-gather on next access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method should be called whenever local parameters change to ensure
    /// the cache is refreshed on the next GatherFullParameters call.
    /// </para>
    /// <para><b>For Beginners:</b> When parameters change, we need to throw away
    /// the old cached full parameters.
    ///
    /// It's like when you update a document - you need to discard the old
    /// saved copy so that next time you need it, you get the updated version.
    /// </para>
    /// </remarks>
    protected void InvalidateCache()
    {
        CachedFullParameters = null;
    }

    /// <summary>
    /// Updates the local parameter shard from the full parameter vector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method extracts this process's shard from a full parameter vector.
    /// Used after training updates or when setting parameters.
    /// </para>
    /// <para><b>For Beginners:</b> After the full model is updated, we need to
    /// extract our piece of it.
    ///
    /// It's like taking your slice of a pizza after it's been prepared - you get
    /// the portion that belongs to you from the whole.
    /// </para>
    /// </remarks>
    /// <param name="fullParameters">The full parameter vector</param>
    protected void UpdateLocalShardFromFull(Vector<T> fullParameters)
    {
        var shardData = new T[ShardSize];
        Array.Copy(fullParameters.ToArray(), ShardStartIndex, shardData, 0, ShardSize);
        LocalShard = new Vector<T>(shardData);
        InvalidateCache();
    }

    /// <inheritdoc/>
    public abstract void Train(TInput input, TOutput expectedOutput);

    /// <inheritdoc/>
    public abstract TOutput Predict(TInput input);

    /// <inheritdoc/>
    public abstract ModelMetadata<T> GetModelMetadata();

    /// <inheritdoc/>
    public virtual Vector<T> GetParameters()
    {
        return GatherFullParameters();
    }

    /// <inheritdoc/>
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Parameter count mismatch. Expected {ParameterCount}, got {parameters.Length}.",
                nameof(parameters));
        }

        // Update local shard
        UpdateLocalShardFromFull(parameters);

        // Update wrapped model
        WrappedModel.SetParameters(parameters);
    }

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);

    /// <inheritdoc/>
    public abstract byte[] Serialize();

    /// <inheritdoc/>
    public abstract void Deserialize(byte[] data);

    /// <inheritdoc/>
    public abstract void SaveModel(string filePath);

    /// <inheritdoc/>
    public abstract void LoadModel(string filePath);

    /// <inheritdoc/>
    public abstract IFullModel<T, TInput, TOutput> Clone();

    /// <inheritdoc/>
    public virtual IFullModel<T, TInput, TOutput> DeepCopy()
    {
        return Clone();
    }

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        return WrappedModel.GetFeatureImportance();
    }

    /// <inheritdoc/>
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        return WrappedModel.GetActiveFeatureIndices();
    }

    /// <inheritdoc/>
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        WrappedModel.SetActiveFeatureIndices(featureIndices);
    }

    /// <inheritdoc/>
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        return WrappedModel.IsFeatureUsed(featureIndex);
    }

    /// <inheritdoc/>
    public virtual ILossFunction<T> DefaultLossFunction => WrappedModel.DefaultLossFunction;

    /// <inheritdoc/>
    public virtual Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        return WrappedModel.ComputeGradients(input, target, lossFunction);
    }

    /// <inheritdoc/>
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        WrappedModel.ApplyGradients(gradients, learningRate);
    }


    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this model currently supports JIT compilation.
    /// </summary>
    /// <value>True if the wrapped model supports JIT compilation, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// Sharded models delegate JIT compilation support to their wrapped model.
    /// JIT compilation is performed on the full model representation, not on individual shards.
    /// </para>
    /// <para><b>For Beginners:</b> Distributed models can be JIT compiled if the underlying model supports it.
    ///
    /// The sharding strategy (splitting parameters across processes) doesn't prevent JIT compilation.
    /// The JIT compiler works with the full computation graph, which is the same across all processes.
    /// Individual processes execute the same compiled code but operate on different parameter shards.
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation
    {
        get
        {
            if (WrappedModel is null || WrappedModel == null)
                return false;

            return WrappedModel.SupportsJitCompilation;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation by delegating to the wrapped model.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the model's prediction.</returns>
    /// <remarks>
    /// <para>
    /// Sharded models delegate graph export to their wrapped model.
    /// The computation graph represents the full model's forward pass, independent of parameter sharding.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a computation graph from the wrapped model.
    ///
    /// Even though parameters are distributed (sharded) across multiple processes:
    /// - The computation graph structure is the same for all processes
    /// - Each process compiles the same graph into fast code
    /// - The only difference is which parameter values each process uses
    ///
    /// This allows distributed models to benefit from JIT compilation while maintaining
    /// their distributed training capabilities.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when the wrapped model does not support JIT compilation.
    /// </exception>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (WrappedModel is null || WrappedModel == null)
            throw new InvalidOperationException(
                "Cannot export computation graph: Wrapped model is null.");

        if (!WrappedModel.SupportsJitCompilation)
            throw new NotSupportedException(
                $"The wrapped model of type {WrappedModel.GetType().Name} does not support JIT compilation. " +
                "JIT compilation availability depends on the wrapped model's capabilities.");

        return WrappedModel.ExportComputationGraph(inputNodes);
    }

    #endregion
    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    public virtual void SaveState(Stream stream)
    {
        WrappedModel.SaveState(stream);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public virtual void LoadState(Stream stream)
    {
        WrappedModel.LoadState(stream);
    }
}
