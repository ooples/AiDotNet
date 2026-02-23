using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;


namespace AiDotNet.DistributedTraining;

/// <summary>
/// Provides base implementation for distributed optimizers with parameter sharding.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class implements common functionality for all sharded optimizers,
/// including optimizer wrapping, parameter synchronization, consensus-based early stopping,
/// and serialization. Derived classes can customize the optimization strategy, implement
/// different sharding approaches (FSDP, ZeRO, etc.), or add optimizer-specific features.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all distributed optimizers build upon.
///
/// Think of this as a template for coordinating optimization across multiple computers or GPUs.
/// It handles common tasks like:
/// - Wrapping regular optimizers to work in distributed mode
/// - Syncing parameters across all processes after updates
/// - Making sure all processes agree on when to stop training
/// - Saving and loading distributed optimizer state
///
/// Specific types of distributed optimizers (like data-parallel or ZeRO) inherit from
/// this and add their own strategies. This prevents code duplication and ensures all
/// distributed optimizers work consistently.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public abstract class ShardedOptimizerBase<T, TInput, TOutput> : IShardedOptimizer<T, TInput, TOutput>
{
    /// <summary>
    /// Provides numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The wrapped optimizer that this sharded optimizer delegates to.
    /// </summary>
    private readonly IOptimizer<T, TInput, TOutput> _wrappedOptimizer;

    /// <summary>
    /// The sharding configuration containing communication backend and settings.
    /// </summary>
    protected readonly IShardingConfiguration<T> Config;

    /// <inheritdoc/>
    public IOptimizer<T, TInput, TOutput> WrappedOptimizer => _wrappedOptimizer;

    /// <summary>
    /// Protected access to wrapped optimizer for derived classes.
    /// </summary>
    protected IOptimizer<T, TInput, TOutput> WrappedOptimizerInternal => _wrappedOptimizer;

    /// <inheritdoc/>
    public int Rank => Config.CommunicationBackend.Rank;

    /// <inheritdoc/>
    public int WorldSize => Config.CommunicationBackend.WorldSize;

    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration => Config;

    /// <summary>
    /// Initializes a new instance of the ShardedOptimizerBase class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This constructor wraps an existing optimizer with distributed training capabilities.
    /// It initializes the communication backend if needed and prepares for distributed optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This constructor takes your regular optimizer and makes it distributed.
    ///
    /// You provide:
    /// 1. The optimizer you want to distribute (like Adam, SGD, etc.)
    /// 2. Configuration that tells us how to distribute it
    ///
    /// The constructor automatically:
    /// - Sets up communication if not already done
    /// - Prepares the optimizer for coordinated training
    /// - Ensures all processes can work together
    /// </para>
    /// </remarks>
    /// <param name="wrappedOptimizer">The optimizer to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    protected ShardedOptimizerBase(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
    {
        Guard.NotNull(wrappedOptimizer);
        _wrappedOptimizer = wrappedOptimizer;
        Guard.NotNull(config);
        Config = config;
        NumOps = MathHelper.GetNumericOperations<T>();

        // Initialize communication backend if needed
        if (!Config.CommunicationBackend.IsInitialized)
        {
            Config.CommunicationBackend.Initialize();
        }
    }

    /// <inheritdoc/>
    public abstract OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData);

    /// <inheritdoc/>
    public abstract void SynchronizeOptimizerState();

    /// <summary>
    /// Synchronizes model parameters across all processes using AllReduce with averaging.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method averages parameters across all processes, ensuring consistency.
    /// It's called after optimization steps to keep all processes synchronized.
    /// </para>
    /// <para><b>For Beginners:</b> After each process updates its model, we need to
    /// make sure everyone has the same parameters.
    ///
    /// This method averages the parameters from all processes. For example, if GPU 0
    /// calculated parameter value 1.0 and GPU 1 calculated 1.2, after sync both will
    /// have 1.1 (the average).
    /// </para>
    /// </remarks>
    /// <param name="model">The model whose parameters to synchronize</param>
    protected virtual void SynchronizeParameters(IFullModel<T, TInput, TOutput>? model)
    {
        if (model == null)
        {
            return;
        }

        // Don't sync if it's already a sharded model (handles its own sync)
        if (model is IShardedModel<T, TInput, TOutput>)
        {
            return;
        }

        // Get current parameters
        var parameters = model.GetParameters();

        // Average parameters across all processes
        Config.CommunicationBackend.AllReduce(parameters, ReductionOperation.Average);

        // Update model with averaged parameters
        model.SetParameters(parameters);
    }

    /// <inheritdoc/>
    public virtual bool ShouldEarlyStop()
    {
        // Delegate to wrapped optimizer
        bool localDecision = WrappedOptimizer.ShouldEarlyStop();

        // In distributed training, we need consensus on early stopping
        // All processes should agree to stop, otherwise some might continue while others stop
        // For now, we'll use a simple approach: if any process wants to stop, all stop

        // Create a vector with the local decision (1 for stop, 0 for continue)
        var decision = new Vector<T>(new[] { localDecision ? NumOps.One : NumOps.Zero });

        // Get the maximum across all processes
        // If any process returns 1 (stop), the max will be 1
        Config.CommunicationBackend.AllReduce(decision, ReductionOperation.Max);

        // Check if the result indicates stopping
        return !NumOps.Equals(decision[0], NumOps.Zero);
    }

    /// <inheritdoc/>
    public virtual OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return WrappedOptimizer.GetOptions();
    }

    /// <summary>
    /// Gets the gradients computed during the last optimization step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sharded optimizers delegate gradient access to the wrapped optimizer.
    /// If the wrapped optimizer is gradient-based, this will return the actual computed gradients.
    /// Otherwise, it returns an empty vector.
    /// </para>
    /// </remarks>
    public virtual Vector<T> LastComputedGradients
    {
        get
        {
            var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
            return gradientOptimizer?.LastComputedGradients ?? Vector<T>.Empty();
        }
    }

    /// <summary>
    /// Applies pre-computed gradients to a model's parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply</param>
    /// <param name="model">The model to update</param>
    /// <returns>The updated model</returns>
    /// <remarks>
    /// <para>
    /// Sharded optimizers delegate gradient application to the wrapped optimizer.
    /// If the wrapped optimizer is gradient-based, this will apply the gradients.
    /// Otherwise, throws NotSupportedException.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">If the wrapped optimizer is not gradient-based</exception>
    public virtual IFullModel<T, TInput, TOutput> ApplyGradients(Vector<T> gradients, IFullModel<T, TInput, TOutput> model)
    {
        var gradientOptimizer = WrappedOptimizer as IGradientBasedOptimizer<T, TInput, TOutput>;
        if (gradientOptimizer == null)
        {
            throw new NotSupportedException(
                $"ApplyGradients requires a gradient-based optimizer, but wrapped optimizer {WrappedOptimizer.GetType().Name} does not implement IGradientBasedOptimizer.");
        }

        return gradientOptimizer.ApplyGradients(gradients, model);
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        // Delegate reset to wrapped optimizer
        WrappedOptimizer.Reset();
    }

    /// <inheritdoc/>
    public virtual void SetModel(IFullModel<T, TInput, TOutput> model)
    {
        // Delegate to wrapped optimizer
        WrappedOptimizer.SetModel(model);
    }

    /// <inheritdoc/>
    public abstract byte[] Serialize();

    /// <inheritdoc/>
    public abstract void Deserialize(byte[] data);

    /// <inheritdoc/>
    public virtual void SaveModel(string filePath)
    {
        // Only rank 0 saves to avoid conflicts
        if (Rank == 0)
        {
            var data = Serialize();
            File.WriteAllBytes(filePath, data);
        }

        // Wait for rank 0 to finish writing
        Config.CommunicationBackend.Barrier();
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        // All processes read the same file
        var data = File.ReadAllBytes(filePath);
        Deserialize(data);

        // Ensure all processes finish loading
        Config.CommunicationBackend.Barrier();
    }
}
