using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements a distributed optimizer wrapper that coordinates optimization across multiple processes.
///
/// For Beginners:
/// This class wraps any existing optimizer (like Adam, SGD, etc.) and makes it work across
/// multiple GPUs or machines. It automatically handles:
/// - Synchronizing gradients across all processes
/// - Coordinating parameter updates
/// - Ensuring all processes stay in sync
///
/// Think of it like a team of coaches working together - each has their own expertise
/// (the wrapped optimizer), but they coordinate their efforts to train the team effectively.
///
/// Example:
/// <code>
/// // Original optimizer
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
///
/// // Wrap it for distributed training
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
/// var distributedOptimizer = new ShardedOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config);
///
/// // Now optimize as usual - distributed magic happens automatically!
/// var result = distributedOptimizer.Optimize(inputData);
/// </code>
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ShardedOptimizer<T, TInput, TOutput> : IShardedOptimizer<T, TInput, TOutput> where T : struct
{
    private readonly IOptimizer<T, TInput, TOutput> _wrappedOptimizer;
    private readonly IShardingConfiguration<T> _config;

    /// <inheritdoc/>
    public IOptimizer<T, TInput, TOutput> WrappedOptimizer => _wrappedOptimizer;

    /// <inheritdoc/>
    public int Rank => _config.CommunicationBackend.Rank;

    /// <inheritdoc/>
    public int WorldSize => _config.CommunicationBackend.WorldSize;

    /// <inheritdoc/>
    public IShardingConfiguration<T> ShardingConfiguration => _config;

    /// <summary>
    /// Creates a new sharded optimizer wrapping an existing optimizer.
    ///
    /// For Beginners:
    /// This constructor takes your existing optimizer and makes it distributed.
    /// You provide:
    /// 1. The optimizer you want to make distributed
    /// 2. A configuration that tells us how to do the distribution
    ///
    /// The optimizer will automatically synchronize across all processes during optimization.
    /// </summary>
    /// <param name="wrappedOptimizer">The optimizer to wrap with distributed capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <exception cref="ArgumentNullException">Thrown if optimizer or config is null</exception>
    public ShardedOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
    {
        _wrappedOptimizer = wrappedOptimizer ?? throw new ArgumentNullException(nameof(wrappedOptimizer));
        _config = config ?? throw new ArgumentNullException(nameof(config));

        // Initialize the communication backend if not already done
        if (!_config.CommunicationBackend.IsInitialized)
        {
            _config.CommunicationBackend.Initialize();
        }
    }

    /// <inheritdoc/>
    public OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
        {
            throw new ArgumentNullException(nameof(inputData));
        }

        // Ensure all processes start together
        _config.CommunicationBackend.Barrier();

        // Perform optimization on the wrapped optimizer
        var result = _wrappedOptimizer.Optimize(inputData);

        // Synchronize parameters across all processes if auto-sync is enabled
        if (_config.AutoSyncGradients && result.BestSolution != null)
        {
            SynchronizeParameters(result.BestSolution);
        }

        // Synchronize optimizer state if needed
        SynchronizeOptimizerState();

        // Ensure all processes finish together
        _config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public void SynchronizeOptimizerState()
    {
        // For now, this is a placeholder
        // In a full implementation, we would synchronize optimizer-specific state
        // like momentum buffers, variance estimates (for Adam), etc.

        // Different optimizers have different state to sync:
        // - SGD with momentum: velocity vectors
        // - Adam: first and second moment estimates
        // - RMSprop: squared gradient moving average

        // This would require either:
        // 1. Extending IOptimizer with state access methods
        // 2. Type-specific handling for known optimizer types
        // 3. A generic state serialization mechanism

        // For the MVP, we assume stateless or that the wrapped optimizer handles its own state
    }

    /// <summary>
    /// Synchronizes model parameters across all processes.
    ///
    /// For Beginners:
    /// After optimization, each process might have slightly different parameters
    /// (if they processed different data). This method averages the parameters
    /// across all processes so everyone has the same model.
    /// </summary>
    /// <param name="model">The model whose parameters to synchronize</param>
    private void SynchronizeParameters(IFullModel<T, TInput, TOutput>? model)
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
        _config.CommunicationBackend.AllReduce(parameters, ReductionOperation.Average);

        // Update model with averaged parameters
        model.SetParameters(parameters);
    }

    /// <inheritdoc/>
    public bool ShouldEarlyStop()
    {
        // Delegate to wrapped optimizer
        bool localDecision = _wrappedOptimizer.ShouldEarlyStop();

        // In distributed training, we need consensus on early stopping
        // All processes should agree to stop, otherwise some might continue while others stop
        // For now, we'll use a simple approach: if any process wants to stop, all stop

        // Create a vector with the local decision (1 for stop, 0 for continue)
        var decision = new Vector<T>(new[] { localDecision ? MathHelper.GetNumericOperations<T>().One : MathHelper.GetNumericOperations<T>().Zero });

        // Get the maximum across all processes
        // If any process returns 1 (stop), the max will be 1
        _config.CommunicationBackend.AllReduce(decision, ReductionOperation.Max);

        // Check if the result indicates stopping
        var numOps = MathHelper.GetNumericOperations<T>();
        return !numOps.Equals(decision[0], numOps.Zero);
    }

    /// <inheritdoc/>
    public OptimizationAlgorithmOptions<T, TInput, TOutput> GetOptions()
    {
        return _wrappedOptimizer.GetOptions();
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Serialize sharding configuration info
        writer.Write(WorldSize);
        writer.Write(Rank);
        writer.Write(_config.AutoSyncGradients);
        writer.Write(_config.MinimumParameterGroupSize);
        writer.Write(_config.EnableGradientCompression);

        // Serialize wrapped optimizer
        var optimizerData = _wrappedOptimizer.Serialize();
        writer.Write(optimizerData.Length);
        writer.Write(optimizerData);

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
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
                $"World size mismatch. Optimizer was saved with {savedWorldSize} processes, " +
                $"but current configuration has {WorldSize} processes.");
        }

        // Validate rank matches - different rank could indicate configuration mismatch
        if (savedRank != Rank)
        {
            throw new InvalidOperationException(
                $"Rank mismatch. Optimizer was saved on rank {savedRank}, " +
                $"but is being loaded on rank {Rank}. This could indicate a configuration error.");
        }

        // Read wrapped optimizer
        int optimizerDataLength = reader.ReadInt32();
        byte[] optimizerData = reader.ReadBytes(optimizerDataLength);
        _wrappedOptimizer.Deserialize(optimizerData);
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        // Barrier before rank check to prevent deadlock if rank 0 fails
        _config.CommunicationBackend.Barrier();

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
            _config.CommunicationBackend.Barrier();
        }
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        // Barrier before loading to ensure all processes start together
        _config.CommunicationBackend.Barrier();

        try
        {
            // All processes read the same file (read-only, no conflicts)
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        finally
        {
            // Ensure all processes finish loading before proceeding
            _config.CommunicationBackend.Barrier();
        }
    }
}
