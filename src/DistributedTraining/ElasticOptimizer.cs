using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements Elastic optimizer - supports dynamic worker addition/removal during training.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// Elastic training (TorchElastic, Horovod Elastic) enables dynamic scaling of workers during
/// training. Workers can be added or removed without stopping the training job, supporting:
/// - Fault tolerance: Replace failed workers automatically
/// - Auto-scaling: Add workers during peak hours, remove during off-peak
/// - Spot instance usage: Tolerate preemptions, use cheaper compute
///
/// When world size changes, the optimizer handles re-sharding parameters and optimizer states
/// across the new worker set. This requires checkpointing and careful state management.
/// </para>
/// <para><b>For Beginners:</b>
/// Elastic training is like having a flexible team size. Workers can join or leave during
/// training without stopping everything:
///
/// Scenario 1 - Fault tolerance:
/// - Start with 8 GPUs training your model
/// - GPU 3 fails → automatically detected
/// - Training continues with 7 GPUs (parameters redistributed)
/// - New GPU joins → training scales back to 8 GPUs
///
/// Scenario 2 - Cloud cost optimization:
/// - Use cheap "spot instances" that can be taken away anytime
/// - When instance is preempted, training continues with remaining workers
/// - New instance joins when available
///
/// This is critical for long training jobs where failures are expected.
/// </para>
/// <para><b>Use Cases:</b>
/// - Long training jobs (days/weeks) where failures will occur
/// - Cloud training with spot/preemptible instances (save 60-90% cost)
/// - Auto-scaling based on load or time of day
/// - Fault tolerance for production training pipelines
/// </para>
/// <para><b>Trade-offs:</b>
/// - Memory: Must handle dynamic re-sharding
/// - Communication: Overhead during worker changes (re-sharding, sync)
/// - Complexity: Very High - requires membership management, state re-distribution
/// - Convergence: Learning rate scheduling must account for dynamic world size
/// - Best for: Long jobs, cost-sensitive scenarios, production ML pipelines
/// - Limitation: Worker changes create temporary slowdown during re-sharding
/// </para>
/// <para><b>Implementation Note:</b>
/// This framework provides elastic optimizer infrastructure. Full production deployment
/// requires:
/// 1. Membership/discovery service (etcd, ZooKeeper, or cloud-native)
/// 2. Automatic checkpointing before worker changes
/// 3. State re-sharding algorithms
/// 4. Rendezvous mechanism for worker coordination
/// This implementation demonstrates the elastic pattern.
/// </para>
/// <para>
/// Example:
/// <code>
/// var optimizer = new AdamOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(model, options);
/// var backend = new InMemoryCommunicationBackend&lt;double&gt;(rank: 0, worldSize: 4);
/// var config = new ShardingConfiguration&lt;double&gt;(backend);
///
/// var elasticOptimizer = new ElasticOptimizer&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     optimizer, config,
///     minWorkers: 2,   // Can run with as few as 2 workers
///     maxWorkers: 16); // Can scale up to 16 workers
///
/// // Training continues through worker changes:
/// // 4 workers → 3 workers (one fails) → 5 workers (two join) → ...
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ElasticOptimizer<T, TInput, TOutput> : ShardedOptimizerBase<T, TInput, TOutput>
{
    private readonly int _minWorkers;
    private readonly int _maxWorkers;
    private int _currentWorldSize;

    /// <summary>
    /// Creates an elastic optimizer.
    /// </summary>
    /// <param name="wrappedOptimizer">The optimizer to wrap with elastic capabilities</param>
    /// <param name="config">Configuration for sharding and communication</param>
    /// <param name="minWorkers">Minimum number of workers (default: 1)</param>
    /// <param name="maxWorkers">Maximum number of workers (default: 1024)</param>
    public ElasticOptimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config,
        int minWorkers = 1,
        int maxWorkers = 1024)
        : base(wrappedOptimizer, config)
    {
        if (minWorkers < 1)
            throw new ArgumentException("Minimum workers must be at least 1", nameof(minWorkers));
        if (maxWorkers < minWorkers)
            throw new ArgumentException("Maximum workers must be >= minimum workers", nameof(maxWorkers));

        _minWorkers = minWorkers;
        _maxWorkers = maxWorkers;
        _currentWorldSize = WorldSize;
    }

    /// <inheritdoc/>
    public override OptimizationResult<T, TInput, TOutput> Optimize(OptimizationInputData<T, TInput, TOutput> inputData)
    {
        if (inputData == null)
            throw new ArgumentNullException(nameof(inputData));

        // Check for world size changes (workers joined/left)
        if (DetectWorldSizeChange())
        {
            HandleWorkerChange();
        }

        // Barrier with current worker set
        Config.CommunicationBackend.Barrier();

        // Optimize with current workers
        var result = WrappedOptimizer.Optimize(inputData);

        // Synchronize parameters
        if (Config.AutoSyncGradients && result.BestSolution != null)
        {
            SynchronizeParameters(result.BestSolution);
        }

        Config.CommunicationBackend.Barrier();

        return result;
    }

    /// <inheritdoc/>
    public override void SynchronizeOptimizerState()
    {
        // When world size changes, optimizer states must be re-sharded
        // This is handled in HandleWorkerChange()
    }

    /// <summary>
    /// Detects if the world size has changed.
    /// </summary>
    private bool DetectWorldSizeChange()
    {
        // In production, this would:
        // 1. Query membership service (etcd, etc.)
        // 2. Detect if workers joined or left
        // 3. Trigger rendezvous if change detected

        // Framework placeholder
        int newWorldSize = Config.CommunicationBackend.WorldSize;
        return newWorldSize != _currentWorldSize;
    }

    /// <summary>
    /// Handles worker addition or removal.
    /// </summary>
    private void HandleWorkerChange()
    {
        int newWorldSize = Config.CommunicationBackend.WorldSize;

        // Validate new world size is within bounds
        if (newWorldSize < _minWorkers || newWorldSize > _maxWorkers)
        {
            throw new InvalidOperationException(
                $"World size {newWorldSize} outside allowed range [{_minWorkers}, {_maxWorkers}]");
        }

        // In production elastic training:
        // 1. Checkpoint current state
        // 2. All workers synchronize at rendezvous point
        // 3. Re-shard parameters and optimizer states across new worker set
        // 4. Broadcast/scatter state to new workers
        // 5. Resume training with new configuration

        // Framework placeholder for re-sharding logic
        _currentWorldSize = newWorldSize;

        // Would call: ReshardParameters() and ReshardOptimizerState()
    }

    /// <summary>
    /// Gets the current number of active workers.
    /// </summary>
    public int CurrentWorkers => _currentWorldSize;

    /// <summary>
    /// Gets whether the optimizer can accept more workers.
    /// </summary>
    public bool CanScaleUp => _currentWorldSize < _maxWorkers;

    /// <summary>
    /// Gets whether the optimizer can tolerate losing workers.
    /// </summary>
    public bool CanScaleDown => _currentWorldSize > _minWorkers;
}
