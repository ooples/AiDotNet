using AiDotNet.Interfaces;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Defines the contract for optimizers that support distributed training with parameter sharding.
///
/// For Beginners:
/// A sharded optimizer is like having a team of coaches working together.
/// Each coach (process) is responsible for updating a portion of the player's (model's) skills.
/// After each round of practice, the coaches share and combine their improvements to ensure
/// everyone stays in sync.
///
/// This allows optimizing very large models that don't fit on a single GPU.
/// </summary>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public interface IShardedOptimizer<T, TInput, TOutput> : IOptimizer<T, TInput, TOutput> where T : struct
{
    /// <summary>
    /// Gets the underlying wrapped optimizer.
    ///
    /// For Beginners:
    /// This is the original optimizer (like Adam, SGD, etc.) that we're adding
    /// distributed training capabilities to. Think of it as the "core brain" that
    /// we're helping to work across multiple processes.
    /// </summary>
    IOptimizer<T, TInput, TOutput> WrappedOptimizer { get; }

    /// <summary>
    /// Gets the rank of this process in the distributed group.
    ///
    /// For Beginners:
    /// Each process has a unique ID (rank). This tells you which process you are.
    /// Rank 0 is typically the "coordinator" process.
    /// </summary>
    int Rank { get; }

    /// <summary>
    /// Gets the total number of processes in the distributed group.
    ///
    /// For Beginners:
    /// This is how many processes are working together to optimize the model.
    /// For example, if you have 4 GPUs, WorldSize would be 4.
    /// </summary>
    int WorldSize { get; }

    /// <summary>
    /// Gets the sharding configuration for this optimizer.
    /// </summary>
    IShardingConfiguration<T> ShardingConfiguration { get; }

    /// <summary>
    /// Synchronizes optimizer state (like momentum buffers) across all processes.
    ///
    /// For Beginners:
    /// Some optimizers (like Adam) keep track of past gradients to make smarter updates.
    /// This method makes sure all processes have the same optimizer state, so they stay
    /// coordinated. It's like making sure all team members are reading from the same playbook.
    /// </summary>
    void SynchronizeOptimizerState();
}
