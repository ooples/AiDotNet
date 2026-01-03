using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Defines the contract for models that support distributed training with parameter sharding.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b>
/// A sharded model is like having a team working on a large puzzle together.
/// Instead of one person holding all the puzzle pieces (parameters), each person
/// holds only a portion. When someone needs to see the full picture, everyone
/// shares their pieces (AllGather). When the team learns something new, everyone
/// combines their learnings (AllReduce).
/// </para>
/// <para>
/// This allows training models that are too large to fit on a single GPU or machine.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for operations</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public interface IShardedModel<T, TInput, TOutput> : IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the underlying wrapped model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is the original model that we're adding distributed training capabilities to.
    /// Think of it as the "core brain" that we're helping to work in a distributed way.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> WrappedModel { get; }

    /// <summary>
    /// Gets the rank of this process in the distributed group.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// Each process has a unique ID (rank). This tells you which process you are.
    /// Rank 0 is typically the "coordinator" process.
    /// </para>
    /// </remarks>
    int Rank { get; }

    /// <summary>
    /// Gets the total number of processes in the distributed group.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is how many processes are working together to train the model.
    /// For example, if you have 4 GPUs, WorldSize would be 4.
    /// </para>
    /// </remarks>
    int WorldSize { get; }

    /// <summary>
    /// Gets the portion of parameters owned by this process.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// This is "your piece of the puzzle" - the parameters that this particular
    /// process is responsible for storing and updating.
    /// </para>
    /// </remarks>
    Vector<T> LocalParameterShard { get; }

    /// <summary>
    /// Gets the full set of parameters by gathering from all processes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This operation involves communication across all processes.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking everyone to share their puzzle pieces so you can
    /// see the complete picture. It requires communication between all processes,
    /// so it's more expensive than just accessing LocalParameterShard.
    /// </para>
    /// </remarks>
    /// <returns>The complete set of parameters gathered from all processes</returns>
    Vector<T> GatherFullParameters();

    /// <summary>
    /// Synchronizes gradients across all processes using AllReduce.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After this operation, all processes have the same averaged gradients.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// During training, each process calculates gradients based on its portion
    /// of the data. This method combines (averages) those gradients so that
    /// everyone is learning from everyone else's experiences. It's like a team
    /// meeting where everyone shares what they learned.
    /// </para>
    /// </remarks>
    void SynchronizeGradients();

    /// <summary>
    /// Gets the configuration for this sharded model.
    /// </summary>
    IShardingConfiguration<T> ShardingConfiguration { get; }
}
