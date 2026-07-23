using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Inputs;
using AiDotNet.Optimizers;

namespace AiDotNet.DistributedTraining;

/// <summary>
/// Implements ZeRO Stage 3 optimizer - full sharding equivalent to FSDP.
/// </summary>
/// <remarks>
/// <para><b>Strategy Overview:</b>
/// ZeRO-3 is equivalent to FSDP optimizer - full sharding of parameters, gradients, and optimizer
/// states. This class is an alias to FSDPOptimizer for ZeRO terminology consistency.
/// </para>
/// <para><b>For Beginners:</b>
/// ZeRO-3 and FSDP optimizers are the same thing. Use whichever name you prefer.
/// Everything is sharded for maximum memory efficiency.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type</typeparam>
/// <typeparam name="TInput">The input type for the model</typeparam>
/// <typeparam name="TOutput">The output type for the model</typeparam>
public class ZeRO3Optimizer<T, TInput, TOutput> : FSDPOptimizer<T, TInput, TOutput>
{
    public ZeRO3Optimizer(
        IOptimizer<T, TInput, TOutput> wrappedOptimizer,
        IShardingConfiguration<T> config)
        : base(wrappedOptimizer, config)
    {
    }
}
