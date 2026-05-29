using OptimizerType = AiDotNet.Tensors.Engines.Compilation.OptimizerType;
using LrSchedule = AiDotNet.Tensors.Engines.Compilation.LrSchedule;

namespace AiDotNet.Optimizers.Fused;

/// <summary>
/// Describes how an optimizer maps onto the compiled fused-optimizer kernel:
/// which <see cref="OptimizerType"/> to run plus the baked hyperparameters and
/// optional fused LR schedule.
/// </summary>
/// <param name="Type">The fused kernel variant to dispatch (Adam, AdamW, AMSGrad, SGD).</param>
/// <param name="LearningRate">Current learning rate, baked into the plan.</param>
/// <param name="Beta1">Adam/AdamW first-moment decay (0 for SGD).</param>
/// <param name="Beta2">Adam/AdamW second-moment decay (0 for SGD).</param>
/// <param name="Epsilon">Denominator epsilon (0 for SGD).</param>
/// <param name="WeightDecay">Decoupled weight decay (AdamW); 0 otherwise.</param>
/// <param name="Schedule">Optional fused-side LR schedule, or null for constant LR.</param>
public readonly record struct FusedOptimizerConfig(
    OptimizerType Type,
    float LearningRate,
    float Beta1,
    float Beta2,
    float Epsilon,
    float WeightDecay,
    LrSchedule? Schedule);

/// <summary>
/// Implemented by optimizers that have a compiled fused-kernel equivalent, so the
/// fused-training dispatcher can ask the optimizer to describe itself instead of
/// switching on its concrete type.
/// </summary>
/// <remarks>
/// <para>
/// Open/closed-compliant by construction: having a fused SIMD kernel
/// (<c>FusedOptimizer.{SGD,Adam,AdamW,AMSGrad}UpdateSimd</c>) is intrinsic to an
/// optimizer, so the optimizer declares it. Only the optimizers that actually have
/// a kernel implement this interface — there is no central catalog and no
/// <c>OptimizerType is (… or … or …)</c> whitelist to keep in sync. An optimizer
/// without a fused kernel simply doesn't implement it and uses the eager tape;
/// adding a kernel later means implementing this interface, with no change to the
/// dispatcher. This is also why only a handful of the ~20 optimizers are
/// fuse-able: the rest have no SIMD kernel.
/// </para>
/// <para>
/// <see cref="TryGetFusedOptimizerConfig"/> returns <c>false</c> when THIS
/// instance is configured in a way the fused kernel can't reproduce (adaptive
/// learning rate, an unsupported LR-scheduler type, etc.), so a fuse-able
/// optimizer family can still fall back per-instance.
/// </para>
/// </remarks>
public interface IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this optimizer for the fused kernel, or returns <c>false</c> to
    /// fall back to the eager tape.
    /// </summary>
    bool TryGetFusedOptimizerConfig(out FusedOptimizerConfig config);
}
