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
internal readonly record struct FusedOptimizerConfig(
    OptimizerType Type,
    float LearningRate,
    float Beta1,
    float Beta2,
    float Epsilon,
    float WeightDecay,
    LrSchedule? Schedule)
{
    /// <summary>
    /// When true, request bfloat16 storage for the fused Adam/AdamW moment buffers
    /// (#1745) — half the optimizer-state footprint, same fp32 update math. Honored
    /// only by the CPU float Adam/AdamW fused kernel; a safe no-op otherwise.
    /// <para>
    /// Init-only property rather than a primary-constructor component so adding it did NOT change the
    /// record's <c>Deconstruct(...)</c> arity or force existing positional construction sites to add an
    /// argument (only the one call that sets it uses object-initializer syntax). It still participates in
    /// the record's value equality/hash, which is correct — two configs differing only in moment storage
    /// are genuinely distinct.
    /// </para>
    /// </summary>
    public bool UseBf16Moments { get; init; }

    /// <summary>
    /// Optimizer-specific coefficients for the kernels that do not read them from the beta/epsilon
    /// slots: LARS (momentum, trust coefficient), FTRL (L1, L2, lr power), ASGD (lambd, alpha, t0) and
    /// Rprop (eta+/eta-, step bounds). Null for every other kernel, which needs none.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Init-only for the same reason as <see cref="UseBf16Moments"/>: adding it does not change the
    /// record's <c>Deconstruct</c> arity or force existing positional construction sites to pass an
    /// extra argument.
    /// </para>
    /// <para>
    /// Deliberately NOT defaulted to a fresh instance. <c>FusedOptimizerExtras</c> carries non-zero
    /// defaults (LARS trust coefficient 0.001, Rprop step bounds, ASGD t0 1e6), so handing the plan a
    /// default-constructed one for an optimizer that has no extras would silently feed it another
    /// optimizer's constants. Null means "this kernel needs none".
    /// </para>
    /// </remarks>
    public AiDotNet.Tensors.Engines.Compilation.FusedOptimizerExtras? Extras { get; init; }
}

/// <summary>
/// Implemented by optimizers that have a compiled fused-kernel equivalent, so the
/// fused-training dispatcher can ask the optimizer to describe itself instead of
/// switching on its concrete type.
/// </summary>
/// <remarks>
/// <para>
/// Open/closed-compliant by construction: having a fused SIMD kernel is intrinsic to an
/// optimizer, so the optimizer declares it. Only the optimizers that actually have
/// a kernel implement this interface — there is no central catalog and no
/// <c>OptimizerType is (… or … or …)</c> whitelist to keep in sync. An optimizer
/// without a fused kernel simply doesn't implement it and uses the eager tape;
/// adding a kernel later means implementing this interface, with no change to the
/// dispatcher.
/// </para>
/// <para>
/// <b>Which optimizers can fuse.</b> Do not guess from a list here — <c>CompiledTrainingPlan</c> in
/// AiDotNet.Tensors is the authority, and a type is fuse-able only if that plan has a
/// <c>case OptimizerType.X</c> for it. As of Tensors 0.127 the plan dispatches: SGD, SGDMomentum,
/// Adam, AdamW, Adagrad, RMSprop, Lion, AdaMax, AMSGrad, Nadam, AdaDelta, LARS, LAMB, FTRL, RAdam,
/// SparseAdam, ASGD, Rprop, LBFGS, HypergradientSGD, ScheduleFreeSGD, DAdaptationSGD, ProximalL1,
/// TrustRegion and ConjugateGradient.
/// </para>
/// <para>
/// <b>"Cannot fuse" is a claim about today, and it keeps turning out to be wrong.</b> An earlier
/// version of this remark said only SGD/Adam/AdamW/AMSGrad had kernels and that "the rest have no SIMD
/// kernel". That was already false when written, and it caused issue #1930 to be scoped as "write new
/// SIMD kernels" when the kernels existed all along and the real gap was optimizers not implementing
/// this interface. The revision after it then asserted that BFGS, LBFGS, DFP, Newton,
/// LevenbergMarquardt, TrustRegion, ConjugateGradient, CoordinateDescent, ADMM and
/// ProximalGradientDescent "genuinely have no fused equivalent" — and five of those ten now fuse,
/// because the missing piece was a kernel nobody had written yet rather than a mathematical obstacle.
/// </para>
/// <para>
/// So: an optimizer not implementing this interface means only that no mapping has been established
/// yet. When one is missing, the question to ask is "what would the kernel have to do?", and the
/// answer generally belongs in a Tensors PR. Three distinct reasons an optimizer legitimately does not
/// implement it, which are worth telling apart:
/// </para>
/// <list type="number">
/// <item><description>
/// <b>The update needs information the fused path does not have.</b> Newton and LevenbergMarquardt
/// need to re-evaluate the model (a Hessian, a residual Jacobian) rather than just transform the
/// gradient. Their <c>Step</c> throws rather than pretending; no per-element kernel can help.
/// </description></item>
/// <item><description>
/// <b>The state is not per-element.</b> BFGS and DFP carry a dense n×n inverse-Hessian approximation,
/// which is why the field's own answer at scale is L-BFGS — the compact representation of Byrd,
/// Nocedal &amp; Schnabel (1994) — and L-BFGS does fuse. PyTorch ships only <c>LBFGS</c> from this
/// family and documents it as supporting neither <c>foreach</c> nor <c>fused</c>.
/// </description></item>
/// <item><description>
/// <b>The kernel exists but disagrees with the eager path.</b> This is the one that must never be
/// papered over by fusing anyway: it produces two different trainings of the same configuration
/// depending on which path is taken, and nothing errors. Fix the kernel, then wire the spec.
/// </description></item>
/// </list>
/// <para>
/// <b>Optimizer-specific parameters.</b> LARS, FTRL, ASGD and Rprop read their coefficients from
/// <c>FusedOptimizerExtras</c> rather than from the beta/epsilon slots on
/// <see cref="FusedOptimizerConfig"/>, so they additionally require that channel to be populated.
/// SGDMomentum is the exception needing no extras: the plan reads its momentum coefficient from
/// <see cref="FusedOptimizerConfig.Beta1"/>.
/// </para>
/// <para>
/// <see cref="TryGetFusedOptimizerConfig"/> returns <c>false</c> when THIS
/// instance is configured in a way the fused kernel can't reproduce (adaptive
/// learning rate, an unsupported LR-scheduler type, etc.), so a fuse-able
/// optimizer family can still fall back per-instance.
/// </para>
/// </remarks>
internal interface IFusedOptimizerSpec
{
    /// <summary>
    /// Describes this optimizer for the fused kernel, or returns <c>false</c> to
    /// fall back to the eager tape.
    /// </summary>
    bool TryGetFusedOptimizerConfig(out FusedOptimizerConfig config);
}
