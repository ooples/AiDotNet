using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>How a committed fused plan's failed step is recovered from (see <see cref="FusedTrainingSession{T}.DropAfterCommittedFailure"/>).</summary>
internal enum CommittedFailureKind
{
    /// <summary>The device ran out of memory.</summary>
    DeviceOutOfMemory,

    /// <summary>A transient device fault: driver, stream, launch, or a buffer released under the step.</summary>
    DeviceTransient,
}

/// <summary>What one <see cref="FusedTrainingSession{T}.TryStep"/> call did.</summary>
internal enum FusedStepOutcome
{
    /// <summary>The compiled plan ran forward, backward and the optimizer update; the step is done.</summary>
    Stepped,

    /// <summary>The fused path does not apply (or proved unsafe and was reset); run the eager tape step.</summary>
    Declined,

    /// <summary>
    /// The plan already trained earlier steps but cannot run this one. Its optimizer moments live in the plan and
    /// cannot be handed to the eager optimizer, so a silent fallback would change the trajectory; the caller
    /// decides (see <see cref="FusedTrainingSession{T}.LastFallbackException"/>).
    /// </summary>
    CommittedFailure,
}

/// <summary>Whether a caller explicitly permits this closure step to use eager gradients with its configured optimizer.</summary>
internal enum ClosureGraphBreakPolicy
{
    Disallow,
    UseConfiguredOptimizer
}


/// <summary>One training step as the fused session sees it: what to train, on what, and how.</summary>
internal sealed class FusedTrainingStepRequest<T>
{
    public required IReadOnlyList<ITrainableLayer<T>> Layers { get; init; }

    /// <summary>
    /// Collects the trainable layers again, in the same order as <see cref="Layers"/>, once the step's
    /// lazy-initialization forward has run. Null = <see cref="Layers"/> is already final.
    /// </summary>
    /// <remarks>
    /// A composite layer whose shape is unknown until its first input builds its children on that forward, so the set
    /// collected before the step is short. Trained from the short set, the step remembered it, the next step arrived
    /// with the full set, and the difference read as a different model: the first step's plan and its optimizer moments
    /// were dropped and a second plan compiled.
    /// </remarks>
    public Func<IReadOnlyList<ITrainableLayer<T>>>? LayersAfterLazyInitialization { get; init; }
    public required Tensor<T> Input { get; init; }
    public required Tensor<T> Target { get; init; }
    public required Func<Tensor<T>, Tensor<T>> Forward { get; init; }
    public required Func<Tensor<T>, Tensor<T>, Tensor<T>> ComputeLoss { get; init; }
    public required IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer { get; init; }

    /// <summary>Trainable tensors not owned by any layer in <see cref="Layers"/>.</summary>
    public IReadOnlyList<Tensor<T>>? ExtraParameters { get; init; }

    /// <summary>The subset to optimize when the model freezes part of itself; null = everything.</summary>
    public IReadOnlyList<Tensor<T>>? Selection { get; init; }

    /// <summary>The model's own global-norm clip (0 = none); combined with the optimizer's by taking the smaller.</summary>
    public double ModelGradientClip { get; init; }

    /// <summary>
    /// The parameters <see cref="ModelGradientClip"/> covers when it does not cover every trained one; null = all.
    /// </summary>
    /// <remarks>A compiled plan clips one global norm, so a request with a scope is not fused (its
    /// <see cref="GraphBreakReason"/> says so). The eager step it falls back to takes the norm over these parameters'
    /// gradients and scales only them; the others reach the optimizer unclipped by the model.</remarks>
    public IReadOnlyList<Tensor<T>>? ModelGradientClipScope { get; init; }

    /// <summary>
    /// Whether the model excludes some of its parameters from the optimizer's L2 term (coupled weight decay); asked only
    /// when the optimizer regularizes with L2. Null = none excluded.
    /// </summary>
    public Func<bool>? ExcludesParametersFromWeightDecay { get; init; }

    /// <summary>Receives the step's gradients while their buffers are valid (the model's public gradient surface).</summary>
    public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnGradients { get; init; }

    /// <summary>
    /// Receives the gradients, before clipping, when the step runs on the eager tape instead
    /// (<see cref="TapeTrainingStepper{T}.EagerStep"/>). Null = <see cref="OnGradients"/>.
    /// </summary>
    /// <remarks>
    /// These are the tape's own tensors, not a plan's persistent buffers: the step clips them in place right after this
    /// call, and their arena storage is reissued a step later. A receiver must copy what it keeps, so one that keeps a
    /// plan's buffers by reference (as <see cref="OnGradients"/> may) needs this separate callback.
    /// </remarks>
    public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnEagerGradients { get; init; }

    /// <summary>
    /// Adds the optimizer's regularization to the gradients when the step runs eager, before clipping, as the fused plan
    /// adds its L2. Null = the optimizer's plain regularization on every reached parameter.
    /// </summary>
    public Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? RegularizeGradients { get; init; }

    /// <summary>
    /// Why this step's forward cannot be compiled once and replayed, or null when it can. A non-null reason keeps the
    /// step on the eager tape: the same graph break <c>torch.compile</c> takes for a dynamic region.
    /// </summary>
    /// <remarks>
    /// A compiled plan replays the graph its first step traced, refreshing only the input, the target, the
    /// parameters and the per-step random tensors declared through <see cref="CompiledStepRandom{T}"/>. A forward
    /// that reads a tensor's values on the host (a data-dependent branch, an index computed from the batch, a matching
    /// step in a detection loss) or that captures other per-step state would replay the first step's decisions
    /// forever. Such a model, or its base class, sets this reason instead of training on a frozen objective.
    /// </remarks>
    public string? GraphBreakReason { get; init; }

    /// <summary>Default strict. An allowed break also requires a nonempty GraphBreakReason on that exact step.</summary>
    public ClosureGraphBreakPolicy ClosureGraphBreakPolicy { get; init; }

    /// <summary>
    /// Check, on the plan's traced step and on its first replay with new data, that the compiled plan computes the
    /// same loss as the eager forward on that data; on disagreement everything the plan's steps changed is put back
    /// and the model stays on the eager tape.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For base classes whose models' training forwards were never audited for replay (a host-side read of a tensor
    /// value is frozen into the trace; see <see cref="GraphBreakReason"/>). The traced step is checked too because a
    /// plan can be wrong there alone: a recorded node that reads a value the trace has not computed yet is right on
    /// every replay. A forward with its own randomness (dropout) disagrees by construction and conservatively stays
    /// eager.
    /// </para>
    /// <para>
    /// What is put back: the trained tensors, the step state (<see cref="StepState"/> and the layers' registered
    /// buffers: running statistics, EMA codebooks), the optimizer's step position, and the per-step draws and seed
    /// counters taken through <see cref="CompiledStepRandom{T}"/>, which the eager steps that follow reuse instead of
    /// consuming their generators again. A generator the forward reads directly is not known to the session.
    /// </para>
    /// <para>
    /// Cost, once per plan: two no-grad forwards, and two copies of the trained tensors and the step state (before
    /// the traced step and before the first replay; the first is held until the replay agrees) plus one copy of the
    /// traced step's draws.
    /// </para>
    /// </remarks>
    public bool VerifyReplayAgreement { get; init; }

    /// <summary>
    /// Tensors a training step advances without the optimizer (running statistics, EMA codebooks) that no layer in
    /// <see cref="Layers"/> registers as a buffer, for <see cref="VerifyReplayAgreement"/>: they are put back with the
    /// trained tensors when a checked plan's step is undone. Null = only the layers' registered buffers. Asked for
    /// only when a snapshot is taken or compared (twice per checked plan), not on every step.
    /// </summary>
    public Func<IReadOnlyList<Tensor<T>>>? StepState { get; init; }

    /// <summary>
    /// A closure objective: the step's loss as a function that builds its own per-step data (a batch, noise, labels,
    /// a sampled timestep) inside itself. When set, the compiled step re-probes the closure every step and replays a
    /// plan only when the closure traces to the same structure, after copying in this step's data (see
    /// <see cref="AiDotNet.Tensors.Engines.Compilation.ClosureObjectiveGuard{T}"/>). <see cref="Forward"/> and
    /// <see cref="ComputeLoss"/> must evaluate the same closure (they drive the eager fallback), and
    /// <see cref="Input"/>/<see cref="Target"/> may be placeholders.
    /// </summary>
    /// <remarks>
    /// The guard checks everything a plan would freeze, so a closure step ignores <see cref="GraphBreakReason"/>'s
    /// per-step host state reason and <see cref="VerifyReplayAgreement"/>: a changed captured value changes the
    /// signature and is compiled from its own trace, and a host read of an intermediate is never replayed.
    /// </remarks>
    public Func<Tensor<T>>? ClosureObjective { get; init; }

    /// <summary>
    /// The caller's own eager step on this request's data, for <see cref="VerifyReplayAgreement"/>: when the plan this
    /// step traces later fails its replay check, the session restores the model and optimizer to before this step and
    /// re-runs it with this, so the run matches one that was eager from the start. Null runs
    /// <see cref="TapeTrainingStepper{T}.EagerStep"/> on a copy of this step's input and target.
    /// </summary>
    public Func<T>? EagerReplay { get; init; }

    /// <summary>
    /// What identifies the objective this step trains, for <see cref="VerifyReplayAgreement"/> (a custom objective's
    /// method). Objectives that share an optimizer share its session and its plan, and the plan is not keyed by the
    /// objective: when this differs from the previous step's, the next replay is checked against the eager forward
    /// again, so a plan traced for one objective is never replayed unchecked as another. Null never re-checks.
    /// </summary>
    public object? ObjectiveKey { get; init; }
}

/// <summary>
/// The fused compiled training lifecycle of one model: maps its optimizer to a fused kernel, runs
/// <see cref="CompiledTapeTrainingStep{T}"/> (forward + backward + update as one compiled plan, GPU-resident on a
/// GPU engine, a fused CPU kernel otherwise), and keeps the plan honest across steps.
/// </summary>
/// <remarks>
/// <para>
/// Every base class that trains on the tape owns one of these, so model and layer authors get the fused path on any
/// engine without writing it. Before this type, <c>NeuralNetworkBase</c> held this logic inline and the time-series
/// models re-implemented reduced copies (#1804).
/// </para>
/// <para>
/// The checks it keeps: the optimizer, regularization and clip must have a fused form; every step until the plan has
/// proven itself, and periodically after, it confirms the update reached the model's live parameter tensors (a plan
/// decoupled from them trains nothing while reporting a falling loss, ooples/AiDotNet#1822); and once committed, a
/// failure is surfaced rather than silently switching optimizers mid-trajectory.
/// </para>
/// </remarks>
internal sealed class FusedTrainingSession<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Steps between persistence re-checks once the plan has proven itself.</summary>
    internal const int PersistenceRecheckInterval = 16;

    // Elements sampled per persistence checksum: enough to see any real update, cheap on billion-parameter models.
    private const int ChecksumTargetSamples = 1 << 16;

    private readonly object _owner;
    private readonly object? _stateKey;
    private readonly Action? _onReset;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _lastOptimizer;
    private bool _persistenceVerified;
    // The plan whose update the numeric probe last saw land; a later plan is verified numerically again. Held weakly:
    // it is only ever compared by identity, and a strong reference kept a replaced plan and every activation buffer
    // it owns (2.3 GB for DenseNet-121 at batch 64) alive until the next check, up to PersistenceRecheckInterval
    // steps later.
    private WeakReference<object>? _verifiedPlan;

    /// <summary>Persistence checks answered by comparing the configured plan with the verified one.</summary>
    internal int StructuralPersistenceRecheckCount { get; private set; }

    /// <summary>Persistence checks that sampled the parameters and read them back.</summary>
    internal int NumericPersistenceProbeCount { get; private set; }
    private int _stepsSincePersistenceCheck;
    private bool _replayAgreementVerified;
    private object? _objectiveKey;
    private bool _zeroGradientCrossChecked;

    // The state before the trace step of a plan whose replay is still to be checked, and that step's eager redo: on a
    // disagreeing replay both fused updates are undone and the trace step re-runs eagerly. Dropped once the replay
    // agrees.
    private PendingTrace? _pendingTrace;

    private sealed class PendingTrace
    {
        public required StepSnapshot Before { get; init; }
        public required object OptimizerPosition { get; init; }
        public required Func<T> EagerRedo { get; init; }

        // The traced step's per-step draws and seed counters, copied: its eager redo reuses them, so the generators
        // are consumed once for that step, as on the eager tape.
        public CompiledStepRandom<T>.CaptureScope? Draws { get; set; }
    }

    // The values of everything a step changes besides the optimizer's own position: the trained tensors and the step
    // state (registered buffers and the request's StepState).
    private sealed class StepSnapshot
    {
        public required (Tensor<T> Live, Tensor<T> Saved)[] Weights { get; init; }
        public required (Tensor<T> Live, Tensor<T> Saved)[] State { get; init; }
    }

    // Relative tolerance for the replay-agreement check: fusion reassociates sums, so float losses differ in the
    // last few digits; a plan replaying stale host-side decisions differs at the scale of the loss itself.
    private const double ReplayAgreementTolerance = 1e-3;

    /// <summary>Creates the fused lifecycle for one model.</summary>
    /// <param name="owner">The model; keys its compiled plan and optimizer moments.</param>
    /// <param name="onReset">Owner cleanup to run when a committed plan is dropped (caches, GPU transients).</param>
    /// <param name="stateKey">
    /// The slot of the owner's compiled state this session trains, when the owner has several (one per training
    /// objective that steps its own optimizer). Null uses the slot of the layer set, and a reset drops every slot.
    /// </param>
    public FusedTrainingSession(object owner, Action? onReset = null, object? stateKey = null)
    {
        Guard.NotNull(owner);
        _owner = owner;
        _onReset = onReset;
        _stateKey = stateKey;
    }

    /// <summary>True once a fused step has trained the model; later steps must stay fused or fail loudly.</summary>
    public bool IsCommitted { get; private set; }

    /// <summary>True when the fused path is off for this model (it proved unusable, or the model opts out);
    /// steps go to the eager tape.</summary>
    public bool IsDisabled { get; set; }

    /// <summary>
    /// Whether the last <see cref="FusedStepOutcome.Stepped"/> step ran outside the compiled plan: the eager tape took
    /// its gradients and the plan's optimizer applied them (a closure objective no plan could replay that step).
    /// </summary>
    public bool LastStepRanOutsidePlan { get; private set; }

    /// <summary>Why the most recent step did not run fused, for diagnostics.</summary>
    public string? LastMissReason { get; private set; }

    /// <summary>The exception the compiled step swallowed on its last failure, if any.</summary>
    public Exception? LastFallbackException { get; private set; }

    /// <summary>Forgets the plan. <paramref name="stickyDisable"/> keeps later steps on the eager tape.</summary>
    public void Reset(bool stickyDisable)
    {
        // Owner cleanup first: it may itself restart this session (a model's cache invalidation re-enables the
        // fused path for ordinary resets), and the state set below must win.
        _onReset?.Invoke();
        if (_stateKey is null) CompiledTapeTrainingStep<T>.Invalidate(_owner);
        else CompiledTapeTrainingStep<T>.Invalidate(_owner, _stateKey);
        IsDisabled = stickyDisable;
        IsCommitted = false;
        _persistenceVerified = false;
        _verifiedPlan = null;
        _stepsSincePersistenceCheck = 0;
        _replayAgreementVerified = false;
        _zeroGradientCrossChecked = false;
        _pendingTrace = null;
    }

    /// <summary>Turns the fused path off for this session's later steps, recording why (a caller-side decline).</summary>
    public void Disable(string reason)
    {
        IsDisabled = true;
        LastMissReason = reason;
    }

    /// <summary>Re-enables the fused path after an explicit model reset (new layers, new optimizer).</summary>
    public void Restart(bool disabled = false)
    {
        IsDisabled = disabled;
        IsCommitted = false;
        _persistenceVerified = false;
        _verifiedPlan = null;
        _stepsSincePersistenceCheck = 0;
        _replayAgreementVerified = false;
        _zeroGradientCrossChecked = false;
        _pendingTrace = null;
        LastMissReason = null;
        LastFallbackException = null;
    }

    /// <summary>
    /// Re-enables the fused path after a reset that keeps the model's optimizer (<c>ResetState</c>, which clears
    /// recurrent state). An uncommitted session restarts. A committed one keeps its plan, and with it the
    /// optimizer's moments, which the eager optimizer would also keep: it stays committed, so a step that cannot
    /// run fused fails loudly instead of continuing eagerly from the eager optimizer's older moments, and its next
    /// step re-checks that the plan still trains the live parameters and, when the caller asks, that its replay
    /// agrees with the eager forward.
    /// </summary>
    public void Rearm(bool disabled = false)
    {
        if (!IsCommitted)
        {
            Restart(disabled);
            return;
        }

        _stepsSincePersistenceCheck = PersistenceRecheckInterval;
        _replayAgreementVerified = false;
        LastMissReason = null;
        LastFallbackException = null;
    }

    /// <summary>Runs one fused step, or says why it cannot.</summary>
    public FusedStepOutcome TryStep(FusedTrainingStepRequest<T> request, out T loss)
    {
        var outcome = TryStep(request, out loss, out var lossTensor);
        if (lossTensor is not null)
            loss = ReadLoss(loss, lossTensor);
        return outcome;
    }

    /// <summary>
    /// Runs one fused step, or says why it cannot. A compiled step returns its loss as <paramref name="lossTensor"/>
    /// (and <paramref name="loss"/> is then zero) without reading it: reading the scalar waits for the device to
    /// finish the step, so a caller that defers the read lets the host prepare the next step meanwhile. Null when
    /// the step produced only a scalar.
    /// </summary>
    public FusedStepOutcome TryStep(FusedTrainingStepRequest<T> request, out T loss, out Tensor<T>? lossTensor)
    {
        Guard.NotNull(request);
        loss = NumOps.Zero;
        lossTensor = null;
        LastFallbackException = null;
        LastStepRanOutsidePlan = false;
        // Draws a declined plan's step left for the eager step that followed it; that step has run by now.
        CompiledStepRandom<T>.DisarmReplay(this);

        // The compiled plan carries the optimizer's moments. A different optimizer instance (a new Train call, a
        // learning-rate overload, an optimizer override) starts from its own fresh state, as it would eagerly, so
        // the old plan is dropped here rather than read as hyperparameter drift on a committed plan, which refuses
        // the step; a replacement with the same hyperparameters must not inherit the old plan's moments either.
        // Every caller goes through this, so none tracks the instance itself. A new optimizer is also a fresh
        // chance for a plan that a device fault disabled; an owner that opts out checks before stepping.
        if (!ReferenceEquals(_lastOptimizer, request.Optimizer))
        {
            if (_lastOptimizer is not null)
                Reset(stickyDisable: false);
            _lastOptimizer = request.Optimizer;
        }

        // A standard objective sharing this session must recheck replay agreement when its identity changes.
        if (request.VerifyReplayAgreement && request.ObjectiveKey is { } objectiveKey)
        {
            if (IsCommitted && _objectiveKey is not null && !_objectiveKey.Equals(objectiveKey))
                _replayAgreementVerified = false;
            _objectiveKey = objectiveKey;
        }

        if (request.ClosureObjective is null && request.GraphBreakReason is { } graphBreak)
        {
            if (!IsCommitted)
                return Decline("graph break: " + graphBreak);
            // The plan holds the optimizer's moments; an eager step now would restart them silently.
            LastFallbackException = new InvalidOperationException("graph break after fused steps ran: " + graphBreak);
            return FusedStepOutcome.CommittedFailure;
        }

        if (_owner is NeuralNetworks.NeuralNetworkBase<T> { ActiveMixedPrecisionContext: not null })
            return Decline("mixed precision active (requires eager tape loss scaling)");

        bool isDistributed = _owner is NeuralNetworks.NeuralNetworkBase<T> { DistributedCoordinator: not null };
        if (isDistributed)
            AiDotNet.Tensors.Engines.Autodiff.RetainedOutputCapture<T>.RejectUnsupportedRoute("distributed");


        if (IsDisabled)
        {
            // Keep the cause: the step that disabled the path recorded why, and a bare "disabled" on every later step
            // hid it from the audit and from the model's diagnostics.
            const string sticky = "fused path sticky-disabled from a prior fallback";
            string? prior = LastMissReason;
            return Decline(prior is null || prior.StartsWith(sticky, StringComparison.Ordinal) ? prior ?? sticky : sticky + ": " + prior);
        }
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return Decline("TensorCodecOptions.EnableCompilation = false");
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            return Decline($"numeric type {typeof(T).Name} not supported by fused kernel");

        var optimizer = request.Optimizer;
        if (OverridesStepBelowFusedSpec(optimizer.GetType()))
            return Decline($"optimizer {optimizer.GetType().Name} overrides its step below its fused mapping; the fused "
                + "kernel would run the base algorithm and bypass that step");
        if (!TryMapToFusedOptimizerConfig(optimizer, out var config))
            return Decline($"optimizer {optimizer.GetType().Name} not compatible with fused kernel");
        if (request.Layers.Count == 0 && (request.ExtraParameters is null || request.ExtraParameters.Count == 0))
            return Decline("no trainable layers");

        // The optimizer's regularization, applied the way its flat path applies it: to the gradient, on the side of the
        // clip its RegularizationOrder chooses. L2 has a fused form (the plan adds strength * theta ahead of its clip);
        // any other regularizer runs on the eager tape. Every decline is decided BEFORE the per-step arena below exists: TensorArena.Create is
        // thread-static and stacks on the enclosing arena, so a decline after it would leave it current for the
        // eager fallback and every later step.
        double l2 = 0.0;
        if (!isDistributed && OptimizerRegularizationOf(optimizer) is { } regularization)
        {
            // The plan adds one L2 strength to every parameter; a model that excludes some (TF EfficientNet decays
            // nothing in batch normalization) trains on the eager tape, which skips them.
            if (regularization is AiDotNet.Regularization.L2Regularization<T, Tensor<T>, Tensor<T>> l2Regularization)
            {
                if (request.ExcludesParametersFromWeightDecay?.Invoke() == true)
                    return RejectUnsupportedConfiguration(
                        "L2 with parameters excluded from weight decay has no fused form; the eager tape applies it");
                l2 = l2Regularization.GetOptions().Strength;
            }
            else
                return RejectUnsupportedConfiguration("regularization " + regularization.GetType().Name
                    + " has no fused form; the eager tape applies it");
        }

        // The eager tape clips twice: the model by its own norm, then the optimizer's step by its MaxGradientNorm.
        // The plan clips once, so it gets the threshold the two compose to (a model whose own clip is 0 would
        // otherwise train unclipped fused and clipped eager).
        double clip = request.ModelGradientClip;
        double optimizerClip = optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> clipping
            ? clipping.TapeStepGradientClipNorm
            : 0.0;
        if (double.IsNaN(optimizerClip))
            return RejectUnsupportedConfiguration("optimizer clips gradients by value; the fused plan clips only by global norm");
        if (!isDistributed && optimizerClip > 0.0)
            clip = clip > 0.0 ? Math.Min(clip, optimizerClip) : optimizerClip;
        // The plan adds its L2 term on the optimizer's side of the clip; after it is PyTorch's order, the default.
        bool l2AfterClip = optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> { RegularizesAfterClip: true };

        // A deliberate break is selected before probing: the objective and its RNG execute exactly once.
        if (request.ClosureGraphBreakPolicy != ClosureGraphBreakPolicy.Disallow)
        {
            if (request.ClosureGraphBreakPolicy != ClosureGraphBreakPolicy.UseConfiguredOptimizer
                || request.ClosureObjective is null || string.IsNullOrWhiteSpace(request.GraphBreakReason)
                || !IsCommitted || isDistributed)
            {
                LastFallbackException = new InvalidOperationException(
                    "A hybrid step requires an explicit closure graph break and an existing local configured optimizer.");
                return FusedStepOutcome.CommittedFailure;
            }
            if (!CompiledTapeTrainingStep<T>.TryDeclaredGraphBreak(_owner, _stateKey, request, config,
                    l2, l2AfterClip, clip, out var hybridLoss))
            {
                LastFallbackException = CompiledTapeTrainingStep<T>.GetLastFallbackException()
                    ?? new InvalidOperationException("The configured optimizer refused the declared graph break.");
                return FusedStepOutcome.CommittedFailure;
            }
            LastStepRanOutsidePlan = true;
            LastMissReason = request.GraphBreakReason;
            lossTensor = hybridLoss!;
            loss = hybridLoss!.Length > 0 ? hybridLoss[0] : NumOps.Zero;
            InvalidateDerivedWeightCaches();
            if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> hybridScheduled)
                hybridScheduled.OnBatchEnd();
            return FusedStepOutcome.Stepped;
        }

        bool verifyPersistence = !isDistributed && ((!_persistenceVerified && !IsCommitted)
            || ++_stepsSincePersistenceCheck >= PersistenceRecheckInterval);
        // Once a replay has been seen to update the live tensors, every later replay runs the same plan, so a
        // periodic re-check only has to confirm the plan still trains the model's live tensors -- a host-side
        // comparison (ConfiguredPlanTrainsLiveParameters). The numeric probe below costs a slice, a copy and a
        // reduction per parameter plus a synchronous readback that drains the device (6-14 ms every 16 steps on
        // the parity Transformer, a fifth of its CUDA step time); it runs only when that comparison cannot answer.
        bool structuralRecheck = verifyPersistence && _persistenceVerified && _verifiedPlan is not null
            && _verifiedPlan.TryGetTarget(out var verifiedPlan)
            && ReferenceEquals(verifiedPlan, CompiledTapeTrainingStep<T>.ConfiguredPlanIdentity);
        using var probe = verifyPersistence && !structuralRecheck ? SampleParameters(request) : null;
        if (structuralRecheck) StructuralPersistenceRecheckCount++;
        else if (verifyPersistence) NumericPersistenceProbeCount++;
        bool gradientsObserved = false;
        bool gradientNonZero = false;
        void OnGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
        {
            // Only the numeric check reads whether the gradients were nonzero (a reduction per gradient on the device).
            if (verifyPersistence && !structuralRecheck)
            {
                gradientsObserved = true;
                gradientNonZero = AnyGradientNonZero(gradients);
            }
            request.OnGradients?.Invoke(gradients);
        }

        // The first replay with new data (the step after the trace) is checked against the eager forward when the
        // caller asks: the eager loss on this batch, and a copy of the trained tensors to undo a disagreeing update.
        // In distributed mode stepLoss is the globally reduced loss across all ranks, so each non-empty rank compares
        // eagerLoss against its pre-reduction local compiled loss before any weight or moment mutation occurs, and
        // coordinates the verdict across all ranks via AllReduce(Min).
        bool checkReplay = request.ClosureObjective is null
            && request.VerifyReplayAgreement && IsCommitted && !_replayAgreementVerified;
        double eagerLoss = 0.0;
        double? distributedEagerReplayLoss = null;
        StepSnapshot? snapshot = null;
        if (checkReplay)
        {
            if (isDistributed)
            {
                if (request.Input.Length > 0 && request.Target.Length > 0)
                {
                    using var noGrad = new NoGradScope<T>();
                    using var noDraws = CompiledStepRandom<T>.SuppressDraws();
                    var eager = request.ComputeLoss(request.Forward(request.Input), request.Target);
                    distributedEagerReplayLoss = eager.Length > 0 ? NumOps.ToDouble(eager[0]) : 0.0;
                }
            }
            else
            {
            try
            {
                using var noGrad = new NoGradScope<T>();
                using var noDraws = CompiledStepRandom<T>.SuppressDraws();
                var eager = request.ComputeLoss(request.Forward(request.Input), request.Target);
                eagerLoss = eager.Length > 0 ? NumOps.ToDouble(eager[0]) : 0.0;
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // The forward cannot run outside a tape (a gradient penalty's nested tape records nothing under the
                // no-grad scope), so the replay cannot be checked: undo the traced step and train eagerly.
                WarnWithException("the replay check's eager forward failed outside a tape", ex);
                return UndoTraceAndDecline(optimizer, null, null, "the replay check's eager forward failed ("
                    + ex.GetType().Name + ": " + ex.Message + "), so the compiled plan's replay cannot be checked");
            }
            snapshot = TakeSnapshot(request);
            }
        }

        // The trace step of a plan whose replay will be checked (against the eager forward, or by a closure
        // objective's signature): keep what it takes to redo it eagerly.
        bool wasCommitted = IsCommitted;
        PendingTrace? trace = null;
        double? eagerLossOfTracedStep = null;
        if (request.VerifyReplayAgreement && request.ClosureObjective is null && !IsCommitted
            && optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> positioned)
        {
            // The traced step is checked against the eager forward as well. A plan can be wrong on its first step only:
            // a node that reads a tensor the trace had not computed yet (a gather through indices built by recorded
            // integer ops read them as zeros) runs correctly on every replay, so the first-replay check below passes
            // while the traced step has already trained the weights on a different forward. Taken before the weights
            // are copied, so the copy holds whatever this forward sized.
            if (!isDistributed)
                eagerLossOfTracedStep = TryEagerLossForCheck(request);
            trace = new PendingTrace
            {
                Before = TakeSnapshot(request),
                OptimizerPosition = positioned.CaptureStepPosition(),
                EagerRedo = request.EagerReplay ?? EagerRedoOnCopies(request),
            };
        }

        // Each step's transient activations are reclaimed when it returns, the way PyTorch's caching allocator
        // returns an iteration's blocks; the plan's moments and persistent input/target are not arena-allocated
        // (#1624 / #1640). Disposed before result handling so a fallback starts from a clean ring. It pools even nested
        // (an adversarial step runs inside the caller's arena): the step's transients are dead at its return either way,
        // and a nested arena that does not pool drops every buffer it rented, including those it took from the
        // cross-arena pool. A DCGAN CPU step dropped 12 of them that way, so the generator's next detached forward
        // allocated 69 MB afresh, and this arena another 16 MB of its own, every step.
        bool ran;
        T stepLoss;
        var deferredLoss = new System.Runtime.CompilerServices.StrongBox<Tensor<T>?>();
        // A step that may be undone keeps its per-step draws and seed counters, so the eager step that then runs the
        // same batch reuses them instead of consuming the generators a second time.
        var stepDraws = !isDistributed && (trace is not null || snapshot is not null) ? CompiledStepRandom<T>.CaptureDraws() : null;
        var stepArena = TensorArena.Create(poolWhenNested: true);
        try
        {
            ran = CompiledTapeTrainingStep<T>.TryStepWithFusedOptimizer(
                request.Layers,
                request.Input,
                request.Target,
                request.Forward,
                request.ComputeLoss,
                config.Type,
                config.LearningRate,
                config.Beta1,
                config.Beta2,
                config.Epsilon,
                config.WeightDecay,
                out stepLoss,
                maxGradNorm: clip,
                lrSchedule: config.Schedule,
                eagerOptimizer: optimizer,
                useBf16Moments: config.UseBf16Moments || config.MomentStorage == AiDotNet.Enums.OptimizerMomentStorage.BFloat16,
                int8MomentBlockSize: config.MomentStorage == AiDotNet.Enums.OptimizerMomentStorage.Int8Blockwise
                    ? config.Int8MomentBlockSize : 0,
                int8MinQuantizedLength: config.Int8MinQuantizedLength,
                extraTensors: request.ExtraParameters,
                fusedExtras: config.Extras,
                onGradients: OnGradients,
                trainableSelection: request.Selection,
                owner: _owner,
                l2Regularization: l2,
                deferredLoss: deferredLoss,
                l2AfterClip: l2AfterClip,
                stateKey: _stateKey,
                verifyDistributedReplay: isDistributed && checkReplay,
                distributedEagerReplayLoss: distributedEagerReplayLoss,
                distributedReplayTolerance: ReplayAgreementTolerance,
                closureObjective: request.ClosureObjective,
                layersAfterLazyInitialization: request.LayersAfterLazyInitialization);
        }
        finally
        {
            stepArena.Dispose();
            stepDraws?.Dispose();
        }

        if (!ran)
        {
            if (isDistributed && CompiledTapeTrainingStep<T>.TryConsumeDistributedReplayDisagreement(out string? disagreeReason))
            {
                Warn(disagreeReason);
                Reset(stickyDisable: true);
                return Decline(disagreeReason);
            }

            LastFallbackException = CompiledTapeTrainingStep<T>.GetLastFallbackException();
            if (CompiledTapeTrainingStep<T>.LastFallbackWasUnreplayableClosureStep)
                return FusedStepOutcome.CommittedFailure;
            // The first replay after the trace could not run (a closure objective whose next step no guarded plan can
            // replay): the traced step is undone and redone eagerly, as for a disagreeing replay.
            if (IsCommitted && _pendingTrace is not null)
                return UndoTraceAndDecline(optimizer, null, null, "the compiled plan cannot replay the objective's next step ("
                    + (LastFallbackException?.Message ?? "no reason") + ")");
            if (IsCommitted)
                return FusedStepOutcome.CommittedFailure;
            IsDisabled = true;
            return Decline("compiled plan declined the step ("
                + (LastFallbackException is null ? "no exception" : LastFallbackException.GetType().Name + ": " + LastFallbackException.Message)
                + ")");
        }

        if (trace is not null && eagerLossOfTracedStep is { } eagerTraced)
        {
            double tracedLoss = NumOps.ToDouble(ReadLoss(stepLoss, deferredLoss.Value));
            if (!LossesAgree(eagerTraced, tracedLoss))
            {
                // Nothing of this plan is kept: the weights, the step state (running statistics, EMA codebooks) and
                // the optimizer go back to before the step, and the caller's eager step then runs this step itself
                // (no redo here, unlike a failed first replay) on the draws this step took.
                bool restored = SameLive(trace.Before.Weights, LiveTrainedTensors(request));
                if (restored) Restore(trace.Before);
                string reason = "the compiled plan's loss on its traced step (" + tracedLoss.ToString("G6")
                    + ") disagrees with the eager forward (" + eagerTraced.ToString("G6") + "): the plan's first step "
                    + "does not compute the training forward (a recorded node read a value the trace had not computed, "
                    + "or the forward has its own randomness), so "
                    + (restored ? "its update was undone and " : "") + "training continues on the eager tape";
                Warn(reason);
                Reset(stickyDisable: true);
                // The optimizer's own position (step count, schedule) is where it was: only a step that is kept
                // advances it (OnBatchEnd, below), and the plan's moments went with the plan.
                if (restored) InvalidateDerivedWeightCaches();
                if (stepDraws is not null) CompiledStepRandom<T>.ArmReplay(stepDraws.Detached(), this);
                return Decline(reason);
            }
        }

        if (checkReplay && isDistributed)
        {
            if (CompiledTapeTrainingStep<T>.WasLastDistributedReplayEvaluated())
            {
                _replayAgreementVerified = true;
            }
        }
        else if (checkReplay && snapshot is not null)
        {
            double replayLoss = NumOps.ToDouble(ReadLoss(stepLoss, deferredLoss.Value));
            // Relative to the losses themselves, with no floor: floored at 1, a loss below 1e-3 (a small-magnitude
            // regression target, a model near convergence) passed whatever the plan had frozen, and the stale plan
            // was committed for the rest of the run.
            if (!LossesAgree(eagerLoss, replayLoss))
            {
                return UndoTraceAndDecline(optimizer, snapshot, stepDraws, "the compiled plan's loss on new data ("
                    + replayLoss.ToString("G6") + ") disagrees with the eager forward (" + eagerLoss.ToString("G6")
                    + "): the training forward depends on more than its input tensor (a host-side read or captured "
                    + "per-step state)");
            }
            _replayAgreementVerified = true;
            _pendingTrace = null;
        }

        if (structuralRecheck)
        {
            bool? stillAttached = CompiledTapeTrainingStep<T>.ConfiguredPlanTrainsLiveParameters(
                (IEnumerable<Tensor<T>>?)request.Selection ?? EnumerateLiveParameters(request));
            if (stillAttached == true)
            {
                _stepsSincePersistenceCheck = 0;
            }
            else if (stillAttached == false)
            {
                string reason = "the fused compiled plan no longer trains the model's live parameter tensors, so "
                    + "training continues on the eager tape (ooples/AiDotNet#1822)";
                Warn(reason);
                Reset(stickyDisable: true);
                return Decline(reason);
            }
            else
            {
                // The plan cannot report its parameters: verify numerically on the next step.
                _persistenceVerified = false;
                _stepsSincePersistenceCheck = PersistenceRecheckInterval;
            }
        }
        else if (verifyPersistence)
        {
            bool persisted = probe is not null && AnySampleChanged(probe);
            bool? attached = CompiledTapeTrainingStep<T>.ConfiguredPlanTrainsLiveParameters(
                (IEnumerable<Tensor<T>>?)request.Selection ?? EnumerateLiveParameters(request));
            bool detached = attached == false;
            // No sample means no trained tensor held an element before the step: lazily shaped weights materialize on
            // the first forward, so there was nothing to compare against. That is inconclusive, not a failed update.
            bool sampled = probe is { Length: > 0 };

            // An all-zero plan gradient reads as "nothing could have moved", which is right for a model at a stationary
            // point and wrong for a plan whose trace lost the path from the loss to the parameters: such a plan stepped
            // a VAE forever without moving a weight. Once per plan, ask the eager tape. The plan moved nothing, so
            // dropping it here loses no update; this step then runs on the eager tape.
            if (!persisted && !detached && gradientsObserved && !gradientNonZero && !_zeroGradientCrossChecked)
            {
                _zeroGradientCrossChecked = true;
                if (EagerGradientNonZero(request))
                {
                    string reason = "fused compiled step produced an all-zero gradient where the eager tape's is not "
                        + "zero; the plan is decoupled from the loss, so training continues on the eager tape";
                    Warn(reason);
                    if (IsCommitted)
                        Reset(stickyDisable: true);
                    else
                        IsDisabled = true;
                    return Decline(reason);
                }
            }

            if (!persisted && !detached
                && (!sampled || !StepCouldHaveMovedParameters(config, gradientsObserved, gradientNonZero, attached == true)))
            {
                // Nothing could have moved (zero gradients, zero learning rate) or nothing was sampled: inconclusive,
                // check again next step.
                _stepsSincePersistenceCheck = PersistenceRecheckInterval;
            }
            else if (!persisted || detached)
            {
                string reason = "fused compiled step ran (loss " + NumOps.ToDouble(ReadLoss(stepLoss, deferredLoss.Value)).ToString("G6")
                    + ") but did not persist a parameter update; the plan is decoupled from the model's live "
                    + "parameter tensors, so training continues on the eager tape (ooples/AiDotNet#1822)"
                    + $" [persisted={persisted}, attached={attached}, sampled={probe?.Length}, gradientsObserved={gradientsObserved}, gradientNonZero={gradientNonZero}]";
                Warn(reason);
                if (IsCommitted)
                    Reset(stickyDisable: true);
                else
                    IsDisabled = true;
                return Decline(reason);
            }
            else
            {
                _persistenceVerified = true;
                _stepsSincePersistenceCheck = 0;
                _verifiedPlan = CompiledTapeTrainingStep<T>.ConfiguredPlanIdentity is { } configured
                    ? new WeakReference<object>(configured)
                    : null;
            }
        }

        IsCommitted = true;
        if (trace is not null)
        {
            // Copied: the plan refills these tensors in place on its next replay.
            trace.Draws = stepDraws?.Detached();
            _pendingTrace = trace;
        }
        else if (wasCommitted && request.ClosureObjective is not null && !checkReplay) _pendingTrace = null;   // a replay ran
        LastMissReason = null;
        loss = stepLoss;
        lossTensor = deferredLoss.Value;

        // The plan wrote the weights in place, past the version counters an eager optimizer step bumps, so every
        // derived weight copy (CPU packed inference weights, GPU-resident uploads) is stale. Inference read those
        // copies and kept predicting with the pre-training weights for every caller but NeuralNetworkBase, which
        // invalidated on its own. One place, so no caller of the session can miss it.
        InvalidateDerivedWeightCaches();
        if (!isDistributed && optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.OnBatchEnd();

        // Counted here, where every base class's fused step lands, and regardless of the diagnostics level: the hit
        // counter exists to answer "did the fused path engage at all?" after a run (#1930), and the network's own
        // hit event fires only at PerStep, so FusedOptimizerHits stayed 0 through any number of fused steps.
        AiDotNet.Configuration.TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: true, reason: null);
        return FusedStepOutcome.Stepped;
    }

    // A checked plan failed its replay check. Undo its fused updates and drop it for good. When the pre-trace state is
    // known (the same tensors as now; a layer the trace materialized has no pre-trace value), restore it (trained
    // tensors and step state), put the optimizer back at its pre-trace position and redo the traced step eagerly on
    // the draws it took, so the run matches one that was eager from the start; otherwise restore the pre-replay state
    // (null when the replay never ran). The replay's own draws are left for the caller's eager step of that batch.
    private FusedStepOutcome UndoTraceAndDecline(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        StepSnapshot? beforeReplay,
        CompiledStepRandom<T>.CaptureScope? replayDraws,
        string cause)
    {
        var pending = _pendingTrace;
        if (pending is not null && (beforeReplay is null || SameTensors(pending.Before.Weights, beforeReplay.Weights)))
        {
            Restore(pending.Before);
            string redone = cause + ", so the fused updates were undone, the traced step was redone on the eager tape, "
                + "and training continues on the eager tape";
            Warn(redone);
            Reset(stickyDisable: true);
            ((Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>>)optimizer).RestoreStepPosition(pending.OptimizerPosition);
            InvalidateDerivedWeightCaches();
            using (pending.Draws is null ? null : CompiledStepRandom<T>.ReplayDraws(pending.Draws))
                pending.EagerRedo();
            if (replayDraws is not null) CompiledStepRandom<T>.ArmReplay(replayDraws.Detached(), this);
            return Decline(redone);
        }

        if (beforeReplay is not null) Restore(beforeReplay);
        string reason = cause + ", so " + (beforeReplay is not null ? "the replay's update was undone and " : "")
            + "training continues on the eager tape";
        Warn(reason);
        Reset(stickyDisable: true);
        if (beforeReplay is not null)
        {
            InvalidateDerivedWeightCaches();
            if (replayDraws is not null) CompiledStepRandom<T>.ArmReplay(replayDraws.Detached(), this);
        }
        return Decline(reason);
    }

    // Relative to the losses themselves, with no floor (see the first-replay check). Paired NaN losses agree; an
    // infinite loss never does: the relative bound scales with the larger loss, so an infinite one would otherwise
    // accept any finite partner (|inf - x| <= tol * inf).
    private static bool LossesAgree(double eager, double compiled)
    {
        if (double.IsNaN(eager) && double.IsNaN(compiled)) return true;
        if (double.IsInfinity(eager) || double.IsInfinity(compiled)) return false;
        return Math.Abs(eager - compiled) <= ReplayAgreementTolerance * Math.Max(Math.Abs(eager), Math.Abs(compiled));
    }

    // The eager loss of the request's step, without recording gradients, drawing per-step randomness or advancing
    // per-step state. Null when the forward cannot run outside a tape: the traced step then goes unchecked, and the
    // first replay's check, which runs the same forward, declines the plan.
    private double? TryEagerLossForCheck(FusedTrainingStepRequest<T> request)
    {
        try
        {
            using var noGrad = new NoGradScope<T>();
            using var noDraws = CompiledStepRandom<T>.SuppressDraws();
            var eager = request.ComputeLoss(request.Forward(request.Input), request.Target);
            return eager.Length > 0 ? NumOps.ToDouble(eager[0]) : 0.0;
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            WarnWithException("the compiled plan's traced step is not checked against the eager forward, which failed "
                + "outside a tape; its first replay is", ex);
            return null;
        }
    }

    // The step's loss as a scalar: the deferred tensor's value when the step returned one, else the scalar it produced.
    private static T ReadLoss(T scalar, Tensor<T>? deferred)
        => deferred is null ? scalar : deferred.Length > 0 ? deferred[0] : NumOps.Zero;

    /// <summary>Maps an optimizer to its fused kernel configuration, when it has one.</summary>
    // Drops every cached derived copy of the weights after the plan updated them in place.
    private static void InvalidateDerivedWeightCaches()
    {
        (AiDotNet.Tensors.Engines.AiDotNetEngine.Current as AiDotNet.Tensors.Engines.DirectGpuTensorEngine)?.InvalidateAllWeightCaches();
        AiDotNet.Tensors.Engines.InferenceWeightCache.InvalidateAll();
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, bool> s_overridesStepBelowSpec = new();

    /// <summary>
    /// Whether this optimizer type overrides its tape step (<c>StepCore</c>) in a class below the one that maps it to a
    /// fused kernel: a subclass of a fused-mapped optimizer with its own update (a user's, a test's observing double).
    /// The mapping it inherits describes the base algorithm, so the fused kernel would bypass the override.
    /// </summary>
    internal static bool OverridesStepBelowFusedSpec(Type optimizerType)
        => s_overridesStepBelowSpec.GetOrAdd(optimizerType, static type =>
        {
            if (!typeof(Optimizers.Fused.IFusedOptimizerSpec).IsAssignableFrom(type)) return false;
            var map = type.GetInterfaceMap(typeof(Optimizers.Fused.IFusedOptimizerSpec));
            Type? specType = null;
            for (int i = 0; i < map.InterfaceMethods.Length; i++)
                if (map.InterfaceMethods[i].Name == nameof(Optimizers.Fused.IFusedOptimizerSpec.TryGetFusedOptimizerConfig))
                    specType = map.TargetMethods[i].DeclaringType;
            var step = type.GetMethod("StepCore", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic
                | System.Reflection.BindingFlags.Public, null, new[] { typeof(TapeStepContext<T>) }, null);
            return specType is not null && step?.DeclaringType is { } stepType
                && stepType != specType && specType.IsAssignableFrom(stepType);
        });

    internal static bool TryMapToFusedOptimizerConfig(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        out Optimizers.Fused.FusedOptimizerConfig config)
    {
        config = default;
        return optimizer is Optimizers.Fused.IFusedOptimizerSpec spec
            && spec.TryGetFusedOptimizerConfig(out config);
    }

    /// <summary>The regularization the caller explicitly configured on the optimizer, if the step must apply it.</summary>
    internal static IRegularization<T, Tensor<T>, Tensor<T>>? OptimizerRegularizationOf(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        if (optimizer is not Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> gradientBased) return null;
        // Only a regularization the caller chose. The options default to L2(0.01); applying that implicit term
        // changed what every network's Train does, and neither master nor PyTorch applies it by default.
        if (!gradientBased.RegularizationExplicitlyConfigured) return null;
        // A proximal optimizer applies its regularizer inside its own step; adding it here would apply it twice.
        if (gradientBased.AppliesRegularizationInStep) return null;
        var regularization = gradientBased.ActiveRegularization;
        return regularization is null or AiDotNet.Regularization.NoRegularization<T, Tensor<T>, Tensor<T>> ? null : regularization;
    }

    /// <summary>
    /// The one recovery policy for a committed plan whose step failed (<see cref="FusedStepOutcome.CommittedFailure"/>).
    /// </summary>
    /// <remarks>
    /// A device fault says nothing about the model, so the plan is dropped for this run (sticky, with a warning that
    /// names the cause) and the kind is returned: the caller continues on the eager tape and adds only what it alone
    /// can do, as a network switching to streaming training on out-of-memory. Any other failure throws, because the
    /// plan's optimizer moments cannot move to the eager optimizer and a silent switch would change the trajectory.
    /// </remarks>
    public CommittedFailureKind DropAfterCommittedFailure()
    {
        var cause = LastFallbackException;
        CommittedFailureKind kind;
        if (TapeTrainingStepper<T>.IsGpuOutOfMemoryFailure(cause))
            kind = CommittedFailureKind.DeviceOutOfMemory;
        else if (TapeTrainingStepper<T>.IsGpuTransientFailure(cause))
            kind = CommittedFailureKind.DeviceTransient;
        else
            throw TapeTrainingStepper<T>.CommittedPlanCannotContinue(cause);

        Warn("committed fused plan dropped after a device failure (" + kind + "), training continues on the eager "
            + "tape with fresh optimizer moments: "
            + (cause is null ? "no exception" : cause.GetType().Name + ": " + cause.Message));
        Reset(stickyDisable: true);
        return kind;
    }
    private FusedStepOutcome Decline(string reason)
    {
        LastMissReason = reason;
        return FusedStepOutcome.Declined;
    }

    // Unsupported settings cannot transfer a committed plan's moments to the eager optimizer. Refuse this step
    // without changing the plan, so restoring supported settings can continue the same optimizer trajectory.
    private FusedStepOutcome RejectUnsupportedConfiguration(string reason)
    {
        if (IsCommitted)
        {
            LastMissReason = reason;
            LastFallbackException = new InvalidOperationException("Unsupported configuration after fused steps ran: " + reason);
            return FusedStepOutcome.CommittedFailure;
        }
        return Decline(reason);
    }

    private void Warn(string message)
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_QUIET")))
            System.Diagnostics.Trace.TraceWarning("[AiDotNet] " + message + " (model: " + _owner.GetType().Name + ").");
    }

    // A caught exception goes to the trace listeners whole (type, message and stack), whatever AIDOTNET_QUIET says:
    // without it a developer cannot tell a forward that legitimately needs a tape from a defect in the model.
    private void WarnWithException(string message, Exception exception)
        => System.Diagnostics.Trace.TraceWarning("[AiDotNet] " + message + " (model: " + _owner.GetType().Name + "): "
            + exception);

    private static bool StepCouldHaveMovedParameters(
        Optimizers.Fused.FusedOptimizerConfig config,
        bool gradientsObserved,
        bool anyGradientNonZero,
        bool planConfirmedAttached)
    {
        if (config.UpdateCanBeExactlyZero && planConfirmedAttached) return false;
        if (gradientsObserved && !anyGradientNonZero) return false;
        if (config.Schedule is null)
            return config.LearningRate != 0f;
        // The plan evaluates its schedule at its own 1-based step; when that step is known, so is this step's rate.
        return !CompiledTapeTrainingStep<T>.TryGetPlanOptimizerStep(out int step)
            || config.Schedule.GetLr(step) != 0.0;
    }

    // Reduced on the engine to ONE scalar, like AnySampleChanged: reading each gradient on the host downloaded the
    // whole gradient set every probed step (on a GPU MLP, most of the per-step device-to-host copies).
    // The eager tape's gradient of the request's loss, with the same random draws suppressed as the replay check.
    private static bool EagerGradientNonZero(FusedTrainingStepRequest<T> request)
    {
        using var noDraws = CompiledStepRandom<T>.SuppressDraws();
        using var tape = new GradientTape<T>();
        var loss = request.ComputeLoss(request.Forward(request.Input), request.Target);
        var parameters = ((IEnumerable<Tensor<T>>?)request.Selection ?? EnumerateLiveParameters(request)).ToArray();
        return AnyGradientNonZero(tape.ComputeGradients(loss, parameters));
    }

    // A host-resident gradient is scanned in place, and its first nonzero element answers. A device gradient is reduced
    // by the engine's sum of squares, which squares into one transient scratch buffer and returns the scalar, and the
    // first nonzero one answers - usually the first gradient, so one readback. Squaring every gradient with
    // TensorMultiply kept a gradient-sized product per parameter alive until its finalizer ran: 36 MB on the GPU
    // ResNet-50 at its peak, about 30 MB on a 7.2M-parameter N-BEATS. A small gradient also squared to zero in float.
    private static bool AnyGradientNonZero(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        foreach (var gradient in gradients.Values)
        {
            if (gradient is null || gradient.Length == 0) continue;
            if (gradient is not SparseTensor<T> && !gradient.HasPendingGpuData && gradient.TryGetGpuBuffer() is null)
            {
                if (HostAnyNonZero(gradient)) return true;
                continue;
            }
            // An exact zero test: a nonzero sum means the gradient reached the parameters. NaN counts as non-zero (NaN
            // never Equals 0), since the gradient did reach them, as non-finite values the plan's guard handles.
            // Through TensorReductionHelper: the engine op's double kernel in AiDotNet.Tensors 0.150.0 drops the
            // odd-index squares, so a double gradient whose non-zero elements all sit at odd indices summed to exactly
            // 0 and read as "no gradient". A float device gradient still reduces on the device (the helper passes it
            // to the engine op); a double one has no device reduction, so the engine downloaded it anyway.
            if (!NumOps.ToDouble(TensorReductionHelper.SumOfSquares(engine, gradient)).Equals(0.0)) return true;
        }
        return false;
    }

    // Exact, like the engine reduction: -0 is zero, and NaN is non-zero (the gradient reached the parameters).
    private static bool HostAnyNonZero(Tensor<T> gradient)
    {
        var values = (gradient.IsContiguous ? gradient : gradient.Contiguous()).AsSpan();
        var comparer = EqualityComparer<T>.Default;
        T zero = NumOps.Zero;
        for (int i = 0; i < values.Length; i++)
        {
            if (!comparer.Equals(values[i], zero)) return true;
        }
        return false;
    }

    // The trained tensors a snapshot can hold, in the order the eager step and the plan collect them.
    private static Tensor<T>[] LiveTrainedTensors(FusedTrainingStepRequest<T> request)
    {
        var trained = request.Selection is not null
            ? request.Selection
            : EnumerateLiveParameters(request).Distinct(TensorReferenceComparer<Tensor<T>>.Instance).ToList();
        return trained.Where(p => p is not null && p.Length != 0 && p is not SparseTensor<T>).ToArray();
    }

    // What a training step advances without the optimizer: the request's StepState and every registered buffer of its
    // layers and their sub-layers (a batch norm's running statistics, a quantizer's EMA codebooks), less anything
    // that is trained.
    private static Tensor<T>[] LiveStepState(FusedTrainingStepRequest<T> request, Tensor<T>[] trained)
    {
        var seen = new HashSet<Tensor<T>>(trained, TensorReferenceComparer<Tensor<T>>.Instance);
        var state = new List<Tensor<T>>();
        void Add(Tensor<T>? tensor)
        {
            if (tensor is not null && tensor.Length != 0 && tensor is not SparseTensor<T> && seen.Add(tensor))
                state.Add(tensor);
        }

        if (request.StepState is { } stepState)
            foreach (var tensor in stepState()) Add(tensor);
        var visited = new HashSet<ILayer<T>>(TensorReferenceComparer<ILayer<T>>.Instance);
        void Walk(ILayer<T> layer)
        {
            if (!visited.Add(layer)) return;
            if (layer is NeuralNetworks.Layers.LayerBase<T> concrete)
                foreach (var (_, buffer) in concrete.GetRegisteredBuffers()) Add(buffer);
            foreach (var child in layer.GetSubLayers()) Walk(child);
        }

        foreach (var layer in request.Layers)
            if (layer is ILayer<T> composite) Walk(composite);
        return state.ToArray();
    }

    private static StepSnapshot TakeSnapshot(FusedTrainingStepRequest<T> request)
    {
        var trained = LiveTrainedTensors(request);
        return new StepSnapshot { Weights = Capture(trained), State = Capture(LiveStepState(request, trained)) };
    }

    private static (Tensor<T> Live, Tensor<T> Saved)[] Capture(Tensor<T>[] live)
    {
        var saved = TensorCheckpoint<T>.Capture(live);
        return live.Select((tensor, i) => (tensor, saved[i])).ToArray();
    }

    private static void Restore(StepSnapshot snapshot)
    {
        RestoreTrainedTensors(snapshot.Weights);
        if (snapshot.State.Length > 0) RestoreTrainedTensors(snapshot.State);
    }

    // Whether the saved tensors are still the live ones: the same instances with the same lengths, compared against
    // the live list (no copy is taken to compare). Step state is not part of the question: a buffer a layer first
    // registered inside the trace has no earlier value to go back to, and that must not keep the weights and the
    // buffers that do have one from being restored.
    private static bool SameLive((Tensor<T> Live, Tensor<T> Saved)[] saved, Tensor<T>[] live)
    {
        if (saved.Length != live.Length) return false;
        for (int i = 0; i < live.Length; i++)
            if (!ReferenceEquals(saved[i].Live, live[i]) || saved[i].Saved.Length != live[i].Length) return false;
        return true;
    }

    private static bool SameTensors((Tensor<T> Live, Tensor<T> Saved)[] a, (Tensor<T> Live, Tensor<T> Saved)[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (!ReferenceEquals(a[i].Live, b[i].Live) || a[i].Saved.Length != b[i].Saved.Length) return false;
        return true;
    }

    // The default eager redo of a trace step: the eager step on copies of its input and target, taken now, since the
    // caller may refill those tensors for later steps.
    private static Func<T> EagerRedoOnCopies(FusedTrainingStepRequest<T> request)
    {
        var copies = TensorCheckpoint<T>.Capture(ReferenceEquals(request.Input, request.Target)
            ? new[] { request.Input }
            : new[] { request.Input, request.Target });
        var redo = new FusedTrainingStepRequest<T>
        {
            Layers = request.Layers,
            Input = copies[0],
            Target = copies.Length > 1 ? copies[1] : copies[0],
            Forward = request.Forward,
            ComputeLoss = request.ComputeLoss,
            Optimizer = request.Optimizer,
            ExtraParameters = request.ExtraParameters,
            Selection = request.Selection,
            ModelGradientClip = request.ModelGradientClip,
            ModelGradientClipScope = request.ModelGradientClipScope,
            ExcludesParametersFromWeightDecay = request.ExcludesParametersFromWeightDecay,
            OnGradients = request.OnGradients,
            RegularizeGradients = request.RegularizeGradients,
        };
        return () => TapeTrainingStepper<T>.EagerStep(redo);
    }

    private static void RestoreTrainedTensors((Tensor<T> Live, Tensor<T> Saved)[] snapshot)
        => TensorCheckpoint<T>.Restore(snapshot.Select(p => p.Live).ToArray(), snapshot.Select(p => p.Saved).ToArray());

    // The tensors the request trains, by the walk the plan's own collection and the eager step use
    // (CompiledTapeTrainingStep.CollectDeduplicatedParametersWithExtras, TapeTrainingStepper.TrainedParameters): a
    // layer's parameters include its sub-layers'. Listing only each layer's own tensors left out everything a composite
    // layer holds in children (a Swin block owns its bias table; its norms, attention and MLP are sub-layers), so the
    // plan's optimized tensors were not all "live" and a step that had updated the weights was reported as detached.
    private static IEnumerable<Tensor<T>> EnumerateLiveParameters(FusedTrainingStepRequest<T> request)
        // Include composite children, exactly as the eager and compiled training steps do.
        => TapeTrainingStepper<T>.TrainedParameters(request);

    // The #1822 persistence probe. Before the step it copies the first elements of every trained tensor (about
    // ChecksumTargetSamples in total, at least one per tensor) on whatever device holds the tensor; after the step it
    // reduces the squared difference to ONE scalar, and only that scalar reaches the host. A host checksum over the
    // tensors instead downloaded every GPU-resident parameter before and after each checked step (#1804: four times
    // the step's allocation on N-BEATS). Any real update moves a sampled element: Adam, SGD and the rest update every
    // parameter whose gradient is nonzero, and a plan decoupled from the live tensors moves none.
    private sealed class PersistenceProbe : IDisposable
    {
        private readonly IEngine _engine;
        private (Tensor<T> Source, int Count, Tensor<T> Before)[]? _samples;
        public PersistenceProbe(IEngine engine, (Tensor<T> Source, int Count, Tensor<T> Before)[] samples)
        { _engine = engine; _samples = samples; }
        public int Length => _samples?.Length ?? 0;
        public (Tensor<T> Source, int Count, Tensor<T> Before)[] Samples
            => _samples ?? throw new ObjectDisposedException(nameof(PersistenceProbe));
        public void Dispose()
        {
            var samples = _samples;
            _samples = null;
            if (samples is null) return;
            foreach (var sample in samples) ReleaseProbeSnapshot(_engine, sample.Source, sample.Before);
        }
    }

    private static void ReleaseProbeSnapshot(IEngine engine, Tensor<T> source, Tensor<T> before)
    {
        if (ReferenceEquals(before, source)) return;
        try
        {
            // Deferred results can belong to the engine's activation cache rather than their tensor.
            // Only the private copied snapshot is dead; its caller-owned parameter remains live.
            if (!before.SharesStorageWith(source) && engine.TryGet<DirectGpuTensorEngine>(out var gpu))
            {
                if (before.DataVector._deviceState is { Buffer: not null })
                    gpu.ReleaseDeadDeviceStorage(before, afterQueuedWork: true);
                else
                {
                    object vector = before.DataVector;
                    var array = before.GetBackingArrayForCacheLookupUnsafe();
                    gpu.EvictActivationsCreatedAfter(0, protect: null, mode: ActivationReleaseMode.Release,
                        releasableOnly: key => ReferenceEquals(key, vector) || (array is not null && ReferenceEquals(key, array)),
                        afterQueuedWork: true);
                }
            }
        }
        finally { before.Dispose(); }
    }

    private static PersistenceProbe SampleParameters(FusedTrainingStepRequest<T> request)
    {
        var parameters = request.Selection is not null
            ? request.Selection
            : EnumerateLiveParameters(request).ToList();
        int perTensor = Math.Max(1, ChecksumTargetSamples / Math.Max(1, parameters.Count));
        var engine = AiDotNetEngine.Current;
        var samples = new List<(Tensor<T>, int, Tensor<T>)>(parameters.Count);
        using var noGrad = new NoGradScope<T>();
        using var scratch = TensorScope.Begin();
        try
        {
            foreach (var parameter in parameters)
            {
                if (parameter.Length == 0) continue;
                int count = Math.Min(perTensor, parameter.Length);
                var before = engine.TensorMultiplyScalar(Head(engine, parameter, count), NumOps.One);
                samples.Add((parameter, count, before));
                scratch.Keep(before);
            }
            return new PersistenceProbe(engine, samples.ToArray());
        }
        catch
        {
            foreach (var sample in samples) ReleaseProbeSnapshot(engine, sample.Item1, sample.Item3);
            throw;
        }
    }

    private static bool AnySampleChanged(PersistenceProbe samples)
    {
        if (samples.Length == 0) return false;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        using var scratch = TensorScope.Begin();
        Tensor<T>? total = null;
        foreach (var (source, count, before) in samples.Samples)
        {
            var delta = engine.TensorSubtract(Head(engine, source, count), before);
            var squares = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(delta, delta), null), new[] { 1 });
            total = total is null ? squares : engine.TensorAdd(total, squares);
        }
        double changed = total is null ? 0.0 : NumOps.ToDouble(total[0]);
        // A sum of squares: positive when anything moved, NaN when the update reached the tensor as non-finite values (the
        // plan's own guard handles those), never negative.
        return changed > 0.0 || double.IsNaN(changed);
    }

    private static Tensor<T> Head(IEngine engine, Tensor<T> tensor, int count)
    {
        // A sparse parameter cannot be reshaped; its trainable values are its stored non-zeros, so sample those.
        if (tensor is SparseTensor<T> sparse)
        {
            int take = Math.Min(count, sparse.NonZeroCount);
            var head = new T[take];
            Array.Copy(sparse.Values, head, take);
            return new Tensor<T>(head, new[] { take });
        }
        // A streamed weight holds an empty vector until it is paged back (WeightRegistry.RegisterWeight), and a
        // Reshape inspects the storage before any kernel would page it in: OpenVocabSAM, which streams at its
        // default size, threw "View exceeds storage bounds" from this probe on its first training step. Page it in
        // first, as LayerBase does at a layer boundary.
        if (tensor.Lifetime == WeightLifetime.Streaming)
            WeightRegistry.Materialize(tensor);
        return engine.TensorNarrow(engine.Reshape(tensor, new[] { tensor.Length }), 0, 0, count);
    }
}
