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

/// <summary>One training step as the fused session sees it: what to train, on what, and how.</summary>
internal sealed class FusedTrainingStepRequest<T>
{
    public required IReadOnlyList<ITrainableLayer<T>> Layers { get; init; }
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

    /// <summary>Receives the step's gradients while their buffers are valid (the model's public gradient surface).</summary>
    public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnGradients { get; init; }

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

    /// <summary>
    /// Check, on the plan's first replay with new data, that the compiled plan computes the same loss as the eager
    /// forward on that data; on disagreement the replay's update is undone and the model stays on the eager tape.
    /// </summary>
    /// <remarks>
    /// For base classes whose models' training forwards were never audited for replay (a host-side read of a tensor
    /// value is frozen into the trace; see <see cref="GraphBreakReason"/>). The check costs one extra no-grad forward
    /// and one copy of the trained tensors, once per plan. A forward with its own randomness (dropout) disagrees by
    /// construction and conservatively stays eager.
    /// </remarks>
    public bool VerifyReplayAgreement { get; init; }
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
    private readonly Action? _onReset;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _lastOptimizer;
    private bool _persistenceVerified;
    private int _stepsSincePersistenceCheck;
    private bool _replayAgreementVerified;

    // Relative tolerance for the replay-agreement check: fusion reassociates sums, so float losses differ in the
    // last few digits; a plan replaying stale host-side decisions differs at the scale of the loss itself.
    private const double ReplayAgreementTolerance = 1e-3;

    /// <summary>Creates the fused lifecycle for one model.</summary>
    /// <param name="owner">The model; keys its compiled plan and optimizer moments.</param>
    /// <param name="onReset">Owner cleanup to run when a committed plan is dropped (caches, GPU transients).</param>
    public FusedTrainingSession(object owner, Action? onReset = null)
    {
        Guard.NotNull(owner);
        _owner = owner;
        _onReset = onReset;
    }

    /// <summary>True once a fused step has trained the model; later steps must stay fused or fail loudly.</summary>
    public bool IsCommitted { get; private set; }

    /// <summary>True when the fused path is off for this model (it proved unusable, or the model opts out);
    /// steps go to the eager tape.</summary>
    public bool IsDisabled { get; set; }

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
        CompiledTapeTrainingStep<T>.Invalidate(_owner);
        IsDisabled = stickyDisable;
        IsCommitted = false;
        _persistenceVerified = false;
        _stepsSincePersistenceCheck = 0;
        _replayAgreementVerified = false;
    }

    /// <summary>Re-enables the fused path after an explicit model reset (new layers, new optimizer).</summary>
    public void Restart(bool disabled = false)
    {
        IsDisabled = disabled;
        IsCommitted = false;
        _persistenceVerified = false;
        _stepsSincePersistenceCheck = 0;
        _replayAgreementVerified = false;
        LastMissReason = null;
        LastFallbackException = null;
    }

    /// <summary>Runs one fused step, or says why it cannot.</summary>
    public FusedStepOutcome TryStep(FusedTrainingStepRequest<T> request, out T loss)
    {
        Guard.NotNull(request);
        loss = NumOps.Zero;
        LastFallbackException = null;

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

        if (request.GraphBreakReason is { } graphBreak)
        {
            if (!IsCommitted)
                return Decline("graph break: " + graphBreak);
            // The plan holds the optimizer's moments; an eager step now would restart them silently.
            LastFallbackException = new InvalidOperationException("graph break after fused steps ran: " + graphBreak);
            return FusedStepOutcome.CommittedFailure;
        }
            return Decline("fused path sticky-disabled from a prior fallback");
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return Decline("TensorCodecOptions.EnableCompilation = false");
        if (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            return Decline($"numeric type {typeof(T).Name} not supported by fused kernel");

        var optimizer = request.Optimizer;
        if (!TryMapToFusedOptimizerConfig(optimizer, out var config))
            return Decline($"optimizer {optimizer.GetType().Name} not compatible with fused kernel");
        if (request.Layers.Count == 0 && (request.ExtraParameters is null || request.ExtraParameters.Count == 0))
            return Decline("no trainable layers");

        // The optimizer's regularization, applied the way its flat path applies it: to the gradient, BEFORE clipping.
        // L2 has a fused form (the plan adds strength * theta ahead of its clip); any other regularizer runs on the
        // eager tape. Every decline is decided BEFORE the per-step arena below exists: TensorArena.Create is
        // thread-static and stacks on the enclosing arena, so a decline after it would leave it current for the
        // eager fallback and every later step.
        double l2 = 0.0;
        if (OptimizerRegularizationOf(optimizer) is { } regularization)
        {
            if (regularization is AiDotNet.Regularization.L2Regularization<T, Tensor<T>, Tensor<T>> l2Regularization)
                l2 = l2Regularization.GetOptions().Strength;
            else
                return DeclineAndDropCommitted("regularization " + regularization.GetType().Name
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
            return DeclineAndDropCommitted("optimizer clips gradients by value; the fused plan clips only by global norm");
        if (optimizerClip > 0.0)
            clip = clip > 0.0 ? Math.Min(clip, optimizerClip) : optimizerClip;

        bool verifyPersistence = (!_persistenceVerified && !IsCommitted)
            || ++_stepsSincePersistenceCheck >= PersistenceRecheckInterval;
        var probe = verifyPersistence ? SampleParameters(request) : null;
        bool gradientsObserved = false;
        bool gradientNonZero = false;
        void OnGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
        {
            if (verifyPersistence)
            {
                gradientsObserved = true;
                gradientNonZero = AnyGradientNonZero(gradients);
            }
            request.OnGradients?.Invoke(gradients);
        }

        // The first replay with new data (the step after the trace) is checked against the eager forward when the
        // caller asks: the eager loss on this batch, and a copy of the trained tensors to undo a disagreeing update.
        bool checkReplay = request.VerifyReplayAgreement && IsCommitted && !_replayAgreementVerified;
        double eagerLoss = 0.0;
        (Tensor<T> Live, Tensor<T> Saved)[]? snapshot = null;
        if (checkReplay)
        {
            using var noGrad = new NoGradScope<T>();
            using var noDraws = CompiledStepRandom<T>.SuppressDraws();
            var eager = request.ComputeLoss(request.Forward(request.Input), request.Target);
            eagerLoss = eager.Length > 0 ? NumOps.ToDouble(eager[0]) : 0.0;
            snapshot = SnapshotTrainedTensors(request);
        }

        // Each step's transient activations are reclaimed when it returns, the way PyTorch's caching allocator
        // returns an iteration's blocks; the plan's moments and persistent input/target are not arena-allocated
        // (#1624 / #1640). Disposed before result handling so a fallback starts from a clean ring.
        bool ran;
        T stepLoss;
        var stepArena = TensorArena.Create();
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
                useBf16Moments: config.UseBf16Moments,
                extraTensors: request.ExtraParameters,
                fusedExtras: config.Extras,
                onGradients: OnGradients,
                trainableSelection: request.Selection,
                owner: _owner,
                l2Regularization: l2);
        }
        finally
        {
            stepArena.Dispose();
        }

        if (!ran)
        {
            LastFallbackException = CompiledTapeTrainingStep<T>.GetLastFallbackException();
            if (IsCommitted)
                return FusedStepOutcome.CommittedFailure;
            IsDisabled = true;
            return Decline("compiled plan declined the step ("
                + (LastFallbackException is null ? "no exception" : LastFallbackException.GetType().Name + ": " + LastFallbackException.Message)
                + ")");
        }

        if (checkReplay && snapshot is not null)
        {
            double replayLoss = NumOps.ToDouble(stepLoss);
            double scale = Math.Max(1.0, Math.Max(Math.Abs(eagerLoss), Math.Abs(replayLoss)));
            bool agree = (double.IsNaN(eagerLoss) && double.IsNaN(replayLoss))
                || Math.Abs(eagerLoss - replayLoss) <= ReplayAgreementTolerance * scale;
            if (!agree)
            {
                RestoreTrainedTensors(snapshot);
                string reason = "the compiled plan's loss on new data (" + replayLoss.ToString("G6")
                    + ") disagrees with the eager forward (" + eagerLoss.ToString("G6") + "): the training forward "
                    + "depends on more than its input tensor (a host-side read or captured per-step state), so the "
                    + "replay's update was undone and training continues on the eager tape";
                Warn(reason);
                Reset(stickyDisable: true);
                return Decline(reason);
            }
            _replayAgreementVerified = true;
        }

        if (verifyPersistence)
        {
            bool persisted = probe is not null && AnySampleChanged(probe);
            bool? attached = CompiledTapeTrainingStep<T>.ConfiguredPlanTrainsLiveParameters(
                (IEnumerable<Tensor<T>>?)request.Selection ?? EnumerateLiveParameters(request));
            bool detached = attached == false;
            // No sample means no trained tensor held an element before the step: lazily shaped weights materialize on
            // the first forward, so there was nothing to compare against. That is inconclusive, not a failed update.
            bool sampled = probe is { Length: > 0 };
            if (!persisted && !detached
                && (!sampled || !StepCouldHaveMovedParameters(config, gradientsObserved, gradientNonZero, attached == true)))
            {
                // Nothing could have moved (zero gradients, zero learning rate) or nothing was sampled: inconclusive,
                // check again next step.
                _stepsSincePersistenceCheck = PersistenceRecheckInterval;
            }
            else if (!persisted || detached)
            {
                string reason = "fused compiled step ran (loss " + NumOps.ToDouble(stepLoss).ToString("G6")
                    + ") but did not persist a parameter update; the plan is decoupled from the model's live "
                    + "parameter tensors, so training continues on the eager tape (ooples/AiDotNet#1822)";
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
            }
        }

        IsCommitted = true;
        LastMissReason = null;
        loss = stepLoss;
        if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
            scheduled.OnBatchEnd();
        return FusedStepOutcome.Stepped;
    }

    /// <summary>Maps an optimizer to its fused kernel configuration, when it has one.</summary>
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

    // A condition that rules the fused path out for the whole run: a committed plan has to be dropped, since its
    // moments cannot continue on the eager optimizer.
    private FusedStepOutcome DeclineAndDropCommitted(string reason)
    {
        if (IsCommitted)
        {
            Warn("committed fused plan dropped: " + reason);
            Reset(stickyDisable: true);
        }
        return Decline(reason);
    }

    private void Warn(string message)
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_QUIET")))
            System.Diagnostics.Trace.TraceWarning("[AiDotNet] " + message + " (model: " + _owner.GetType().Name + ").");
    }

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
    private static bool AnyGradientNonZero(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        Tensor<T>? total = null;
        foreach (var gradient in gradients.Values)
        {
            if (gradient is null || gradient.Length == 0) continue;
            var squares = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(gradient, gradient), null), new[] { 1 });
            total = total is null ? squares : engine.TensorAdd(total, squares);
        }
        // NaN counts as non-zero: the gradient reached the parameters (as non-finite values the plan's guard handles).
        return total is not null && NumOps.ToDouble(total[0]) != 0.0;
    }

    private static (Tensor<T> Live, Tensor<T> Saved)[] SnapshotTrainedTensors(FusedTrainingStepRequest<T> request)
    {
        var engine = AiDotNetEngine.Current;
        var trained = request.Selection is not null
            ? request.Selection
            : EnumerateLiveParameters(request).Distinct(TensorReferenceComparer<Tensor<T>>.Instance).ToList();
        var saved = new List<(Tensor<T>, Tensor<T>)>(trained.Count);
        foreach (var parameter in trained)
        {
            if (parameter is null || parameter.Length == 0 || parameter is SparseTensor<T>) continue;
            saved.Add((parameter, engine.TensorMultiplyScalar(parameter, NumOps.One)));
        }
        return saved.ToArray();
    }

    private static void RestoreTrainedTensors((Tensor<T> Live, Tensor<T> Saved)[] snapshot)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        foreach (var (live, saved) in snapshot)
        {
            engine.TensorCopy(saved, live);
            live.IncrementVersion();
            engine.InvalidatePersistentTensor(live);
        }
    }

    private static IEnumerable<Tensor<T>> EnumerateLiveParameters(FusedTrainingStepRequest<T> request)
    {
        foreach (var layer in request.Layers)
            foreach (var parameter in layer.GetTrainableParameters())
                if (parameter is not null) yield return parameter;
        if (request.ExtraParameters is null) yield break;
        foreach (var parameter in request.ExtraParameters) yield return parameter;
    }

    // The #1822 persistence probe. Before the step it copies the first elements of every trained tensor (about
    // ChecksumTargetSamples in total, at least one per tensor) on whatever device holds the tensor; after the step it
    // reduces the squared difference to ONE scalar, and only that scalar reaches the host. A host checksum over the
    // tensors instead downloaded every GPU-resident parameter before and after each checked step (#1804: four times
    // the step's allocation on N-BEATS). Any real update moves a sampled element: Adam, SGD and the rest update every
    // parameter whose gradient is nonzero, and a plan decoupled from the live tensors moves none.
    private static (Tensor<T> Source, int Count, Tensor<T> Before)[] SampleParameters(FusedTrainingStepRequest<T> request)
    {
        var parameters = request.Selection is not null
            ? request.Selection
            : EnumerateLiveParameters(request).ToList();
        int perTensor = Math.Max(1, ChecksumTargetSamples / Math.Max(1, parameters.Count));
        var engine = AiDotNetEngine.Current;
        var samples = new List<(Tensor<T>, int, Tensor<T>)>(parameters.Count);
        using var noGrad = new NoGradScope<T>();
        foreach (var parameter in parameters)
        {
            if (parameter.Length == 0) continue;
            int count = Math.Min(perTensor, parameter.Length);
            samples.Add((parameter, count, engine.TensorMultiplyScalar(Head(engine, parameter, count), NumOps.One)));
        }
        return samples.ToArray();
    }

    private static bool AnySampleChanged((Tensor<T> Source, int Count, Tensor<T> Before)[] samples)
    {
        if (samples.Length == 0) return false;
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        Tensor<T>? total = null;
        foreach (var (source, count, before) in samples)
        {
            var delta = engine.TensorSubtract(Head(engine, source, count), before);
            var squares = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(delta, delta), null), new[] { 1 });
            total = total is null ? squares : engine.TensorAdd(total, squares);
        }
        double changed = total is null ? 0.0 : NumOps.ToDouble(total[0]);
        // NaN means the update reached the tensor (as non-finite values, which the plan's own guard handles).
        return changed != 0.0;
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
        return engine.TensorNarrow(engine.Reshape(tensor, new[] { tensor.Length }), 0, 0, count);
    }
}