using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

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
    private bool _persistenceVerified;
    private int _stepsSincePersistenceCheck;

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
    }

    /// <summary>Re-enables the fused path after an explicit model reset (new layers, new optimizer).</summary>
    public void Restart(bool disabled = false)
    {
        IsDisabled = disabled;
        IsCommitted = false;
        _persistenceVerified = false;
        _stepsSincePersistenceCheck = 0;
        LastMissReason = null;
        LastFallbackException = null;
    }

    /// <summary>Runs one fused step, or says why it cannot.</summary>
    public FusedStepOutcome TryStep(FusedTrainingStepRequest<T> request, out T loss)
    {
        Guard.NotNull(request);
        loss = NumOps.Zero;
        LastFallbackException = null;

        if (IsDisabled)
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
        double checksumBefore = verifyPersistence ? ParameterChecksum(request) : 0.0;
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

        if (verifyPersistence)
        {
            bool persisted = ParameterChecksum(request) != checksumBefore;
            bool? attached = CompiledTapeTrainingStep<T>.ConfiguredPlanTrainsLiveParameters(
                (IEnumerable<Tensor<T>>?)request.Selection ?? EnumerateLiveParameters(request));
            bool detached = attached == false;
            if (!persisted && !detached
                && !StepCouldHaveMovedParameters(config, gradientsObserved, gradientNonZero, attached == true))
            {
                // Nothing could have moved (zero gradients, zero learning rate): inconclusive, check again next step.
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

    private static bool AnyGradientNonZero(IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        foreach (var gradient in gradients.Values)
        {
            if (gradient is null) continue;
            var span = gradient.AsSpan();
            for (int i = 0; i < span.Length; i++)
                if (!NumOps.Equals(span[i], NumOps.Zero)) return true;
        }
        return false;
    }

    private static IEnumerable<Tensor<T>> EnumerateLiveParameters(FusedTrainingStepRequest<T> request)
    {
        foreach (var layer in request.Layers)
            foreach (var parameter in layer.GetTrainableParameters())
                if (parameter is not null) yield return parameter;
        if (request.ExtraParameters is null) yield break;
        foreach (var parameter in request.ExtraParameters) yield return parameter;
    }

    // Sum of squares over a strided sample of the trained tensors, capped near ChecksumTargetSamples elements: a full
    // generic-ToDouble sum costs seconds on the critical first step of a 385M-parameter model (#1822). The same
    // indices are read before and after, so != stays valid, and small models are covered completely.
    private static double ParameterChecksum(FusedTrainingStepRequest<T> request)
    {
        var parameters = request.Selection is not null
            ? request.Selection
            : EnumerateLiveParameters(request).ToList();
        long total = 0;
        foreach (var parameter in parameters) total += parameter.Length;
        if (total == 0) return 0.0;
        int stride = (int)Math.Max(1, total / ChecksumTargetSamples);
        double sum = 0.0;
        foreach (var parameter in parameters)
        {
            var span = parameter.AsSpan();
            for (int i = 0; i < span.Length; i += stride)
            {
                double value = NumOps.ToDouble(span[i]);
                sum += value * value;
            }
        }
        return sum;
    }
}