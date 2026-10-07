using System.Runtime.CompilerServices;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Training;

/// <summary>
/// One training step for any model that trains on the gradient tape: the fused compiled plan when it applies (CPU
/// or GPU), otherwise the eager tape with the model's optimizer. Model base classes own one (or reach theirs through
/// <see cref="ForOwner"/>) and call <see cref="Step"/> per batch, so no model writes its own tape loop or its own
/// GPU-resident path.
/// </summary>
/// <remarks>
/// <para>
/// The eager branch is the loop the time-series, diffusion, VAE, vision and generative models each carried a copy
/// of: record the objective on a tape, take the gradients of the trained tensors, hand them to
/// <see cref="IGradientBasedOptimizer{T, TInput, TOutput}.Step(TapeStepContext{T})"/>, advance the optimizer's
/// schedule. It lives here once: <see cref="EagerStep"/> for a forward and a loss, and the
/// <c>EagerObjectiveStep</c> overloads for an objective that is not a forward of one input tensor. The CI ratchet
/// <c>TapeTrainingLoopRatchetTests</c> keeps new copies from appearing. The fused branch is
/// <see cref="FusedTrainingSession{T}"/>.
/// </para>
/// <para>
/// The eager update runs INSIDE the tape's scope. Disposing the outermost tape rewinds the active
/// <c>TensorArena</c> (per-step recycling, AiDotNet #1804) and the gradients live in that arena, so an update applied
/// after the dispose reads storage that the update's own temporaries are already being handed. The model builder
/// trains inside an arena, so every copy of this loop that disposed its tape first was exposed to that.
/// </para>
/// </remarks>
internal sealed class TapeTrainingStepper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Steppers for models that do not hold one in a field: the static helpers (FusedTrainingStep, the vision
    // trainer) key them by the model so its plan, committed state and optimizer identity follow the model.
    private static readonly ConditionalWeakTable<object, TapeTrainingStepper<T>> ByOwner = new();

    /// <param name="owner">The model being trained; keys its compiled plan.</param>
    /// <param name="onReset">Owner cleanup when a committed fused plan is dropped.</param>
    public TapeTrainingStepper(object owner, Action? onReset = null)
    {
        Guard.NotNull(owner);
        _owner = owner;
        _onReset = onReset;
        Session = new FusedTrainingSession<T>(owner, onReset);
    }

    /// <summary>The stepper of <paramref name="owner"/>, created on first use and released with the owner.</summary>
    public static TapeTrainingStepper<T> ForOwner(object owner)
    {
        Guard.NotNull(owner);
        return ByOwner.GetValue(owner, static key => new TapeTrainingStepper<T>(key));
    }

    /// <summary>
    /// The fused lifecycle of the most recently stepped parameter group (the owner's only group for a model with one
    /// optimizer); exposed for diagnostics and explicit resets.
    /// </summary>
    public FusedTrainingSession<T> Session { get; private set; }

    private readonly object _owner;
    private readonly Action? _onReset;

    // One lane per parameter group the owner trains: a GAN steps its critic and its generator, each with its own
    // optimizer, through the same owner. Each lane holds its own compiled plan, committed state and optimizer
    // identity. With a single lane per owner, every alternation between the two optimizers dropped the other's plan
    // (and with it the fused optimizer's moments), so a GAN re-traced both plans and restarted Adam on every step.
    private sealed class Lane
    {
        public Lane(FusedTrainingSession<T> session) => Session = session;
        public FusedTrainingSession<T> Session { get; }
        public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? LastOptimizer { get; set; }
    }

    private Lane? _firstLane;
    private object? _firstLaneKey;
    private Dictionary<object, Lane>? _otherLanes;

    // The identity of a request's parameter group: its first trainable layer, else its first selected or extra tensor.
    private static object? ParameterGroupKey(FusedTrainingStepRequest<T> request)
    {
        if (request.Layers.Count > 0) return request.Layers[0];
        if (request.Selection is { Count: > 0 } selection) return selection[0];
        if (request.ExtraParameters is { Count: > 0 } extras) return extras[0];
        return null;
    }

    private Lane LaneFor(FusedTrainingStepRequest<T> request)
    {
        var key = ParameterGroupKey(request) ?? _owner;
        if (_firstLane is null)
        {
            // The first group keeps the owner as its plan key, exactly as a single-group model always has.
            _firstLane = new Lane(Session);
            _firstLaneKey = key;
            return _firstLane;
        }

        if (ReferenceEquals(key, _firstLaneKey))
            return _firstLane;

        _otherLanes ??= new Dictionary<object, Lane>(ReferenceKeyComparer.Instance);
        if (!_otherLanes.TryGetValue(key, out var lane))
        {
            // A distinct plan key per further group, so dropping one group's plan never drops another's.
            lane = new Lane(new FusedTrainingSession<T>(new object(), _onReset));
            _otherLanes[key] = lane;
        }

        return lane;
    }

    private sealed class ReferenceKeyComparer : IEqualityComparer<object>
    {
        public static readonly ReferenceKeyComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
    }

    /// <summary>Whether the most recent <see cref="Step"/> ran on the fused compiled plan.</summary>
    public bool LastStepFused { get; private set; }

    /// <summary>Trains one batch and returns its loss.</summary>
    public T Step(FusedTrainingStepRequest<T> request)
    {
        Guard.NotNull(request);
        if (TryFusedStep(request, out T fusedLoss))
        {
            LastStepFused = true;
            return fusedLoss;
        }

        LastStepFused = false;
        return EagerStep(request);
    }

    /// <summary>
    /// Runs the step on the fused compiled plan and returns true, or returns false when the fused path does not apply
    /// and the caller must take the eager step. Throws when a plan that already trained this owner cannot continue,
    /// unless the cause is a device out-of-memory or transient fault (then the plan is dropped and false returned).
    /// </summary>
    public bool TryFusedStep(FusedTrainingStepRequest<T> request, out T loss)
    {
        Guard.NotNull(request);
        // The compiled plan carries the optimizer's moments. A different optimizer instance (a model's next
        // Train call builds a fresh one) starts from its own fresh state, as it would eagerly, so the old plan is
        // dropped rather than treated as hyperparameter drift on a committed plan, which refuses the step.
        var lane = LaneFor(request);
        var session = lane.Session;
        Session = session;
        if (!ReferenceEquals(lane.LastOptimizer, request.Optimizer))
        {
            if (lane.LastOptimizer is not null)
                session.Reset(stickyDisable: false);
            lane.LastOptimizer = request.Optimizer;
        }

        var outcome = session.TryStep(request, out loss);
        if (outcome == FusedStepOutcome.Stepped)
            return true;

        if (outcome == FusedStepOutcome.CommittedFailure)
        {
            var cause = session.LastFallbackException;
            // Device memory pressure or a transient device fault: the plan is unusable for this run, but nothing
            // about the model is wrong, so continue on the eager tape from the current weights.
            if (!IsGpuOutOfMemoryFailure(cause) && !IsGpuTransientFailure(cause))
                throw CommittedPlanCannotContinue(cause);
            session.Reset(stickyDisable: true);
        }

        return false;
    }

    /// <summary>Forward, loss and backward on the tape, then the model's optimizer.</summary>
    public static T EagerStep(FusedTrainingStepRequest<T> request)
    {
        Guard.NotNull(request);
        foreach (var layer in request.Layers)
            layer.ZeroGrad();

        var forward = request.Forward;
        var computeLoss = request.ComputeLoss;
        var input = request.Input;
        var target = request.Target;
        return EagerTapeStep(
            () => computeLoss(forward(input), target),
            // Read after the forward: a lazily shaped layer allocates its weights on its first forward.
            () => TrainedParameters(request),
            OptimizerUpdate(request.Optimizer),
            input,
            target,
            (x, _) => forward(x),
            (predicted, y) => computeLoss(predicted, y),
            new EagerStepHooks { OnGradients = request.OnGradients, ModelGradientClip = request.ModelGradientClip });
    }

    /// <summary>
    /// One eager update of <paramref name="parameters"/> against an objective that is not a forward of one input
    /// tensor: an adversarial or contrastive loss, a multi-term loss over several networks, a typed task loss.
    /// </summary>
    /// <param name="parameters">The tensors to update; only those the objective reaches are stepped.</param>
    /// <param name="objective">Builds the scalar loss from the live parameters. It runs once under the tape and again
    /// whenever the optimizer re-evaluates (line search), so it must replay the same random draws each time.</param>
    /// <param name="optimizer">The update rule; its moments are keyed by tensor reference.</param>
    /// <param name="onGradients">Receives the gradients before clipping (the model's public gradient surface).</param>
    /// <param name="modelGradientClip">The model's own global-norm clip (0 = none), applied before the optimizer's.</param>
    /// <param name="beforeUpdate">Runs after the gradients exist and before the update (e.g. restoring full-precision
    /// shadow weights for quantization-aware training).</param>
    /// <returns>The objective's value before the update.</returns>
    public static T EagerObjectiveStep(
        IReadOnlyList<Tensor<T>> parameters,
        Func<Tensor<T>> objective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null,
        double modelGradientClip = 0.0,
        Action? beforeUpdate = null)
    {
        Guard.NotNull(parameters);
        return EagerObjectiveStep(() => parameters, objective, optimizer, onGradients, modelGradientClip, beforeUpdate);
    }

    /// <summary>
    /// The objective step with the parameters read AFTER the objective has run, for models whose lazy layers allocate
    /// their weights on the first forward.
    /// </summary>
    public static T EagerObjectiveStep(
        Func<IReadOnlyList<Tensor<T>>> parameters,
        Func<Tensor<T>> objective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null,
        double modelGradientClip = 0.0,
        Action? beforeUpdate = null)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(objective);
        Guard.NotNull(optimizer);
        var placeholder = new Tensor<T>(new[] { 1 });
        return EagerTapeStep(
            objective,
            parameters,
            OptimizerUpdate(optimizer),
            placeholder,
            placeholder,
            (_, _) => objective(),
            (value, _) => value,
            new EagerStepHooks { OnGradients = onGradients, ModelGradientClip = modelGradientClip, BeforeUpdate = beforeUpdate });
    }

    /// <summary>
    /// One eager plain-SGD update (<c>p -= learningRate * grad</c>) of the tensors <paramref name="objective"/>
    /// reaches, for the trainers that have no optimizer object. Parameters are read after the objective runs.
    /// </summary>
    public static T EagerSgdObjectiveStep(
        Func<IReadOnlyList<Tensor<T>>> parameters,
        Func<Tensor<T>> objective,
        T learningRate,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(objective);
        var placeholder = new Tensor<T>(new[] { 1 });
        return EagerTapeStep(
            objective,
            parameters,
            context =>
            {
                var engine = AiDotNetEngine.Current;
                using var noGrad = new NoGradScope<T>();
                foreach (var parameter in context.Parameters)
                    engine.TensorSubtractInPlace(parameter, engine.TensorMultiplyScalar(context.Gradients[parameter], learningRate));
            },
            placeholder,
            placeholder,
            (_, _) => objective(),
            (value, _) => value,
            new EagerStepHooks { OnGradients = onGradients });
    }

    /// <summary>
    /// One update from the MEAN gradient of <paramref name="microBatchCount"/> objectives, each recorded and
    /// differentiated on its own tape: gradient accumulation, so peak memory holds one micro-batch's graph instead of
    /// the whole batch's, while the optimizer takes a single step (its moments see the exact mini-batch gradient).
    /// </summary>
    /// <param name="parameters">The tensors to update; only those some micro-objective reaches are stepped.</param>
    /// <param name="microBatchCount">How many micro-objectives make up the batch (at least one).</param>
    /// <param name="microObjective">Builds the scalar loss of micro-batch <c>i</c> from the live parameters.</param>
    /// <param name="optimizer">The update rule; its moments are keyed by tensor reference.</param>
    /// <param name="onGradients">Receives the averaged gradients before the update.</param>
    /// <returns>The mean micro-objective value before the update.</returns>
    /// <remarks>
    /// Each micro-batch's gradients are copied into owned accumulators before its tape is disposed (the dispose
    /// rewinds the arena the gradients live in), summed in place, and scaled by 1/count, so the result equals the
    /// sum-then-scale a hand-written accumulation loop computes.
    /// </remarks>
    public static T EagerAccumulatedObjectiveStep(
        IReadOnlyList<Tensor<T>> parameters,
        int microBatchCount,
        Func<int, Tensor<T>> microObjective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(microObjective);
        Guard.NotNull(optimizer);
        Guard.Positive(microBatchCount);

        var engine = AiDotNetEngine.Current;
        var accumulated = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);
        var reached = new List<Tensor<T>>(parameters.Count);
        double lossSum = 0.0;
        for (int i = 0; i < microBatchCount; i++)
        {
            using var tape = new GradientTape<T>();
            var loss = microObjective(i);
            if (loss is null)
                throw new InvalidOperationException("A training objective returned null instead of a scalar loss tensor.");
            var gradients = tape.ComputeGradients(loss, parameters);
            lossSum += loss.Length > 0 ? NumOps.ToDouble(loss[0]) : 0.0;
            foreach (var parameter in parameters)
            {
                if (parameter is null || !gradients.TryGetValue(parameter, out var gradient))
                    continue;
                if (accumulated.TryGetValue(parameter, out var sum))
                {
                    engine.TensorAddInPlace(sum, gradient);
                }
                else
                {
                    // An owned copy: the gradient's own storage is recycled when this tape is disposed.
                    accumulated[parameter] = new Tensor<T>(gradient.ToArray(), gradient.Shape.ToArray());
                    reached.Add(parameter);
                }
            }
        }

        T inverseCount = NumOps.FromDouble(1.0 / microBatchCount);
        foreach (var sum in accumulated.Values)
            engine.TensorMultiplyScalarInPlace(sum, inverseCount);

        T meanLoss = NumOps.FromDouble(lossSum / microBatchCount);
        onGradients?.Invoke(accumulated);

        // A line-searching optimizer re-evaluates the batch objective: the mean of the micro-objectives.
        Tensor<T> MeanObjective()
        {
            Tensor<T>? total = null;
            for (int i = 0; i < microBatchCount; i++)
            {
                var loss = microObjective(i);
                total = total is null ? loss : engine.TensorAdd(total, loss);
            }
            return engine.TensorMultiplyScalar(total ?? new Tensor<T>(new[] { 1 }), inverseCount);
        }

        var placeholder = new Tensor<T>(new[] { 1 });
        OptimizerUpdate(optimizer)(new TapeStepContext<T>(
            reached, accumulated, meanLoss, placeholder, placeholder,
            (_, _) => MeanObjective(), (value, _) => value, parameterBuffer: null));
        return meanLoss;
    }

    private static Action<TapeStepContext<T>> OptimizerUpdate(IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
        => context =>
        {
            optimizer.Step(context);
            if (optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> scheduled)
                scheduled.OnBatchEnd();
        };

    /// <summary>
    /// Extension points of the shared eager step, for a base class that layers its own bookkeeping onto the one loop
    /// (gradient publication, a diagnostic reachability probe, regularization, a precision restore) instead of
    /// writing its own copy of it.
    /// </summary>
    internal sealed class EagerStepHooks
    {
        /// <summary>
        /// Widens the tensors the backward differentiates beyond the parameters (a diagnostic probe asking whether an
        /// input is reachable). Only the parameters are ever updated. Null differentiates exactly the parameters.
        /// </summary>
        public Func<IReadOnlyList<Tensor<T>>, IReadOnlyList<Tensor<T>>>? GradientSources { get; init; }

        /// <summary>Receives every gradient the backward produced, widened sources included, before any clip.</summary>
        public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnTapeGradients { get; init; }

        /// <summary>Receives the parameters' gradients before clipping (the model's public gradient surface).</summary>
        public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnGradients { get; init; }

        /// <summary>The model's own global-norm clip (0 = none), applied before the update.</summary>
        public double ModelGradientClip { get; init; }

        /// <summary>Runs after the gradients exist and before the update.</summary>
        public Action? BeforeUpdate { get; init; }
    }

    /// <summary>
    /// The single eager tape step every training path shares: record <paramref name="objective"/> on a tape, take the
    /// gradients of the parameters it reaches, and apply <paramref name="update"/>.
    /// </summary>
    /// <remarks>
    /// Everything from the backward to the update stays inside the tape's scope (see the type remarks: the outermost
    /// tape's dispose rewinds the arena holding the gradients). An optimizer's Step turns recording off itself and back
    /// on only while it re-evaluates the objective (line search), so it is not wrapped. The parameters are read AFTER
    /// the objective runs, so a lazily shaped layer's first-forward weights are trained on that same step.
    /// </remarks>
    /// <returns>The objective's value before the update.</returns>
    internal static T EagerTapeStep(
        Func<Tensor<T>> objective,
        Func<IReadOnlyList<Tensor<T>>> parameterProvider,
        Action<TapeStepContext<T>> update,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeForward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeLoss,
        EagerStepHooks? hooks = null)
    {
        Guard.NotNull(objective);
        Guard.NotNull(parameterProvider);
        Guard.NotNull(update);
        using var tape = new GradientTape<T>();
        var loss = objective();
        if (loss is null)
            throw new InvalidOperationException("A training objective returned null instead of a scalar loss tensor.");
        var parameters = parameterProvider();
        if (parameters is null)
            throw new InvalidOperationException("A training step's parameter provider returned null.");

        var sources = hooks?.GradientSources is { } widen ? widen(parameters) ?? parameters : parameters;
        var all = tape.ComputeGradients(loss, sources);
        hooks?.OnTapeGradients?.Invoke(all);
        var reached = new List<Tensor<T>>(parameters.Count);
        var gradients = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);
        foreach (var parameter in parameters)
        {
            if (parameter is not null && !gradients.ContainsKey(parameter) && all.TryGetValue(parameter, out var gradient))
            {
                gradients[parameter] = gradient;
                reached.Add(parameter);
            }
        }

        T lossValue = loss.Length > 0 ? loss[0] : NumOps.Zero;
        hooks?.OnGradients?.Invoke(gradients);
        if (hooks is { ModelGradientClip: > 0.0 })
            ClipByGlobalNorm(gradients, hooks.ModelGradientClip);
        hooks?.BeforeUpdate?.Invoke();

        update(new TapeStepContext<T>(
            reached, gradients, lossValue, input, target, recomputeForward, recomputeLoss, parameterBuffer: null));
        return lossValue;
    }

    /// <summary>A device out-of-memory failure, possibly wrapped.</summary>
    internal static bool IsGpuOutOfMemoryFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            if (e is OutOfMemoryException)
                return true;
            var message = e.Message;
            if (message.Contains("Out of memory", StringComparison.OrdinalIgnoreCase)
                && (message.Contains("cuMem", StringComparison.Ordinal)
                    || message.Contains("CUDA", StringComparison.OrdinalIgnoreCase)
                    || message.Contains("GPU", StringComparison.OrdinalIgnoreCase)
                    || message.Contains("CL_OUT_OF", StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>A transient device fault (driver, stream, launch, or a buffer released under the step).</summary>
    internal static bool IsGpuTransientFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            var message = e.Message;
            if (message.Contains("CUDA error", StringComparison.OrdinalIgnoreCase)
                || message.Contains("cuMem", StringComparison.Ordinal)
                || message.Contains("cuStream", StringComparison.Ordinal)
                || message.Contains("cuLaunch", StringComparison.Ordinal)
                || message.Contains("OpenCL", StringComparison.OrdinalIgnoreCase)
                || message.Contains("released before materialization", StringComparison.OrdinalIgnoreCase)
                || message.Contains("buffer was released", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>The error for a committed fused plan that cannot run the next step.</summary>
    internal static InvalidOperationException CommittedPlanCannotContinue(Exception? cause)
    {
        string rootCause = cause is not null
            ? $" Root-cause exception (caught in CompiledTapeTrainingStep): {cause.GetType().FullName}: {cause.Message}"
            : " (No exception was caught: the fused path returned false from one of the explicit refuse paths: plan "
              + "reference changed, optimizer hyperparameters drifted, TensorCodecOptions.EnableCompilation=false, or "
              + "numeric/optimizer type unsupported.)";
        return new InvalidOperationException(
            "Fused compiled training has already run successfully, but the current step cannot engage the fused "
            + "path. The plan-embedded optimizer state cannot be transferred to the eager optimizer, so falling back "
            + "silently would produce a trajectory that diverges from the previous fused steps. Common causes: "
            + "variable input/target shape (new compiled plan), LR scheduler or adaptive-rate changes, attached "
            + "AMSGrad, a step that declares a graph break after fused steps already ran, or a kernel-level exception "
            + "in plan.Step/ConfigureOptimizer that was caught and swallowed. "
            + "Resolution: keep shapes and optimizer hyperparameters stable across steps, OR reset the model's "
            + "training state, OR disable compilation "
            + "(AiModelBuilder.ConfigureJitCompilation(JitCompilationConfig.Disabled))." + rootCause,
            cause);
    }

    private static IReadOnlyList<Tensor<T>> TrainedParameters(FusedTrainingStepRequest<T> request)
    {
        if (request.Selection is not null)
            return request.Selection;
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var parameters = new List<Tensor<T>>();
        foreach (var layer in request.Layers)
            foreach (var parameter in layer.GetTrainableParameters())
                if (parameter is not null && seen.Add(parameter))
                    parameters.Add(parameter);
        if (request.ExtraParameters is not null)
            foreach (var parameter in request.ExtraParameters)
                if (parameter is not null && seen.Add(parameter))
                    parameters.Add(parameter);
        return parameters;
    }

    // PyTorch clip_grad_norm_ on the engine: scale every gradient by max / (norm + 1e-6) when the norm exceeds max.
    private static void ClipByGlobalNorm(Dictionary<Tensor<T>, Tensor<T>> gradients, double maxNorm)
    {
        var engine = AiDotNetEngine.Current;
        using var noGrad = new NoGradScope<T>();
        Tensor<T>? total = null;
        foreach (var gradient in gradients.Values)
        {
            if (gradient.Length == 0) continue;
            var squares = engine.Reshape(engine.ReduceSum(engine.TensorMultiply(gradient, gradient), null), new[] { 1 });
            total = total is null ? squares : engine.TensorAdd(total, squares);
        }
        if (total is null) return;
        double norm = Math.Sqrt(NumOps.ToDouble(total[0]));
        if (!(norm > maxNorm) || double.IsInfinity(norm)) return;
        T scale = NumOps.FromDouble(maxNorm / (norm + 1e-6));
        foreach (var gradient in gradients.Values)
            engine.TensorMultiplyScalarInPlace(gradient, scale);
    }
}
