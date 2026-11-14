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

    /// <summary>Creates the stepper for one model, with its own fused session.</summary>
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
            lane = new Lane(new FusedTrainingSession<T>(_owner, _onReset, stateKey: new object()));
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

    /// <summary>Steps this stepper ran on the fused compiled plan.</summary>
    public int FusedStepCount { get; private set; }

    /// <summary>Steps this stepper ran on the eager tape.</summary>
    public int EagerStepCount { get; private set; }

    /// <summary>Trains one batch and returns its loss.</summary>
    public T Step(FusedTrainingStepRequest<T> request)
    {
        Guard.NotNull(request);
        if (TryFusedStep(request, out T fusedLoss))
        {
            LastStepFused = true;
            FusedStepCount++;
            return fusedLoss;
        }

        LastStepFused = false;
        EagerStepCount++;
        return EagerStep(request, _owner);
    }

    /// <summary>
    /// Runs the step on the fused compiled plan and returns true, or returns false when the fused path does not apply
    /// and the caller must take the eager step. Throws when a plan that already trained this owner cannot continue,
    /// unless the cause is a device out-of-memory or transient fault (then the plan is dropped and false returned).
    /// </summary>
    public bool TryFusedStep(FusedTrainingStepRequest<T> request, out T loss)
    {
        Guard.NotNull(request);
        // Each lane's session drops its plan when a different optimizer instance arrives (a new Train call builds a
        // fresh one), so a replacement never inherits the old plan's moments or reads as hyperparameter drift.
        var lane = LaneFor(request);
        var session = lane.Session;
        Session = session;

        var outcome = session.TryStep(request, out loss);
        if (outcome == FusedStepOutcome.Stepped)
            return true;

        // A device fault drops the plan and continues on the eager tape from the current weights; anything else
        // throws. The policy is the session's, shared with every other fused caller.
        if (outcome == FusedStepOutcome.CommittedFailure)
            session.DropAfterCommittedFailure();

        return false;
    }

    /// <summary>Forward, loss and backward on the tape, then the model's optimizer.</summary>
    public static T EagerStep(FusedTrainingStepRequest<T> request)
        => EagerStep(request, owner: null);

    /// <summary>Forward, loss and backward on the tape with explicit owner context for distributed/mixed-precision coordination.</summary>
    internal static T EagerStep(FusedTrainingStepRequest<T> request, object? owner)
    {
        Guard.NotNull(request);
        owner ??= ResolveOwnerFromDelegates(request.Forward, request.ComputeLoss, request.OnGradients);
        foreach (var layer in request.Layers)
            layer.ZeroGrad();

        // The request's members are init-only, so the lambdas read them directly rather than through copies.
        return EagerTapeStep(
            () => request.ComputeLoss(request.Forward(request.Input), request.Target),
            // Read after the forward: a lazily shaped layer allocates its weights on its first forward.
            () => TrainedParameters(request),
            OptimizerUpdate(request.Optimizer),
            request.Input,
            request.Target,
            (x, _) => request.Forward(x),
            (predicted, y) => request.ComputeLoss(predicted, y),
            new EagerStepHooks
            {
                Owner = owner,
                OnGradients = request.OnGradients,
                // The fused plan adds the optimizer's L2 on the same side of its clip; the eager step it falls back to
                // must too, or a model trains on a different objective whenever compilation declines.
                RegularizeGradients = request.RegularizeGradients ?? OptimizerRegularization(request.Optimizer),
                ModelGradientClip = request.ModelGradientClip,
                Optimizer = request.Optimizer,
            });
    }

    /// <summary>
    /// The optimizer's own regularization as a gradient update, applied the way its flat path applies it (L2 adds
    /// strength * theta), or null when it has none or applies it inside its own step.
    /// </summary>
    private static Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? OptimizerRegularization(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        if (FusedTrainingSession<T>.OptimizerRegularizationOf(optimizer) is not { } regularization) return null;
        return (parameters, gradients) =>
        {
            var engine = AiDotNetEngine.Current;
            // Recording off: theta is a trainable parameter, and recording g + lambda * theta would join the
            // optimizer's arithmetic to the next step's graph.
            using var noGrad = new NoGradScope<T>();
            foreach (var parameter in parameters)
            {
                if (!gradients.TryGetValue(parameter, out var gradient) || gradient is null) continue;
                var theta = parameter._shape.SequenceEqual(gradient._shape) ? parameter : engine.Reshape(parameter, gradient._shape);
                gradients[parameter] = regularization is AiDotNet.Regularization.L2Regularization<T, Tensor<T>, Tensor<T>> l2
                    ? engine.TensorAdd(gradient, engine.TensorMultiplyScalar(theta, NumOps.FromDouble(l2.GetOptions().Strength)))
                    : regularization.Regularize(gradient, theta);
            }
        };
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
    /// <param name="applyOptimizerRegularization">Whether the optimizer's own configured regularization (a paper's
    /// coupled weight decay as L2) is added to the gradients before the clip, as the fused session and
    /// <see cref="EagerStep"/> add it. False only for an objective that already contains that term.</param>
    /// <returns>The objective's value before the update.</returns>
    public static T EagerObjectiveStep(
        IReadOnlyList<Tensor<T>> parameters,
        Func<Tensor<T>> objective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null,
        double modelGradientClip = 0.0,
        Action? beforeUpdate = null,
        bool applyOptimizerRegularization = true)
    {
        Guard.NotNull(parameters);
        return EagerObjectiveStep(() => parameters, objective, optimizer, onGradients, modelGradientClip, beforeUpdate,
            applyOptimizerRegularization);
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
        Action? beforeUpdate = null,
        bool applyOptimizerRegularization = true)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(objective);
        Guard.NotNull(optimizer);
        var owner = ResolveOwnerFromDelegates(parameters, objective, onGradients, beforeUpdate);
        var placeholder = new Tensor<T>(new[] { 1 });
        return EagerTapeStep(
            objective,
            parameters,
            OptimizerUpdate(optimizer),
            placeholder,
            placeholder,
            (_, _) => objective(),
            (value, _) => value,
            new EagerStepHooks
            {
                Owner = owner,
                OnGradients = onGradients,
                RegularizeGradients = applyOptimizerRegularization ? OptimizerRegularization(optimizer) : null,
                ModelGradientClip = modelGradientClip,
                Optimizer = optimizer,
                BeforeUpdate = beforeUpdate,
            });
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
        => EagerSgdObjectiveStep(parameters, objective, learningRate, onGradients, owner: null);

    /// <summary>
    /// <see cref="EagerSgdObjectiveStep(Func{IReadOnlyList{Tensor{T}}}, Func{Tensor{T}}, T, Action{IReadOnlyDictionary{Tensor{T}, Tensor{T}}})"/>
    /// governed by <paramref name="owner"/>: a model with a distributed coordinator runs it as a coordinated step
    /// (the same collectives as its other ranks), and a mixed-precision context scales and unscales it.
    /// </summary>
    internal static T EagerSgdObjectiveStep(
        Func<IReadOnlyList<Tensor<T>>> parameters,
        Func<Tensor<T>> objective,
        T learningRate,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients,
        object? owner)
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
            new EagerStepHooks { OnGradients = onGradients, Owner = owner });
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
    /// <param name="applyOptimizerRegularization">Whether the optimizer's own configured regularization is added to the
    /// averaged gradients before the update (see <see cref="EagerObjectiveStep(IReadOnlyList{Tensor{T}}, Func{Tensor{T}}, IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}}, Action{IReadOnlyDictionary{Tensor{T}, Tensor{T}}}?, double, Action?, bool)"/>).</param>
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
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients = null,
        bool applyOptimizerRegularization = true)
    {
        Guard.NotNull(parameters);
        Guard.NotNull(microObjective);
        Guard.NotNull(optimizer);
        Guard.Positive(microBatchCount);

        var ownerNet = ResolveOwnerFromDelegates(microObjective, onGradients) as NeuralNetworks.NeuralNetworkBase<T>;
        if (ownerNet?.DistributedCoordinator is { } preCoord)
        {
            preCoord.EnsureInitialParametersSynchronized(parameters, ownerNet.InvalidateLayerWeightCaches);
        }

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
        if (ownerNet?.DistributedCoordinator is { } distCoord)
        {
            return ExecuteDistributedAccumulatedObjectiveStep(
                ownerNet,
                distCoord,
                parameters,
                microBatchCount,
                microObjective,
                optimizer,
                onGradients,
                applyOptimizerRegularization,
                accumulated,
                lossSum,
                inverseCount);
        }
        onGradients?.Invoke(accumulated);
        RegularizeAndClip(optimizer, reached, accumulated,
            applyOptimizerRegularization ? OptimizerRegularization(optimizer) : null, modelClip: null);

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

    /// <summary>
    /// The one place a training step orders the optimizer's regularization against the clips, between the backward
    /// and the optimizer's <c>Step</c>. <see cref="Models.Options.RegularizationOrder.RegularizeThenClip"/> adds the
    /// regularization, then the model's clip, then <c>Step</c> applies the optimizer's own.
    /// <see cref="Models.Options.RegularizationOrder.ClipThenRegularize"/> (PyTorch: <c>clip_grad_norm_</c>, then
    /// <c>step()</c> adds <c>weight_decay * theta</c>) runs the model's clip and the optimizer's clip first and adds the
    /// regularization last; that <c>Step</c> then skips its clip on this dictionary.
    /// </summary>
    /// <param name="optimizer">The optimizer about to step, whose options choose the order. An optimizer outside
    /// <c>GradientBasedOptimizerBase</c> cannot be clipped ahead and keeps regularize-then-clip.</param>
    /// <param name="parameters">The parameters being stepped.</param>
    /// <param name="gradients">Their gradients, updated in place (entries may be replaced).</param>
    /// <param name="regularize">Adds the optimizer's regularization; null when it has none.</param>
    /// <param name="modelClip">The model's own clip over its gradients (MaxGradNorm); null when it has none.</param>
    internal static void RegularizeAndClip(
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        IReadOnlyList<Tensor<T>> parameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? regularize,
        Action<Dictionary<Tensor<T>, Tensor<T>>>? modelClip)
    {
        if (regularize is not null
            && optimizer is Optimizers.GradientBasedOptimizerBase<T, Tensor<T>, Tensor<T>> ordered
            && ordered.RegularizesAfterClip)
        {
            modelClip?.Invoke(gradients);
            ordered.ClipTapeGradientsAhead(parameters, gradients);
            regularize(parameters, gradients);
            return;
        }
        regularize?.Invoke(parameters, gradients);
        modelClip?.Invoke(gradients);
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
        /// <summary>Optional model owner whose distributed coordinator or mixed-precision context governs the step.</summary>
        public object? Owner { get; init; }

        /// <summary>
        /// Widens the tensors the backward differentiates beyond the parameters (a diagnostic probe asking whether an
        /// input is reachable). Only the parameters are ever updated. Null differentiates exactly the parameters.
        /// </summary>
        public Func<IReadOnlyList<Tensor<T>>, IReadOnlyList<Tensor<T>>>? GradientSources { get; init; }

        /// <summary>Receives every gradient the backward produced, widened sources included, before any clip.</summary>
        public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnTapeGradients { get; init; }

        /// <summary>Receives the parameters' gradients before clipping (the model's public gradient surface).</summary>
        public Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? OnGradients { get; init; }

        /// <summary>
        /// Adds the optimizer's regularization to the reached parameters' gradients, on the side of the clips
        /// <see cref="Optimizer"/>'s options choose (see <see cref="RegularizeAndClip"/>).
        /// </summary>
        public Action<IReadOnlyList<Tensor<T>>, Dictionary<Tensor<T>, Tensor<T>>>? RegularizeGradients { get; init; }

        /// <summary>The model's own global-norm clip (0 = none), applied before the update.</summary>
        public double ModelGradientClip { get; init; }

        /// <summary>The optimizer the update steps, whose options order the regularization against the clips.</summary>
        public IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer { get; init; }

        /// <summary>Regularizes and clips <paramref name="gradients"/> as <see cref="RegularizeAndClip"/> orders it.</summary>
        internal void RegularizeAndClip(IReadOnlyList<Tensor<T>> parameters, Dictionary<Tensor<T>, Tensor<T>> gradients)
        {
            double modelClip = ModelGradientClip;
            TapeTrainingStepper<T>.RegularizeAndClip(Optimizer, parameters, gradients, RegularizeGradients,
                modelClip > 0.0 ? g => ClipByGlobalNorm(g, modelClip) : null);
        }

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
        EagerStepHooks? hooks = null,
        IReadOnlyList<Func<Tensor<T>>>? objectiveTerms = null)
    {
        Guard.NotNull(objective);
        Guard.NotNull(parameterProvider);
        Guard.NotNull(update);
        if (hooks?.Owner is NeuralNetworks.NeuralNetworkBase<T> { DistributedCoordinator: { } distCoord } distNet)
        {
            return ExecuteDistributedEagerTapeStep(
                distNet, distCoord, objective, parameterProvider, update,
                input, target, recomputeForward, recomputeLoss, hooks, objectiveTerms);
        }
        if (objectiveTerms is null
            && hooks?.Owner is NeuralNetworks.NeuralNetworkBase<T> { ActiveMixedPrecisionContext: { } mpCtx } mpNet)
        {
            return ExecuteMixedPrecisionEagerTapeStep(
                mpNet, mpCtx, objective, parameterProvider, update,
                input, target, recomputeForward, recomputeLoss, hooks);
        }
        if (objectiveTerms is not null)
            return SummedTapeObjective<T>.Step(objectiveTerms, parameterProvider, update,
                input, target, recomputeForward, recomputeLoss, hooks);
        using var tape = new GradientTape<T>();
        var loss = objective();
        if (loss is null)
            throw new InvalidOperationException("A training objective returned null instead of a scalar loss tensor.");
        var parameters = parameterProvider();
        if (parameters is null)
            throw new InvalidOperationException("A training step's parameter provider returned null.");

        var widenSources = hooks?.GradientSources;
        var sources = widenSources is null ? parameters : widenSources(parameters) ?? parameters;
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
        hooks?.RegularizeAndClip(reached, gradients);
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
            if (e is OutOfMemoryException or AiDotNet.Tensors.Engines.DirectGpu.GpuOutOfMemoryException)
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
    /// <remarks>
    /// The backends report most device faults as messages rather than typed exceptions, so this matches the
    /// driver-level markers, and first rules out the failures those markers also appear in that would fail again
    /// on the next step: a kernel that does not compile, an argument or shape the device rejects. Dropping a
    /// committed plan for one of those would hide a bug behind an optimizer switch.
    /// </remarks>
    internal static bool IsGpuTransientFailure(Exception? exception)
    {
        if (IsDeterministicDeviceFailure(exception))
            return false;

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

    // A failure that recurs on every attempt: the step's own arguments or a kernel build, whichever backend reports it.
    private static bool IsDeterministicDeviceFailure(Exception? exception)
    {
        for (var e = exception; e is not null; e = e.InnerException)
        {
            if (e is ArgumentException or IndexOutOfRangeException or NotSupportedException
                or AiDotNet.Tensors.Engines.DirectGpu.HIP.HipKernelCompilationException)
            {
                return true;
            }

            var message = e.Message;
            if (message.Contains("CL_BUILD_PROGRAM_FAILURE", StringComparison.Ordinal)
                || message.Contains("CL_COMPILE_PROGRAM_FAILURE", StringComparison.Ordinal)
                || message.Contains("CL_INVALID_", StringComparison.Ordinal)
                || message.Contains("CUDA_ERROR_INVALID_VALUE", StringComparison.Ordinal)
                || message.Contains("CUDA_ERROR_INVALID_PTX", StringComparison.Ordinal)
                || message.Contains("CUDA_ERROR_NO_BINARY_FOR_GPU", StringComparison.Ordinal)
                || message.Contains("compilation failed", StringComparison.OrdinalIgnoreCase))
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
    internal static void ClipByGlobalNorm(Dictionary<Tensor<T>, Tensor<T>> gradients, double maxNorm)
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
        // A NaN norm is left alone rather than spread to every gradient as a NaN scale.
        if (double.IsNaN(norm) || double.IsInfinity(norm) || norm <= maxNorm) return;
        T scale = NumOps.FromDouble(maxNorm / (norm + 1e-6));
        foreach (var gradient in gradients.Values)
            engine.TensorMultiplyScalarInPlace(gradient, scale);
    }

    private static object? ResolveOwnerFromDelegates(params Delegate?[] delegates)
    {
        for (int i = 0; i < delegates.Length; i++)
        {
            var targetObj = delegates[i]?.Target;
            if (targetObj is null) continue;
            if (targetObj is NeuralNetworks.NeuralNetworkBase<T>)
                return targetObj;

            var targetType = targetObj.GetType();
            if (targetType.Name.StartsWith("<", StringComparison.Ordinal))
            {
                var fields = targetType.GetFields(
                    System.Reflection.BindingFlags.Instance
                    | System.Reflection.BindingFlags.Public
                    | System.Reflection.BindingFlags.NonPublic);
                for (int f = 0; f < fields.Length; f++)
                {
                    if (typeof(NeuralNetworks.NeuralNetworkBase<T>).IsAssignableFrom(fields[f].FieldType)
                        && fields[f].GetValue(targetObj) is NeuralNetworks.NeuralNetworkBase<T> capturedNet)
                    {
                        return capturedNet;
                    }
                }
            }
        }
        return null;
    }

    private static T ExecuteDistributedAccumulatedObjectiveStep(
        NeuralNetworks.NeuralNetworkBase<T> ownerNet,
        DistributedTraining.DistributedTapeCoordinator<T> distCoord,
        IReadOnlyList<Tensor<T>> parameters,
        int microBatchCount,
        Func<int, Tensor<T>> microObjective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>>? onGradients,
        bool applyOptimizerRegularization,
        Dictionary<Tensor<T>, Tensor<T>> accumulated,
        double lossSum,
        T inverseCount)
    {
        var engine = AiDotNetEngine.Current;
        double localMeanLoss = lossSum / microBatchCount;
        var (rankWeight, globalLoss, _) = distCoord.SynchronizeStepMetadata(microBatchCount, localMeanLoss);
        T globalLossT = NumOps.FromDouble(globalLoss);
        ownerNet.SetLastLossFromStepper(globalLossT);
        distCoord.BeginBackwardStep(parameters, rankWeight);

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

        var regularizer = applyOptimizerRegularization
            ? (ownerNet.GetDistributedOptimizerRegularizer(optimizer) ?? OptimizerRegularization(optimizer))
            : null;
        var placeholder = new Tensor<T>(new[] { 1 });

        ownerNet.MarkDistributedMutationStarted();
        bool stepped = distCoord.SynchronizeUnscaleClipAndStep(
            parameters,
            accumulated,
            optimizer,
            ownerNet.ActiveMixedPrecisionContext,
            restoreFp32Masters: null,
            syncFp32MastersFromWorking: null,
            regularizer,
            ownerNet.MaxGradNormValue,
            ctx => OptimizerUpdate(optimizer)(new TapeStepContext<T>(
                ctx.Parameters,
                ctx.Gradients,
                globalLossT,
                placeholder,
                placeholder,
                (_, _) => MeanObjective(),
                (value, _) => value,
                parameterBuffer: null)),
            ownerNet.InvalidateLayerWeightCaches);

        onGradients?.Invoke(accumulated);
        if (stepped && distCoord.Strategy != AiDotNet.Enums.DistributedStrategy.DDP)
        {
            ownerNet.FinalizeDistributedStepPostUpdate(optimizer);
        }

        return globalLossT;
    }

    private static T ExecuteDistributedEagerTapeStep(
        NeuralNetworks.NeuralNetworkBase<T> distNet,
        DistributedTraining.DistributedTapeCoordinator<T> distCoord,
        Func<Tensor<T>> objective,
        Func<IReadOnlyList<Tensor<T>>> parameterProvider,
        Action<TapeStepContext<T>> update,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeForward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeLoss,
        EagerStepHooks hooks,
        IReadOnlyList<Func<Tensor<T>>>? objectiveTerms)
    {
        if (input.Length == 0 || target.Length == 0)
        {
            var idleParameters = parameterProvider()
                ?? throw new InvalidOperationException("A training step's parameter provider returned null.");
            distCoord.EnsureInitialParametersSynchronized(idleParameters, distNet.InvalidateLayerWeightCaches);
            var (_, idleGlobalLoss, _) = distCoord.SynchronizeStepMetadata(0L, 0.0);
            T idleLossT = NumOps.FromDouble(idleGlobalLoss);
            distNet.SetLastLossFromStepper(idleLossT);
            distCoord.BeginBackwardStep(idleParameters, 0.0);

            var emptyGrads = new Dictionary<Tensor<T>, Tensor<T>>(idleParameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);
            hooks.BeforeUpdate?.Invoke();
            distNet.MarkDistributedMutationStarted();
            bool idleStepped = distCoord.SynchronizeUnscaleClipAndStep(
                idleParameters,
                emptyGrads,
                hooks.Optimizer,
                distNet.ActiveMixedPrecisionContext,
                restoreFp32Masters: null,
                syncFp32MastersFromWorking: null,
                hooks.RegularizeGradients,
                hooks.ModelGradientClip,
                ctx => update(new TapeStepContext<T>(
                    ctx.Parameters,
                    ctx.Gradients,
                    idleLossT,
                    input,
                    target,
                    recomputeForward,
                    recomputeLoss,
                    parameterBuffer: null)),
                distNet.InvalidateLayerWeightCaches);

            hooks.OnGradients?.Invoke(emptyGrads);
            if (idleStepped && distCoord.Strategy != AiDotNet.Enums.DistributedStrategy.DDP)
            {
                distNet.FinalizeDistributedStepPostUpdate(hooks.Optimizer);
            }
            return idleLossT;
        }

        if (distNet.HasLazyUninitializedLayers())
        {
            bool wasTraining = distNet.IsTrainingMode;
            if (wasTraining) distNet.SetTrainingMode(false);
            try
            {
                using var noGrad = new NoGradScope<T>();
                recomputeForward(input, target);
            }
            catch
            {
                // Best-effort warmup; training objective below surfaces actionable failures.
            }
            finally
            {
                if (wasTraining) distNet.SetTrainingMode(true);
            }
        }

        var preParams = parameterProvider()
            ?? throw new InvalidOperationException("A training step's parameter provider returned null.");
        distCoord.EnsureInitialParametersSynchronized(preParams, distNet.InvalidateLayerWeightCaches);

        var mpCtx = distNet.ActiveMixedPrecisionContext;
        List<float[]>? mpSnapshots = null;
        List<Tensor<T>>? mpTargets = null;
        bool mpRestored = true;

        try
        {
            distNet.PrepareMixedPrecisionWorkingWeights(mpCtx, preParams, out mpSnapshots, out mpTargets, ref mpRestored);

            using var tape = new GradientTape<T>();
            var loss = objective();
            if (loss is null)
                throw new InvalidOperationException("A training objective returned null instead of a scalar loss tensor.");

            var parameters = parameterProvider()
                ?? throw new InvalidOperationException("A training step's parameter provider returned null.");

            long localTokens = DistributedTraining.DistributedTapeCoordinator<T>.CountValidElements(target);
            if (localTokens <= 0) localTokens = 1L;
            double localUnscaledLoss = loss.Length > 0 ? NumOps.ToDouble(loss.Data.Span[0]) : 0.0;
            var lossForBackward = distNet.ScaleLossForMixedPrecision(mpCtx, loss);

            var (rankWeight, globalLoss, _) = distCoord.SynchronizeStepMetadata(localTokens, localUnscaledLoss);
            T globalLossT = NumOps.FromDouble(globalLoss);
            distNet.SetLastLossFromStepper(globalLossT);
            distCoord.BeginBackwardStep(parameters, rankWeight);

            var widenSources = hooks.GradientSources;
            var gradients = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);

            if (widenSources is null
                && objectiveTerms is null
                && hooks.OnTapeGradients is null
                && distCoord is { WorldSize: > 1, OverlapCommunication: true }
                && !distNet.IsGradientCheckpointingEnabled)
            {
                tape.ComputeGradientsStreaming(
                    lossForBackward,
                    parameters,
                    (param, grad) =>
                    {
                        if (param is null || grad is null || grad.Length == 0 || gradients.ContainsKey(param))
                            return;
                        distCoord.OnParameterGradientReady(param, grad);
                        gradients[param] = distCoord.GetOrCreateGradientSlot(param);
                    });
            }
            else
            {
                var sources = widenSources is null ? parameters : widenSources(parameters) ?? parameters;
                var all = tape.ComputeGradients(lossForBackward, sources);
                hooks.OnTapeGradients?.Invoke(all);
                CopySelectedParameterGradients(parameters, all, gradients, out _);
            }

            hooks.BeforeUpdate?.Invoke();
            distNet.MarkDistributedMutationStarted();
            bool stepped = distCoord.SynchronizeUnscaleClipAndStep(
                parameters,
                gradients,
                hooks.Optimizer,
                mpCtx,
                () => NeuralNetworks.NeuralNetworkBase<T>.RestoreMixedPrecisionMasterWeights(mpSnapshots, mpTargets, ref mpRestored),
                syncFp32MastersFromWorking: null,
                hooks.RegularizeGradients,
                hooks.ModelGradientClip,
                ctx => update(new TapeStepContext<T>(
                    ctx.Parameters,
                    ctx.Gradients,
                    globalLossT,
                    input,
                    target,
                    recomputeForward,
                    recomputeLoss,
                    parameterBuffer: null)),
                distNet.InvalidateLayerWeightCaches);

            hooks.OnGradients?.Invoke(gradients);
            if (stepped && distCoord.Strategy != AiDotNet.Enums.DistributedStrategy.DDP)
            {
                distNet.FinalizeDistributedStepPostUpdate(hooks.Optimizer);
            }
            return globalLossT;
        }
        finally
        {
            NeuralNetworks.NeuralNetworkBase<T>.RestoreMixedPrecisionMasterWeights(mpSnapshots, mpTargets, ref mpRestored);
        }
    }

    private static T ExecuteMixedPrecisionEagerTapeStep(
        NeuralNetworks.NeuralNetworkBase<T> mpNet,
        MixedPrecision.MixedPrecisionContext mpCtx,
        Func<Tensor<T>> objective,
        Func<IReadOnlyList<Tensor<T>>> parameterProvider,
        Action<TapeStepContext<T>> update,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeForward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> recomputeLoss,
        EagerStepHooks hooks)
    {
        if (input.Length > 0 && mpNet.HasLazyUninitializedLayers())
        {
            bool wasTraining = mpNet.IsTrainingMode;
            if (wasTraining) mpNet.SetTrainingMode(false);
            try
            {
                using var noGrad = new NoGradScope<T>();
                recomputeForward(input, target);
            }
            catch
            {
                // Best-effort warmup.
            }
            finally
            {
                if (wasTraining) mpNet.SetTrainingMode(true);
            }
        }

        var preParams = parameterProvider()
            ?? throw new InvalidOperationException("A training step's parameter provider returned null.");
        List<float[]>? mpSnapshots = null;
        List<Tensor<T>>? mpTargets = null;
        bool mpRestored = true;

        try
        {
            mpNet.PrepareMixedPrecisionWorkingWeights(mpCtx, preParams, out mpSnapshots, out mpTargets, ref mpRestored);

            using var tape = new GradientTape<T>();
            var loss = objective();
            if (loss is null)
                throw new InvalidOperationException("A training objective returned null instead of a scalar loss tensor.");

            var parameters = parameterProvider()
                ?? throw new InvalidOperationException("A training step's parameter provider returned null.");
            var lossForBackward = mpNet.ScaleLossForMixedPrecision(mpCtx, loss);

            var widenSources = hooks.GradientSources;
            var sources = widenSources is null ? parameters : widenSources(parameters) ?? parameters;
            var all = tape.ComputeGradients(lossForBackward, sources);
            hooks.OnTapeGradients?.Invoke(all);

            var gradients = new Dictionary<Tensor<T>, Tensor<T>>(parameters.Count, TensorReferenceComparer<Tensor<T>>.Instance);
            CopySelectedParameterGradients(parameters, all, gradients, out var reached);

            T lossValue = loss.Length > 0 ? loss.Data.Span[0] : NumOps.Zero;
            bool hasOverflow = mpNet.UnscaleAndCheckMixedPrecisionGradients(mpCtx, reached, gradients);
            NeuralNetworks.NeuralNetworkBase<T>.RestoreMixedPrecisionMasterWeights(mpSnapshots, mpTargets, ref mpRestored);

            if (hasOverflow)
            {
                hooks.OnGradients?.Invoke(gradients);
                return lossValue;
            }

            hooks.OnGradients?.Invoke(gradients);
            hooks.RegularizeAndClip(reached, gradients);
            hooks.BeforeUpdate?.Invoke();

            update(new TapeStepContext<T>(
                reached, gradients, lossValue, input, target, recomputeForward, recomputeLoss, parameterBuffer: null));
            return lossValue;
        }
        finally
        {
            NeuralNetworks.NeuralNetworkBase<T>.RestoreMixedPrecisionMasterWeights(mpSnapshots, mpTargets, ref mpRestored);
        }
    }

    private static void CopySelectedParameterGradients(
        IReadOnlyList<Tensor<T>> parameters,
        IReadOnlyDictionary<Tensor<T>, Tensor<T>> allGradients,
        Dictionary<Tensor<T>, Tensor<T>> selectedGradients,
        out List<Tensor<T>> reached)
    {
        reached = new List<Tensor<T>>(parameters.Count);
        for (int i = 0; i < parameters.Count; i++)
        {
            var parameter = parameters[i];
            if (parameter is not null
                && !selectedGradients.ContainsKey(parameter)
                && allGradients.TryGetValue(parameter, out var gradient)
                && gradient is not null)
            {
                selectedGradients[parameter] = gradient;
                reached.Add(parameter);
            }
        }
    }
}
