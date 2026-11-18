using System.Runtime.CompilerServices;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision;

/// <summary>
/// Marks two-stage region-proposal detectors whose forward and multi-stage losses run host-side NMS and
/// variable-cardinality RoI sampling per step and therefore step directly on the shared eager tape.
/// </summary>
internal interface IHostProposalVisionModel
{
}

/// <summary>
/// Tape-based training for the computer-vision models that are built on
/// <c>ModelBase&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c> rather than <c>NeuralNetworkBase</c> -
/// the object detectors, text detectors and OCR recognizers.
/// </summary>
/// <remarks>
/// <para>
/// The weights a step updates come from the model's parameter registry: every LIVE chunk with the
/// <see cref="ParameterSlotRole.Trainable"/> role, i.e. the exact tensor instances the forward pass
/// reads. That makes the registry the single source of truth for training, <c>GetParameters()</c>,
/// serialization and cloning. A weight the registry cannot see is therefore not silently skipped by
/// one surface and handled by another; it is missing from all of them, which is what the family
/// conformance audit checks for.
/// </para>
/// <para>
/// A chunk that is only a COPY of a weight (not writable in place) is excluded: the autodiff tape
/// keys gradients by tensor reference, so updating a copy would change nothing the model uses.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type the model is expressed in.</typeparam>
internal static class TensorModelTrainer<T>
{
    /// <summary>
    /// Models whose lazy layers have already resolved their shapes, so the warm-up forward is paid
    /// once per model rather than on every training step.
    /// </summary>
    private static readonly ConditionalWeakTable<object, WarmupState> Warmed = new();

    private sealed class WarmupState
    {
        public bool Complete;
    }

    /// <summary>
    /// Runs one tape-based training step: forward under a gradient tape, the loss (mean squared
    /// error unless the model supplies its own), then a stochastic-gradient update of every live
    /// trainable tensor.
    /// </summary>
    /// <param name="model">The model being trained.</param>
    /// <param name="input">The training input.</param>
    /// <param name="target">The desired output, shaped like the model prediction.</param>
    /// <param name="learningRate">Step size for the parameter update.</param>
    /// <param name="forward">
    /// The model's differentiable forward pass. It must be built from engine operations so the tape
    /// records it - a forward that drops to scalar loops severs the chain and the parameters upstream
    /// of the break receive no gradient.
    /// </param>
    /// <param name="loss">
    /// The training objective as <c>loss(predicted, target)</c>, a one-element tensor built from engine
    /// operations. Null means mean squared error. A model whose paper trains it with another objective
    /// (TrOCR: cross-entropy under teacher forcing) passes it here.
    /// </param>
    /// <param name="optimizer">The update rule. Null means plain SGD at <paramref name="learningRate"/>.</param>
    /// <returns>The loss value for this step, measured before the update.</returns>
    public static T Step(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>>? loss = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        var objective = loss ?? MeanSquaredError;
        return model is not IHostProposalVisionModel
            && optimizer is not null
            && TryFusedStep(model, input, target, forward, objective, optimizer, out T fusedLoss)
            ? fusedLoss
            : StepWithTargets(model, input, target, learningRate, forward, objective, optimizer);
    }

    /// <summary>Steps this process has run on the fused compiled plan, for tests that must see the plan engage.</summary>
    internal static int FusedStepCount;

    // One fused session per model: the compiled plan owns the optimizer moments across steps.
    private static readonly ConditionalWeakTable<object, AiDotNet.Training.FusedTrainingSession<T>> Sessions = new();

    /// <summary>
    /// Forward, loss, backward and the optimizer update as one compiled plan, on the shared session every base
    /// class trains through, when an explicit optimizer and a tensor target are supplied. Plain-SGD and typed-target
    /// objectives route through <see cref="StepWithTargets{TPrediction, TTarget}"/>, which delegates to the shared
    /// persistent closure session on <see cref="AiDotNet.Training.TapeTrainingStepper{T}"/> and preserves its
    /// continuity and committed-plan refusal checks.
    /// </summary>
    private static bool TryFusedStep(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> loss,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer,
        out T lossValue)
    {
        lossValue = MathHelper.GetNumericOperations<T>().Zero;
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation)
            return false;

        EnsureWarm(model, forward, input);
        var parameters = LiveTrainableTensors(model);
        if (parameters.Length == 0)
            return false;

        var session = Sessions.GetValue(model, static m => new AiDotNet.Training.FusedTrainingSession<T>(m));
        var outcome = session.TryStep(new AiDotNet.Training.FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Input = input,
            Target = target,
            Forward = forward,
            ComputeLoss = loss,
            Optimizer = optimizer,
            ExtraParameters = parameters,
            // A vision forward can read host-side state a trace would bake in; the first replay is checked against
            // the eager forward and a disagreeing update is undone.
            VerifyReplayAgreement = true,
        }, out lossValue);

        if (outcome == AiDotNet.Training.FusedStepOutcome.CommittedFailure)
            session.DropAfterCommittedFailure();
        if (outcome != AiDotNet.Training.FusedStepOutcome.Stepped)
            return false;
        System.Threading.Interlocked.Increment(ref FusedStepCount);
        return true;
    }

    /// <summary>
    /// Runs the same single update for structured heads and typed task targets, without flattening
    /// away their meaning. Forward outputs and the loss are consumed inside the tape/arena lifetime.
    /// </summary>
    public static T StepWithTargets<TPrediction, TTarget>(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        TTarget target,
        T learningRate,
        Func<Tensor<T>, TPrediction> forward,
        Func<TPrediction, TTarget, Tensor<T>> loss,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>> checkShapes = gradients => RequireMatchingShapes(model, gradients);
        Tensor<T> Objective() => loss(forward(input), target);
        bool eagerHostProposals = model is IHostProposalVisionModel;

        // When compilation is off (or the detector runs variable-cardinality host-side RPN NMS / RoI sampling),
        // run the single InferenceMode warmup first so lazy layers own their weights before reading the registry.
        if (!AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation || eagerHostProposals)
        {
            EnsureWarm(model, forward, input);
            var eagerParameters = RequireTrainableTensors(model);
            return ExecuteObjectiveStep(
                model,
                () => eagerParameters,
                Objective,
                learningRate,
                optimizer,
                checkShapes,
                allowCompiledClosure: !eagerHostProposals);
        }

        // When compilation is enabled, TapeTrainingStepper's closure path runs its own throwaway warmup invocation
        // before querying the parameter provider. Coordinate with WarmupState so concurrent callers serialize cold
        // initialization, failed warmups remain retryable, and a fresh model pays only one warmup forward total
        // while keeping owner: model routing (compiled closure execution + committed-state refusal checks) intact.
        // Only the model's first (cold) step runs under the lock; every later step takes the unlocked path below.
        // The warmup forward runs exactly once whichever side reaches it first: the stepper's closure warmup
        // (objective), or the parameter provider when that stepper was already warmed by an earlier caller.
        var warmup = Warmed.GetValue(model, static _ => new WarmupState());
        if (!System.Threading.Volatile.Read(ref warmup.Complete))
        {
            lock (warmup)
            {
                if (!warmup.Complete)
                {
                    Tensor<T> WarmAwareObjective()
                    {
                        if (!System.Threading.Volatile.Read(ref warmup.Complete))
                        {
                            using var noBackward = AiDotNet.NeuralNetworks.Layers.InferenceMode.Enter();
                            var warmupLoss = loss(forward(input), target);
                            System.Threading.Volatile.Write(ref warmup.Complete, true);
                            return warmupLoss;
                        }

                        return loss(forward(input), target);
                    }

                    Tensor<T>[] ResolveParametersAfterWarmup()
                    {
                        if (!System.Threading.Volatile.Read(ref warmup.Complete))
                        {
                            using var noBackward = AiDotNet.NeuralNetworks.Layers.InferenceMode.Enter();
                            forward(input);
                            System.Threading.Volatile.Write(ref warmup.Complete, true);
                        }

                        return RequireTrainableTensors(model);
                    }

                    return ExecuteObjectiveStep(
                        model,
                        ResolveParametersAfterWarmup,
                        WarmAwareObjective,
                        learningRate,
                        optimizer,
                        checkShapes);
                }
            }
        }

        var parameters = RequireTrainableTensors(model);
        return ExecuteObjectiveStep(model, () => parameters, Objective, learningRate, optimizer, checkShapes);
    }

    private static T ExecuteObjectiveStep(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Func<IReadOnlyList<Tensor<T>>> parameters,
        Func<Tensor<T>> objective,
        T learningRate,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>> checkShapes,
        bool allowCompiledClosure = true)
    {
        if (optimizer is null)
        {
            return AiDotNet.Training.TapeTrainingStepper<T>.EagerSgdObjectiveStep(
                parameters,
                objective,
                learningRate,
                checkShapes,
                owner: allowCompiledClosure ? model : null);
        }

        if (allowCompiledClosure)
        {
            return AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(
                parameters,
                objective,
                optimizer,
                checkShapes);
        }

        var placeholder = new Tensor<T>(new[] { 1 });
        return AiDotNet.Training.TapeTrainingStepper<T>.EagerStep(
            new AiDotNet.Training.FusedTrainingStepRequest<T>
            {
                Layers = Array.Empty<ITrainableLayer<T>>(),
                Input = placeholder,
                Target = placeholder,
                Forward = _ => objective(),
                ComputeLoss = static (loss, _) => loss,
                Optimizer = optimizer,
                ExtraParameters = parameters(),
                OnGradients = checkShapes,
            },
            model);
    }

    private static Tensor<T>[] RequireTrainableTensors(ModelBase<T, Tensor<T>, Tensor<T>> model)
    {
        var parameters = LiveTrainableTensors(model);
        if (parameters.Length == 0)
        {
            throw new InvalidOperationException(
                $"No live trainable tensors were discovered for model '{model.GetType().FullName}'.");
        }

        return parameters;
    }

    // A gradient whose shape differs from its tensor means the forward used a reshaped copy or an
    // out-of-engine view instead of the registered tensor, so the tape recorded the wrong one.
    private static void RequireMatchingShapes(
        ModelBase<T, Tensor<T>, Tensor<T>> model, IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        foreach (var pair in gradients)
        {
            if (!SameShape(pair.Key, pair.Value))
            {
                throw new InvalidOperationException(
                    $"{model.GetType().Name}: the gradient for a trainable tensor of shape "
                    + $"[{string.Join(", ", pair.Key._shape)}] has shape [{string.Join(", ", pair.Value._shape)}]. "
                    + "The forward pass must use this tensor exactly as registered - a reshaped copy or "
                    + "a view created outside the engine records the wrong tensor on the tape.");
            }
        }
    }
    // Runs one forward the first time a model trains, so lazily shaped layers own their parameters before the
    // registry is read.
    private static void EnsureWarm<TPrediction>(
        ModelBase<T, Tensor<T>, Tensor<T>> model, Func<Tensor<T>, TPrediction> forward, Tensor<T> input)
    {
        var warmup = Warmed.GetValue(model, static _ => new WarmupState());
        if (System.Threading.Volatile.Read(ref warmup.Complete))
            return;

        // GetValue may invoke competing factories; its returned state is the one shared by every caller. Serialize
        // initialization, not the whole training step. A failed forward leaves Complete false so a later call
        // retries instead of caching the exception.
        lock (warmup)
        {
            if (!warmup.Complete)
            {
                // No backward follows this forward. In training mode with no tape, activation layers would push their
                // pre-activations for a manual backward that never pops them, and keep them for the model's life
                // (100 MB on FasterRCNN at 320x320). The scope skips only those caches.
                using var noBackward = AiDotNet.NeuralNetworks.Layers.InferenceMode.Enter();
                forward(input);
                System.Threading.Volatile.Write(ref warmup.Complete, true);
            }
        }
    }
    private static bool SameShape(Tensor<T> a, Tensor<T> b)
    {
        if (a._shape.Length != b._shape.Length)
        {
            return false;
        }

        for (int i = 0; i < a._shape.Length; i++)
        {
            if (a._shape[i] != b._shape[i])
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The distinct live tensors the registry marks trainable, in registry order.
    /// </summary>
    public static Tensor<T>[] LiveTrainableTensors(ModelBase<T, Tensor<T>, Tensor<T>> model)
    {
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer.Instance);
        var result = new List<Tensor<T>>();
        foreach (var chunk in model.GetParameterStateChunks())
        {
            if (chunk.Role == ParameterSlotRole.Trainable && chunk.IsWritableInPlace && seen.Add(chunk.Tensor))
            {
                result.Add(chunk.Tensor);
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// Mean squared error built from engine operations so the gradient tape can differentiate it.
    /// </summary>
    internal static Tensor<T> MeanSquaredError(Tensor<T> predicted, Tensor<T> target)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();

        var difference = engine.TensorSubtract(predicted, target);
        var squared = engine.TensorMultiply(difference, difference);
        return engine.TensorMultiplyScalar(
            engine.ReduceSum(squared, null),
            numOps.FromDouble(1.0 / Math.Max(1, squared.Length)));
    }

    private sealed class TensorReferenceComparer : IEqualityComparer<Tensor<T>>
    {
        public static readonly TensorReferenceComparer Instance = new();

        public bool Equals(Tensor<T>? x, Tensor<T>? y) => (object?)x == y;   // reference comparison

        public int GetHashCode(Tensor<T> obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
