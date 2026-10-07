using System.Runtime.CompilerServices;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.ComputerVision;

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
    /// Runs one training step on raw tensors: the forward, the loss (mean squared error unless the model supplies its
    /// own), then an update of every live trainable tensor.
    /// </summary>
    /// <param name="model">The model being trained.</param>
    /// <param name="input">The training input.</param>
    /// <param name="target">The desired output, shaped like the model prediction.</param>
    /// <param name="learningRate">Step size for the plain-SGD update used when <paramref name="optimizer"/> is null.</param>
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
    /// <remarks>
    /// With an optimizer the step goes through the model's <see cref="AiDotNet.Training.TapeTrainingStepper{T}"/>: the
    /// fused compiled plan (forward, backward and update in one replay, on any engine) when it applies, the shared
    /// eager tape step otherwise. These models' forwards were written for the eager tape, so the plan's first replay
    /// on new data is checked against the eager forward and the model stays eager when they disagree.
    /// </remarks>
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
        var parameters = WarmedTrainableTensors(model, input, forward);
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>> checkShapes = gradients => RequireMatchingShapes(model, gradients);
        if (optimizer is null)
        {
            return AiDotNet.Training.TapeTrainingStepper<T>.EagerSgdObjectiveStep(
                () => parameters, () => objective(forward(input), target), learningRate, checkShapes);
        }

        return AiDotNet.Training.TapeTrainingStepper<T>.ForOwner(model).Step(new AiDotNet.Training.FusedTrainingStepRequest<T>
        {
            Layers = Array.Empty<ITrainableLayer<T>>(),
            Selection = parameters,
            ExtraParameters = parameters,
            Input = input,
            Target = target,
            Forward = forward,
            ComputeLoss = objective,
            Optimizer = optimizer,
            OnGradients = checkShapes,
            VerifyReplayAgreement = true,
        });
    }

    /// <summary>
    /// Runs the same single update for structured heads and typed task targets, without flattening
    /// away their meaning. Forward outputs and the loss are consumed inside the tape/arena lifetime.
    /// </summary>
    /// <remarks>
    /// A typed task loss assigns targets to predictions on the host (anchor matching, bipartite matching), so this
    /// step is a graph break: it always runs on the shared eager tape step.
    /// </remarks>
    public static T StepWithTargets<TPrediction, TTarget>(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        TTarget target,
        T learningRate,
        Func<Tensor<T>, TPrediction> forward,
        Func<TPrediction, TTarget, Tensor<T>> loss,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        var parameters = WarmedTrainableTensors(model, input, forward);
        Action<IReadOnlyDictionary<Tensor<T>, Tensor<T>>> checkShapes = gradients => RequireMatchingShapes(model, gradients);
        Tensor<T> Objective() => loss(forward(input), target);
        return optimizer is null
            ? AiDotNet.Training.TapeTrainingStepper<T>.EagerSgdObjectiveStep(() => parameters, Objective, learningRate, checkShapes)
            : AiDotNet.Training.TapeTrainingStepper<T>.EagerObjectiveStep(parameters, Objective, optimizer, checkShapes);
    }

    // Resolves lazy layer shapes BEFORE reading the registry, then returns the live trainable tensors. The
    // convolutions behind the Conv2D adapter infer their input depth on first Forward and own no parameters until
    // then, so the registry would report none and the step would silently do nothing. No tape is active here, so
    // this records nothing.
    private static Tensor<T>[] WarmedTrainableTensors<TPrediction>(
        ModelBase<T, Tensor<T>, Tensor<T>> model, Tensor<T> input, Func<Tensor<T>, TPrediction> forward)
    {
        var warmup = Warmed.GetValue(model, static _ => new WarmupState());
        if (!System.Threading.Volatile.Read(ref warmup.Complete))
        {
            // GetValue may invoke competing factories; its returned state is the one shared by
            // every caller. Serialize initialization, not the whole training step. A failed forward
            // leaves Complete false so a later call retries instead of caching the exception.
            lock (warmup)
            {
                if (!warmup.Complete)
                {
                    forward(input);
                    System.Threading.Volatile.Write(ref warmup.Complete, true);
                }
            }
        }

        var parameters = LiveTrainableTensors(model);
        if (parameters.Length == 0)
        {
            throw new InvalidOperationException(
                $"No live trainable tensors were discovered for model '{model.GetType().FullName}'.");
        }

        return parameters;
    }

    private static void RequireMatchingShapes(
        ModelBase<T, Tensor<T>, Tensor<T>> model, IReadOnlyDictionary<Tensor<T>, Tensor<T>> gradients)
    {
        foreach (var pair in gradients)
        {
            var parameter = pair.Key;
            var gradient = pair.Value;
            if (!SameShape(parameter, gradient))
            {
                throw new InvalidOperationException(
                    $"{model.GetType().Name}: the gradient for a trainable tensor of shape "
                    + $"[{string.Join(", ", parameter._shape)}] has shape [{string.Join(", ", gradient._shape)}]. "
                    + "The forward pass must use this tensor exactly as registered - a reshaped copy or "
                    + "a view created outside the engine records the wrong tensor on the tape.");
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

        public bool Equals(Tensor<T>? x, Tensor<T>? y) => ReferenceEquals(x, y);

        public int GetHashCode(Tensor<T> obj) => RuntimeHelpers.GetHashCode(obj);
    }
}
