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
    private static readonly ConditionalWeakTable<object, object> Warmed = new();

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
    /// <returns>The loss value for this step, measured before the update.</returns>
    public static T Step(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>>? loss = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Resolve lazy layer shapes BEFORE reading the registry. The convolutions behind the Conv2D
        // adapter infer their input depth on first Forward and own no parameters until then, so the
        // registry would report none and the step would silently do nothing. No tape is active here,
        // so this records nothing.
        if (!Warmed.TryGetValue(model, out _))
        {
            forward(input);
            Warmed.Add(model, model);
        }

        var parameters = LiveTrainableTensors(model);
        if (parameters.Length == 0)
        {
            return numOps.Zero;
        }

        var engine = AiDotNetEngine.Current;
        using (var tape = new GradientTape<T>())
        {
            var predicted = forward(input);
            var objective = (loss ?? MeanSquaredError)(predicted, target);
            var gradients = tape.ComputeGradients(objective, parameters);

            // The update runs INSIDE the tape's scope. Disposing the outermost tape rewinds the
            // active TensorArena (the per-step recycling of AiDotNet #1804), and the gradients and
            // the loss live in that arena: consumed after the dispose, their storage is already
            // being reissued to the update's own temporaries. Every model trained inside an arena
            // then applied a mix of its gradients and unrelated scratch - and threw only when a
            // reissued buffer happened to have a different shape (a [256, 1024] weight receiving
            // [1024, 256]). The no-grad scope keeps the update itself off the tape.
            using (new NoGradScope<T>())
            {
                foreach (var parameter in parameters)
                {
                    if (gradients.TryGetValue(parameter, out var gradient))
                    {
                        if (!SameShape(parameter, gradient))
                        {
                            throw new InvalidOperationException(
                                $"{model.GetType().Name}: the gradient for a trainable tensor of shape "
                                + $"[{string.Join(", ", parameter._shape)}] has shape [{string.Join(", ", gradient._shape)}]. "
                                + "The forward pass must use this tensor exactly as registered - a reshaped copy or "
                                + "a view created outside the engine records the wrong tensor on the tape.");
                        }

                        engine.TensorSubtractInPlace(parameter, engine.TensorMultiplyScalar(gradient, learningRate));
                    }
                }

                return objective.Length > 0 ? objective[0] : numOps.Zero;
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
    private static Tensor<T> MeanSquaredError(Tensor<T> predicted, Tensor<T> target)
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
