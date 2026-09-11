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
    /// Runs one tape-based training step: forward under a gradient tape, mean-squared-error loss,
    /// then a stochastic-gradient update of every live trainable tensor.
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
    /// <returns>The loss value for this step.</returns>
    public static T Step(
        ModelBase<T, Tensor<T>, Tensor<T>> model,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward)
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
        Tensor<T> loss;
        Dictionary<Tensor<T>, Tensor<T>> gradients;
        using (var tape = new GradientTape<T>())
        {
            var predicted = forward(input);
            loss = MeanSquaredError(predicted, target);
            gradients = tape.ComputeGradients(loss, parameters);
        }

        foreach (var parameter in parameters)
        {
            if (gradients.TryGetValue(parameter, out var gradient))
            {
                engine.TensorSubtractInPlace(parameter, engine.TensorMultiplyScalar(gradient, learningRate));
            }
        }

        return loss.Length > 0 ? loss[0] : numOps.Zero;
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
