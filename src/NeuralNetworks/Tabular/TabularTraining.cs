using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Training;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// The shared training step of the tabular models: reverse-mode gradients of a scalar objective over the model's
/// live parameter tensors on the autodiff tape, then one plain SGD update <c>p -= learningRate * grad</c>.
/// </summary>
/// <remarks>
/// The tabular <c>TrainStep</c> methods used to run a forward pass, compute the loss and call
/// <c>UpdateParameters</c> with no backward pass in between, so no weight ever moved. Every step now records the
/// forward and the loss as engine operations and differentiates them through the shared tape stepper.
/// </remarks>
internal static class TabularTraining
{
    /// <summary>
    /// Runs one tape step: records <paramref name="objective"/>, differentiates it with respect to every tensor
    /// <paramref name="parameters"/> returns (read after the objective runs, so lazily shaped layers are included)
    /// and applies plain SGD at <paramref name="learningRate"/>.
    /// </summary>
    /// <returns>The objective value before the update.</returns>
    public static T SgdStep<T>(Func<IEnumerable<Tensor<T>>> parameters, Func<Tensor<T>> objective, T learningRate)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (objective is null) throw new ArgumentNullException(nameof(objective));

        List<Tensor<T>>? stepped = null;
        IReadOnlyList<Tensor<T>> Collect()
        {
            stepped = parameters()
                .Where(tensor => tensor is not null && tensor.Length > 0)
                .Distinct(TensorReferenceComparer<Tensor<T>>.Instance)
                .ToList();
            return stepped;
        }

        try
        {
            return TapeTrainingStepper<T>.EagerSgdObjectiveStep(Collect, objective, learningRate);
        }
        finally
        {
            if (stepped is not null)
            {
                var engine = AiDotNetEngine.Current;
                foreach (var parameter in stepped)
                    engine.InvalidatePersistentTensor(parameter);
            }
        }
    }

    /// <summary>
    /// The live trainable tensors of a set of layers (recursing into their sub-layers).
    /// </summary>
    public static IReadOnlyList<Tensor<T>> LayerTensors<T>(IEnumerable<ILayer<T>> layers)
        => TapeTrainingStep<T>.CollectParameters(layers, -1);

    /// <summary>
    /// The first <paramref name="batchSize"/> class targets as a one-hot [batch, classes] tensor.
    /// </summary>
    /// <exception cref="ArgumentException">There are fewer targets than samples, or the batch is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A target lies outside [0, classes).</exception>
    public static Tensor<T> OneHotTargets<T>(Vector<int> targets, int batchSize, int numClasses)
    {
        if (targets is null) throw new ArgumentNullException(nameof(targets));
        if (batchSize <= 0 || targets.Length < batchSize)
            throw new ArgumentException("Targets must cover a nonempty input batch.", nameof(targets));

        var engine = AiDotNetEngine.Current;
        var classes = Tensor<int>.FromVector(targets.GetSubVector(0, batchSize));
        if (engine.TensorMinValue(classes) < 0 || engine.TensorMaxValue(classes) >= numClasses)
            throw new ArgumentOutOfRangeException(nameof(targets), "Target class is out of range.");

        return engine.TensorOneHot<T>(classes, numClasses).Reshape(batchSize, numClasses);
    }

    /// <summary>
    /// Mean cross-entropy <c>-(1/B) sum_b sum_c y[b, c] log(p[b, c] + epsilon)</c> of class probabilities, on the tape.
    /// </summary>
    public static Tensor<T> CrossEntropyObjective<T>(Tensor<T> probabilities, Tensor<T> oneHot, T epsilon)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int batchSize = oneHot.Shape[0];
        // Entries with a zero label contribute 0, so their log argument is taken as 1: a probability that
        // underflowed to 0 (with an epsilon that rounds to 0 in Half) would otherwise give 0 * log(0) = NaN in
        // the loss and its gradient.
        var shaped = probabilities.Reshape(oneHot._shape);
        var logArgument = engine.TensorWhere(
            engine.TensorNotEquals(oneHot, ops.Zero),
            TensorTapeOps.TapeAddScalar(engine, shaped, epsilon),
            engine.TensorOnesLike(shaped));
        var total = engine.ReduceSum(
            engine.TensorMultiply(oneHot, engine.TensorLog(logArgument)), new[] { 0, 1 }, false);
        return TensorTapeOps.TapeMultiplyScalar(engine, total, ops.FromDouble(-1.0 / batchSize));
    }

    /// <summary>
    /// Mean squared error over every element, on the tape.
    /// </summary>
    public static Tensor<T> MeanSquaredErrorObjective<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (predictions.Length != targets.Length)
            throw new ArgumentException("Targets must match the predictions element for element.", nameof(targets));

        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int count = predictions.Length;
        var diff = engine.TensorSubtract(predictions.Reshape(count), targets.Reshape(count));
        var total = engine.ReduceSum(engine.TensorMultiply(diff, diff), new[] { 0 }, false);
        return TensorTapeOps.TapeMultiplyScalar(engine, total, ops.FromDouble(1.0 / count));
    }
}
