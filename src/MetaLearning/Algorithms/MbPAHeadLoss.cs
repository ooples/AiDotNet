using System.Linq;
using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// The MbPA output head, expressed as a loss function over the EMBEDDING network's output so that
/// f_gamma can be trained through the head rather than around it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>What this exists to fix.</b> MbPA's embedding network emits a key <c>h</c>, not a prediction;
/// the prediction is <c>g_theta(h)</c> from a linear head the algorithm owns and adapts by hand.
/// Meta-training used to differentiate the configured loss directly against <c>h</c> and the label
/// <c>y</c>, which is a category error: it trains f_gamma to output the label itself, and the head
/// never appears in the meta-objective at all. The head then drifts as a function of an embedding
/// that was optimized for a different job, and the paper's invariant -- that local adaptation moves
/// only <c>theta</c>, on top of an embedding trained for the composed loss -- does not hold.
/// </para>
/// <para>
/// Wrapping the head in an <see cref="ILossFunction{T}"/> puts it back where it belongs. The
/// embedding network differentiates <c>L(g_theta(h), y)</c>, and the chain rule through the head is
/// stated once here, in closed form, rather than approximated per call site.
/// </para>
/// <para>
/// <b>theta is a constant here.</b> The derivative returned is with respect to <c>h</c> only. That is
/// deliberate and is the invariant: the head is trained by the support-set loop in the algorithm and
/// adapted locally at prediction time, and neither of those paths runs through this object.
/// </para>
/// <para>
/// <b>On batching.</b> The loss is handed whatever the model flattens its batch into, with no shape
/// metadata, so each embedding has the configured feature width. Targets carry either one class
/// index per categorical example or one full output vector per example. Invalid widths are rejected.
/// </para>
/// </remarks>
internal sealed partial class MbPAHeadLoss<T> : ILossFunction<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _headParameters;
    private readonly int _featureDim;
    private readonly int _outputDim;
    private readonly MbPAOutputDistribution _distribution;

    /// <summary>
    /// Creates a loss that composes the given head with whatever produced the embedding.
    /// </summary>
    /// <param name="headParameters">
    /// theta, read but never written. The caller passes its live head vector; the head continues to be
    /// trained elsewhere and this object simply reads whatever it currently holds.
    /// </param>
    /// <param name="featureDim">The head's input width.</param>
    /// <param name="outputDim">The head's output width.</param>
    /// <param name="distribution">Which log-likelihood the head models.</param>
    internal MbPAHeadLoss(
        Vector<T> headParameters, int featureDim, int outputDim, MbPAOutputDistribution distribution)
    {
        _headParameters = headParameters ?? throw new ArgumentNullException(nameof(headParameters));
        if (featureDim <= 0) throw new ArgumentOutOfRangeException(nameof(featureDim));
        if (outputDim <= 0) throw new ArgumentOutOfRangeException(nameof(outputDim));
        if (headParameters.Length != (featureDim + 1) * outputDim)
            throw new ArgumentException("Head parameters must contain the output weights and biases.", nameof(headParameters));
        _featureDim = featureDim;
        _outputDim = outputDim;
        _distribution = distribution;
    }

    /// <inheritdoc/>
    public T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        int batch = BatchSize(predicted.Length, actual.Length, out _);
        if (batch == 0) return Ops.Zero;
        actual = NormalizeTargets(actual, batch);

        T total = MbPAOutputNetwork<T>.Loss(
            _headParameters, Rows(predicted, batch, _featureDim), Rows(actual, batch, _outputDim),
            weight: 1.0, _featureDim, _outputDim, _distribution);
        return Ops.Divide(total, Ops.FromDouble(batch));
    }

    /// <inheritdoc/>
    /// <summary>Tape-recorded loss, required by <see cref="ILossFunction{T}"/> since #1994.</summary>
    /// <remarks>
    /// Applies the fixed linear output head before categorical cross-entropy or Gaussian negative
    /// log-likelihood. Both objectives sum over output coordinates and average over examples,
    /// matching <see cref="CalculateLoss"/> and its embedding derivative.
    /// </remarks>
    public Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        var engine = AiDotNetEngine.Current;
        int batch = BatchSize(predicted.Length, target.Length, out _);
        if (batch == 0)
            return engine.ReduceSum(predicted, Enumerable.Range(0, predicted.Rank).ToArray(), keepDims: false);
        // The head is read as a fixed copy, off the tape, exactly as the fresh tensors it replaces were.
        Tensor<T> weights;
        Tensor<T> bias;
        using (new NoGradScope<T>())
        {
            var (headWeights, headBias) = MbPAOutputNetwork<T>.Split(_headParameters.Clone(), _featureDim, _outputDim);
            weights = engine.TensorTranspose(headWeights).Clone();
            bias = engine.Reshape(headBias, [1, _outputDim]).Clone();
        }
        var logits = engine.TensorAdd(engine.TensorMatMul(engine.Reshape(predicted, [batch, _featureDim]), weights), bias);
        var normalized = new Tensor<T>([batch, _outputDim], NormalizeTargets(target.ToVector(), batch));
        if (_distribution == MbPAOutputDistribution.Categorical)
            return new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>().ComputeTapeLoss(logits, normalized);
        var diff = engine.TensorSubtract(logits, normalized);
        var squared = engine.TensorMultiply(diff, diff);
        return engine.TensorMultiplyScalar(engine.ReduceSum(squared, [0, 1], keepDims: false),
            Ops.FromDouble(0.5 / batch));
    }

    public Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        int batch = BatchSize(predicted.Length, actual.Length, out _);
        if (batch == 0) return new Vector<T>(predicted.Length);
        actual = NormalizeTargets(actual, batch);

        // Averaged over the batch to match CalculateLoss. A loss that is a mean and a derivative that
        // is a sum differ by the batch size, which turns the effective learning rate into a function
        // of how the caller happened to chunk its data.
        double scale = 1.0 / batch;

        // Each embedding row is exactly _featureDim wide (BatchSize enforces it), so the [batch, featureDim]
        // input gradients are the derivative in the embedding's own layout.
        return MbPAOutputNetwork<T>.InputGradient(
            _headParameters, Rows(predicted, batch, _featureDim), Rows(actual, batch, _outputDim),
            weight: scale, _featureDim, _outputDim, _distribution).ToVector();
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <b>Routed through the CPU path rather than left unimplemented.</b> The composition this loss
    /// expresses is a small linear head over a handful of query examples per task -- there is nothing
    /// here a device kernel would win back, and the alternative shapes are worse: throwing would make
    /// the meta-learner's correctness depend on whether its model happens to take the GPU path, and
    /// returning a zero gradient would stop training without reporting anything.
    /// </remarks>
    public (T Loss, Tensor<T> Gradient) CalculateLossAndGradientGpu(Tensor<T> predicted, Tensor<T> actual)
    {
        var flatPredicted = predicted.ToVector();
        var flatActual = actual.ToVector();

        var loss = CalculateLoss(flatPredicted, flatActual);
        var gradient = Tensor<T>.FromVector(CalculateDerivative(flatPredicted, flatActual))
            .Reshape(predicted.Shape.ToArray());

        return (loss, gradient);
    }

    private int BatchSize(int predictedLength, int targetLength, out int stride)
    {
        stride = _featureDim;
        if (predictedLength % _featureDim != 0)
            throw new ArgumentException("Embedding values must contain complete rows of the configured feature width.", nameof(predictedLength));
        int batch = predictedLength / _featureDim;
        bool indices = _distribution == MbPAOutputDistribution.Categorical && targetLength == batch;
        if (!indices && targetLength != batch * _outputDim)
            throw new ArgumentException("Target count must match the embedding batch and output width.", nameof(targetLength));
        return batch;
    }

    /// <summary>
    /// One class index per example becomes a one-hot row per example. Indices must be integers in the
    /// head's class range, as <see cref="MbPAConversions{T}.CategoricalTarget"/> requires of each one.
    /// </summary>
    private Vector<T> NormalizeTargets(Vector<T> actual, int batch)
    {
        if (actual.Length == batch * _outputDim) return actual;
        var engine = AiDotNetEngine.Current;
        var indices = Tensor<T>.FromVector(actual);
        var integral = engine.TensorEquals(engine.TensorFloor(indices), indices);
        var nonNegative = engine.ScalarMinusTensor(Ops.One, engine.TensorLessThan(indices, Ops.Zero));
        var inRange = engine.TensorLessThan(indices, Ops.FromDouble(_outputDim));
        T valid = engine.TensorSum(engine.TensorMultiply(engine.TensorMultiply(integral, nonNegative), inRange));
        if (!Ops.Equals(valid, Ops.FromDouble(actual.Length)))
            throw new ArgumentOutOfRangeException(nameof(actual), "Class indices must be integers in the head's class range.");
        return engine.TensorOneHot<T>(indices.Cast<int>(), _outputDim).ToVector();
    }

    /// <summary>Views a flat vector of <paramref name="batch"/> rows as a [batch, width] tensor.</summary>
    private static Tensor<T> Rows(Vector<T> flat, int batch, int width)
        => Tensor<T>.FromVector(flat, new[] { batch, width });
}
