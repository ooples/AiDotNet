using System.Linq;
using System;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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
        double total = 0.0;
        int batch = BatchSize(predicted.Length, actual.Length, out int stride);
        actual = NormalizeTargets(actual, batch);

        for (int i = 0; i < batch; i++)
        {
            total += MbPAOutputNetwork<T>.Loss(
                _headParameters, Slice(predicted, i * stride, stride), Slice(actual, i * _outputDim, _outputDim),
                weight: 1.0, _featureDim, _outputDim, _distribution);
        }

        return Ops.FromDouble(batch > 0 ? total / batch : 0.0);
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
        var weights = new Tensor<T>([_featureDim, _outputDim]);
        var bias = new Tensor<T>([1, _outputDim]);
        for (int o = 0; o < _outputDim; o++)
        {
            for (int f = 0; f < _featureDim; f++) weights[f * _outputDim + o] = _headParameters[o * _featureDim + f];
            bias[o] = _headParameters[_featureDim * _outputDim + o];
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
        var derivative = new Vector<T>(predicted.Length);
        int batch = BatchSize(predicted.Length, actual.Length, out int stride);
        if (batch == 0) return derivative;
        actual = NormalizeTargets(actual, batch);

        // Averaged over the batch to match CalculateLoss. A loss that is a mean and a derivative that
        // is a sum differ by the batch size, which turns the effective learning rate into a function
        // of how the caller happened to chunk its data.
        double scale = 1.0 / batch;

        for (int i = 0; i < batch; i++)
        {
            var perExample = MbPAOutputNetwork<T>.InputGradient(
                _headParameters, Slice(predicted, i * stride, stride), Slice(actual, i * _outputDim, _outputDim),
                weight: scale, _featureDim, _outputDim, _distribution);

            // The head reads the first _featureDim entries of the embedding, so those are the only
            // entries that contributed to the loss and the only ones that receive gradient. Anything
            // beyond that width genuinely has a zero derivative -- it is not a dropped term.
            int copy = Math.Min(perExample.Length, stride);
            for (int f = 0; f < copy; f++)
            {
                derivative[(i * stride) + f] = perExample[f];
            }
        }

        return derivative;
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

    private Vector<T> NormalizeTargets(Vector<T> actual, int batch)
    {
        if (actual.Length == batch * _outputDim) return actual;
        var normalized = new Vector<T>(batch * _outputDim);
        for (int i = 0; i < batch; i++)
        {
            var row = MbPAConversions<T>.CategoricalTarget(Slice(actual, i, 1), _outputDim);
            for (int o = 0; o < _outputDim; o++) normalized[i * _outputDim + o] = row[o];
        }
        return normalized;
    }

    private static Vector<T> Slice(Vector<T> source, int offset, int length)
    {
        int available = Math.Max(0, Math.Min(length, source.Length - offset));
        var slice = new Vector<T>(length);
        for (int i = 0; i < available; i++) slice[i] = source[offset + i];
        return slice;
    }
}
