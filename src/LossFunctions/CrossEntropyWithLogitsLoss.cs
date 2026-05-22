using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements Cross-Entropy loss that accepts raw logits (not probabilities).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This loss function is equivalent to PyTorch's
/// <c>nn.CrossEntropyLoss</c>. It combines LogSoftmax and Negative Log-Likelihood
/// in a single numerically stable computation.
///
/// Unlike <see cref="CrossEntropyLoss{T}"/> which expects probability inputs
/// (after softmax), this version accepts raw logits (unbounded model outputs)
/// and applies the softmax internally. This is the correct choice when your
/// model's final layer outputs raw scores without softmax activation.
///
/// The formula uses the log-sum-exp trick for numerical stability:
///   loss = -logit[target_class] + log(sum(exp(logit_i)))
///
/// For soft targets (one-hot encoded):
///   loss = -sum(target_i * (logit_i - log(sum(exp(logit_j)))))
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.MultiClass)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, RequiresProbabilityInputs = false, TestInputFormat = LossTestInputFormat.RawLogits, ExpectedOutput = OutputType.Logits)]
public class CrossEntropyWithLogitsLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Calculates the Cross-Entropy loss from raw logits using log-sum-exp for stability.
    /// </summary>
    /// <param name="predicted">Raw logits (unbounded model outputs, NOT probabilities).</param>
    /// <param name="actual">One-hot encoded target vector or soft target distribution.</param>
    /// <returns>The cross-entropy loss value.</returns>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // LogSoftmax using log-sum-exp trick for numerical stability:
        // log_softmax_i = logit_i - log(sum(exp(logit_j)))
        // First find max for numerical stability
        T maxLogit = predicted[0];
        for (int i = 1; i < predicted.Length; i++)
        {
            if (NumOps.GreaterThan(predicted[i], maxLogit))
                maxLogit = predicted[i];
        }

        // Compute log(sum(exp(logit_i - max))) + max
        T sumExp = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sumExp = NumOps.Add(sumExp, NumOps.Exp(NumOps.Subtract(predicted[i], maxLogit)));
        }
        T logSumExp = NumOps.Add(NumericalStabilityHelper.SafeLog(sumExp), maxLogit);

        // CE = -sum(actual_i * log_softmax_i) = -sum(actual_i * (logit_i - logSumExp))
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T logSoftmax = NumOps.Subtract(predicted[i], logSumExp);
            loss = NumOps.Add(loss, NumOps.Multiply(actual[i], logSoftmax));
        }

        return NumOps.Negate(loss);
    }

    /// <summary>
    /// Calculates the derivative of cross-entropy loss with respect to logits.
    /// </summary>
    /// <param name="predicted">Raw logits.</param>
    /// <param name="actual">One-hot encoded target vector.</param>
    /// <returns>Gradient: softmax(logits) - targets.</returns>
    /// <remarks>
    /// <para>
    /// The gradient of cross-entropy loss with respect to logits has the elegant form:
    ///   d(loss)/d(logit_i) = softmax(logit_i) - target_i
    ///
    /// This is one of the key advantages of combining softmax and cross-entropy:
    /// the gradient is simply the difference between the predicted probabilities
    /// and the targets, which is both easy to compute and numerically stable.
    /// </para>
    /// </remarks>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        // Compute softmax probabilities
        T maxLogit = predicted[0];
        for (int i = 1; i < predicted.Length; i++)
        {
            if (NumOps.GreaterThan(predicted[i], maxLogit))
                maxLogit = predicted[i];
        }

        var expValues = new Vector<T>(predicted.Length);
        T sumExp = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            expValues[i] = NumOps.Exp(NumOps.Subtract(predicted[i], maxLogit));
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Gradient = softmax(logits) - targets
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T softmaxI = NumOps.Divide(expValues[i], sumExp);
            derivative[i] = NumOps.Subtract(softmaxI, actual[i]);
        }

        return derivative;
    }

    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // LogSoftmax + NLLLoss via Engine ops for tape tracking.
        // log_softmax = logits - log(sum(exp(logits))) along last axis.
        int lastAxis = predicted.Shape.Length - 1;
        var logSoftmax = Engine.TensorLogSoftmax(predicted, axis: lastAxis);

        // PyTorch's nn.CrossEntropyLoss accepts BOTH target formats and the
        // PR #1404 blanket CE→CEWithLogits swap brought models that pass
        // EITHER form into this code path:
        //   (a) soft / one-hot targets: target.Shape == predicted.Shape
        //       (e.g. predicted=[N,C], target=[N,C] one-hot or distribution)
        //   (b) class-index targets:    target.Shape == predicted.Shape[:-1]
        //       (e.g. predicted=[N,C], target=[N] integer class indices —
        //       this is what TinyBERTNER and similar NER / classification
        //       heads emit when they treat their gold labels as ID arrays
        //       rather than one-hot encodings).
        // The original implementation only handled (a). For (b) the broadcast
        // multiply on the next line throws ArgumentException because target's
        // shape doesn't broadcast onto logSoftmax's shape. Detect form (b)
        // by rank comparison and one-hot encode target along the class axis
        // before the multiply. The one-hot conversion is a non-tape op —
        // target is supervision, no gradient flows through it — so building
        // a fresh tensor here doesn't break gradient flow back through
        // predicted → logSoftmax → product.
        if (target.Shape.Length == predicted.Shape.Length - 1)
        {
            target = ClassIndicesToOneHot(target, predicted.Shape[lastAxis]);
        }

        // CE per-sample = -Σ_class target_i * log_softmax_i  (sum over class axis only).
        // Match PyTorch's nn.CrossEntropyLoss(reduction='mean') and the scalar CalculateLoss
        // path above — those return the per-sample sum across classes, NOT a mean across
        // classes. Averaging over the class axis here would silently divide gradients by the
        // class count, making tape training disagree with the CPU code path (that's the
        // exact bug noted in PR review).
        var product = Engine.TensorMultiply(target, logSoftmax);
        var perSample = Engine.ReduceSum(product, new[] { lastAxis }, keepDims: false);

        // Mean over remaining (batch/sample) axes if any. For 1D inputs (logits was a single
        // sample with no batch axis), perSample is rank-0 and there's nothing to average.
        if (perSample.Shape.Length == 0)
            return Engine.TensorNegate(perSample);

        var batchAxes = Enumerable.Range(0, perSample.Shape.Length).ToArray();
        var mean = Engine.ReduceMean(perSample, batchAxes, keepDims: false);
        return Engine.TensorNegate(mean);
    }

    /// <summary>
    /// Converts a class-index tensor of shape <c>S</c> into a one-hot tensor
    /// of shape <c>[S ; numClasses]</c> by appending a class axis of length
    /// <paramref name="numClasses"/> and setting one entry per slot to
    /// <c>1</c>. Out-of-range indices (negative or ≥ numClasses) are
    /// silently dropped (their one-hot slot stays zero) so an ignore-index
    /// sentinel like -1 contributes zero gradient — same convention as
    /// PyTorch's <c>ignore_index</c> default. The result is NOT registered
    /// with the autodiff tape; the loss treats target as a constant
    /// supervision signal, so a fresh tensor here doesn't disturb gradient
    /// flow through the logits side of the multiply.
    /// </summary>
    private Tensor<T> ClassIndicesToOneHot(Tensor<T> indices, int numClasses)
    {
        var indicesShape = indices.Shape;
        var oneHotShape = new int[indicesShape.Length + 1];
        for (int i = 0; i < indicesShape.Length; i++) oneHotShape[i] = indicesShape[i];
        oneHotShape[indicesShape.Length] = numClasses;

        var oneHot = new Tensor<T>(oneHotShape);
        var indicesSpan = indices.Data.Span;
        var oneHotSpan = oneHot.Data.Span;
        // oneHot is initialized to zero on construction; we only need to
        // set the active class index per row.
        for (int i = 0; i < indicesSpan.Length; i++)
        {
            int classIdx = (int)Math.Round(NumOps.ToDouble(indicesSpan[i]));
            if (classIdx >= 0 && classIdx < numClasses)
            {
                oneHotSpan[i * numClasses + classIdx] = NumOps.One;
            }
        }
        return oneHot;
    }
}
