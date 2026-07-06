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
        // log_softmax = logits - log(sum(exp(logits))) along the class axis.
        // For sequence/classification logits this is usually the final axis
        // ([B, S, V] + [B, S]). For dense segmentation, PyTorch's standard
        // layout is channel-first ([B, C, H, W] + [B, H, W]), so infer the
        // class axis from the compact class-index target shape when present.
        int classAxis = ResolveClassAxis(predicted, target);
        var logSoftmax = ComputeLogSoftmax(predicted, classAxis);

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
            target = ClassIndicesToOneHot(target, predicted.Shape[classAxis], classAxis, predicted.Shape.ToArray());
        }

        // CE per-sample = -Σ_class target_i * log_softmax_i  (sum over class axis only).
        // Match PyTorch's nn.CrossEntropyLoss(reduction='mean') and the scalar CalculateLoss
        // path above — those return the per-sample sum across classes, NOT a mean across
        // classes. Averaging over the class axis here would silently divide gradients by the
        // class count, making tape training disagree with the CPU code path (that's the
        // exact bug noted in PR review).
        var product = Engine.TensorMultiply(target, logSoftmax);
        var perSample = Engine.ReduceSum(product, new[] { classAxis }, keepDims: false);

        // Mean over remaining (batch/sample) axes if any. For 1D inputs (logits was a single
        // sample with no batch axis), perSample is rank-0 and there's nothing to average.
        if (perSample.Shape.Length == 0)
            return Engine.TensorNegate(perSample);

        var batchAxes = Enumerable.Range(0, perSample.Shape.Length).ToArray();
        var mean = Engine.ReduceMean(perSample, batchAxes, keepDims: false);
        return Engine.TensorNegate(mean);
    }

    private Tensor<T> ComputeLogSoftmax(Tensor<T> logits, int classAxis)
    {
        int rank = logits.Shape.Length;
        int normalizedAxis = classAxis < 0 ? rank + classAxis : classAxis;
        if (normalizedAxis < 0 || normalizedAxis >= rank)
            throw new ArgumentOutOfRangeException(nameof(classAxis), $"Class axis {classAxis} is outside logits rank {rank}.");

        // TensorLogSoftmax currently has a correct forward value, but some
        // tape paths do not propagate the full softmax(target)-one_hot
        // gradient across every class channel. Use primitive reductions and
        // broadcasts for the gradient, then add a detached correction so the
        // forward value remains the stable engine LogSoftmax result.
        var stableForward = Engine.TensorLogSoftmax(logits, axis: normalizedAxis);
        var maxLogit = Engine.ReduceMax(logits, new[] { normalizedAxis }, keepDims: true, out _);
        var shifted = Engine.TensorBroadcastAdd(logits, Engine.TensorNegate(maxLogit));
        var expShifted = Engine.TensorExp(shifted);
        var sumExp = Engine.ReduceSum(expShifted, new[] { normalizedAxis }, keepDims: true);
        var logSumExp = Engine.TensorLog(sumExp);
        var primitiveGradient = Engine.TensorBroadcastAdd(shifted, Engine.TensorNegate(logSumExp));
        var detachedForwardCorrection = Engine.StopGradient(Engine.TensorSubtract(stableForward, primitiveGradient));
        return Engine.TensorAdd(primitiveGradient, detachedForwardCorrection);
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
    private int ResolveClassAxis(Tensor<T> predicted, Tensor<T> target)
    {
        int rank = predicted.Shape.Length;
        int defaultAxis = rank - 1;

        if (target.Shape.Length == rank - 1)
        {
            int matchedAxis = -1;
            for (int axis = 0; axis < rank; axis++)
            {
                if (!ShapeMatchesWithAxisRemoved(predicted.Shape.ToArray(), target.Shape.ToArray(), axis))
                    continue;

                if (axis == defaultAxis)
                    return axis;

                if (matchedAxis < 0)
                    matchedAxis = axis;
            }

            if (matchedAxis >= 0)
                return matchedAxis;
        }

        if (target.Shape.Length == rank && ShapesEqual(predicted.Shape.ToArray(), target.Shape.ToArray()))
            return InferSoftTargetClassAxis(target, defaultAxis);

        return defaultAxis;
    }

    private static bool ShapeMatchesWithAxisRemoved(int[] predictedShape, int[] targetShape, int removedAxis)
    {
        if (predictedShape.Length != targetShape.Length + 1)
            return false;

        int targetAxis = 0;
        for (int axis = 0; axis < predictedShape.Length; axis++)
        {
            if (axis == removedAxis)
                continue;

            if (predictedShape[axis] != targetShape[targetAxis])
                return false;

            targetAxis++;
        }

        return true;
    }

    private static bool ShapesEqual(int[] left, int[] right)
    {
        if (left.Length != right.Length)
            return false;

        for (int i = 0; i < left.Length; i++)
            if (left[i] != right[i])
                return false;

        return true;
    }

    private int InferSoftTargetClassAxis(Tensor<T> target, int defaultAxis)
    {
        int rank = target.Shape.Length;
        if (rank <= 2)
            return defaultAxis;

        int alternateAxis = rank == 3 ? 0 : 1;
        if (alternateAxis == defaultAxis || alternateAxis >= rank)
            return defaultAxis;

        double defaultScore = ProbabilityAxisScore(target, defaultAxis);
        double alternateScore = ProbabilityAxisScore(target, alternateAxis);

        return alternateScore < 1e-6 && alternateScore + 1e-6 < defaultScore
            ? alternateAxis
            : defaultAxis;
    }

    private double ProbabilityAxisScore(Tensor<T> target, int axis)
    {
        int[] shape = target.Shape.ToArray();
        int axisSize = shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < shape.Length; i++)
            inner *= shape[i];

        int outer = target.Length / (axisSize * inner);
        var span = target.Data.Span;
        double totalDeviation = 0.0;
        int slices = 0;

        for (int o = 0; o < outer; o++)
        {
            int outerOffset = o * axisSize * inner;
            for (int i = 0; i < inner; i++)
            {
                double sum = 0.0;
                for (int c = 0; c < axisSize; c++)
                    sum += NumOps.ToDouble(span[outerOffset + c * inner + i]);

                totalDeviation += Math.Abs(sum - 1.0);
                slices++;
            }
        }

        return slices == 0 ? double.PositiveInfinity : totalDeviation / slices;
    }

    private Tensor<T> ClassIndicesToOneHot(Tensor<T> indices, int numClasses, int classAxis, int[] oneHotShape)
    {
        var indicesShape = indices.Shape.ToArray();
        var oneHot = new Tensor<T>(oneHotShape.ToArray());
        var indicesStrides = ComputeStrides(indicesShape);
        var oneHotStrides = ComputeStrides(oneHotShape);
        var indicesSpan = indices.Data.Span;
        var oneHotSpan = oneHot.Data.Span;
        // oneHot is initialized to zero on construction; we only need to
        // set the active class index per row.
        for (int i = 0; i < indicesSpan.Length; i++)
        {
            int classIdx = (int)Math.Round(NumOps.ToDouble(indicesSpan[i]));
            if (classIdx >= 0 && classIdx < numClasses)
            {
                int remaining = i;
                int indexAxis = 0;
                int oneHotOffset = 0;

                for (int axis = 0; axis < oneHotShape.Length; axis++)
                {
                    if (axis == classAxis)
                        continue;

                    int coord = indicesShape.Length == 0
                        ? 0
                        : remaining / indicesStrides[indexAxis];
                    if (indicesShape.Length > 0)
                        remaining %= indicesStrides[indexAxis];

                    oneHotOffset += coord * oneHotStrides[axis];
                    indexAxis++;
                }

                oneHotSpan[oneHotOffset + classIdx * oneHotStrides[classAxis]] = NumOps.One;
            }
        }
        return oneHot;
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }
}
