using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Implements the Sparse Categorical Cross Entropy loss function for multi-class classification with integer labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sparse Categorical Cross Entropy is similar to Categorical Cross Entropy but is used
/// when labels are provided as class indices (0, 1, 2, ...) rather than one-hot encoded vectors.
///
/// This is more memory efficient for problems with many classes, as you only need to store the class index
/// instead of a full one-hot encoded vector.
///
/// The formula is: SCCE = -(1/n) * Σ[log(predicted[actual_class_index])]
///
/// Where:
/// - actual contains the class indices (e.g., 0, 1, 2, 3 for a 4-class problem)
/// - predicted contains the predicted probabilities for all classes
/// - We extract the probability for the correct class using the index from actual
///
/// Example:
/// - If actual[i] = 2.0 (class index 2), and predicted has probabilities [0.1, 0.2, 0.6, 0.1],
///   then we take predicted[2] = 0.6 and compute -log(0.6)
///
/// Key properties:
/// - More memory efficient than categorical cross-entropy for many-class problems
/// - Predicted values should be probabilities (between 0 and 1) from a softmax layer
/// - Actual values should be valid class indices (0 to num_classes-1)
/// - Often used with the softmax activation function in neural networks
///
/// To use this loss function with the Vector interface:
/// - For a single sample: predicted = [p_class0, p_class1, ..., p_classN], actual = [true_class_index]
/// - For batches: flatten your data appropriately or process samples individually
/// </para>
/// </remarks>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.MultiClass)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = false, RequiresProbabilityInputs = true, ApiShape = LossApiShape.SparseIndex, ExpectedOutput = OutputType.Probabilities)]
public class SparseCategoricalCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Initializes a new instance of the SparseCategoricalCrossEntropyLoss class.
    /// </summary>
    public SparseCategoricalCrossEntropyLoss()
    {
    }

    /// <summary>
    /// Calculates the Sparse Categorical Cross Entropy loss between predicted probabilities and class indices.
    /// </summary>
    /// <param name="predicted">The predicted probability values for all classes (length = num_classes).</param>
    /// <param name="actual">The actual class indices as floating-point values (length = batch_size or 1 for single sample).</param>
    /// <returns>The sparse categorical cross entropy loss value.</returns>
    /// <remarks>
    /// For single-sample usage, if predicted has N classes and actual[0] = k (class index k),
    /// the loss is -log(predicted[k]).
    ///
    /// Unlike other loss functions, predicted and actual can have different lengths:
    /// - predicted.Length = number of classes (N)
    /// - actual.Length = number of samples in batch (M)
    /// Each actual[i] contains the class index for sample i.
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when class indices are invalid or vectors are empty.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        // Note: We do NOT validate that predicted and actual have the same length
        // In sparse categorical cross-entropy, they can differ:
        // - predicted contains N class probabilities
        // - actual contains M class indices (where M can differ from N)

        if (predicted.Length == 0)
        {
            throw new ArgumentException("Predicted vector cannot be empty.");
        }

        int sampleCount = actual.Length;
        int numClasses = predicted.Length;

        // Class indices are the targets rounded to the nearest integer (NumOps.ToInt32);
        // every one must lie in [0, numClasses).
        var rawIndices = actual.AsTensor();
        var classIndices = Engine.TensorRound(rawIndices);
        var invalid = Engine.TensorMax(
            Engine.TensorMax(
                Engine.TensorLessThan(classIndices, NumOps.Zero),
                Engine.TensorGreaterThan(classIndices, NumOps.FromDouble(numClasses - 1))),
            Engine.TensorNotEquals(classIndices, classIndices));
        if (sampleCount > 0 && NumOps.GreaterThan(Engine.TensorMaxValue(invalid), NumOps.Zero))
        {
            // First offending position (nonzero lists positions in ascending order on every engine). The raw
            // value is reported as is: converting it could itself overflow (for example 3000000000m).
            int position = Engine.TensorNonzero(invalid).GetFlat(0);
            T classIndex = rawIndices.GetFlat(position);
            throw new ArgumentException(
                $"Class index {classIndex} at position {position} is out of bounds. " +
                $"Expected value between 0 and {predicted.Length - 1}.");
        }

        T sum = NumOps.Zero;
        if (sampleCount > 0)
        {
            // Get predicted probability for each sample's true class
            // Indexed gather straight from the class vector (the indices were validated above)
            var predictedProb = Engine.TensorIndexSelect(predicted.AsTensor(), classIndices.Cast<int>(), 0);

            // Clamp to prevent log(0) (NumericalStabilityHelper.ClampProbability), then -log(p)
            T eps = NumOps.FromDouble(NumericalStabilityHelper.SmallEpsilon);
            predictedProb = Engine.TensorClamp(predictedProb, eps, NumOps.Subtract(NumOps.One, eps));
            sum = NumOps.Negate(Engine.TensorSum(Engine.TensorLog(predictedProb)));
        }

        // Return average loss
        return NumOps.Divide(sum, NumOps.FromDouble(sampleCount));
    }


    /// <inheritdoc />
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted.Length == 0)
            throw new ArgumentException("Predicted tensor cannot be empty.", nameof(predicted));

        // This loss consumes probabilities, not logits (see RequiresProbabilityInputs above).
        // Keep the supplied prediction on the tape and use the same lower clamp as CalculateLoss.
        // Applying Softmax here would change both the documented
        // objective and its derivative from -1 / p to softmax(p) - target.
        var safePredicted = Engine.TensorClampMin(
            predicted,
            NumOps.FromDouble(NumericalStabilityHelper.SmallEpsilon));
        var logP = Engine.TensorLog(safePredicted);

        // SparseCategoricalCrossEntropy always consumes integer class indices. Shape equality is
        // not evidence of a dense target: a rank-1 prediction with C classes can legitimately be
        // evaluated against C sparse labels (for example [0, 1, ..., C-1]). Dense/one-hot callers
        // belong on CategoricalCrossEntropyLoss instead.
        //
        // Converting the supervision to an
        // integer tensor is intentionally non-differentiable, but class selection itself must be
        // composed from IEngine operations. The former element-wise copy into gatheredLogP
        // detached logP from the tape and made every sparse gradient undefined.
        int sampleCount = target.Length;
        if (sampleCount == 0)
            throw new ArgumentException("Target tensor cannot be empty.", nameof(target));

        int numClasses = predicted.Shape[^1];
        if (predicted.Rank > 1)
        {
            int expectedSamples = predicted.Length / numClasses;
            if (sampleCount != expectedSamples)
            {
                throw new ArgumentException(
                    $"Sparse target contains {sampleCount} class indices, but prediction shape " +
                    $"[{string.Join(", ", predicted.Shape)}] requires {expectedSamples}.",
                    nameof(target));
            }
        }

        // Validate every class index at once: (int)rawIdx truncates, the value must be within 1e-6 of
        // that integer, and the integer must lie in [0, numClasses). NaN is never a valid index.
        var flatTarget = Engine.Reshape(target, new[] { sampleCount });
        var truncated = Engine.TensorTrunc(flatTarget);
        // Outside the int range the old (int) cast could not represent the value: not an integer index.
        var notInteger = Engine.TensorMax(
            Engine.TensorGreaterThan(
                Engine.TensorAbs(Engine.TensorSubtract(flatTarget, truncated)), NumOps.FromDouble(1e-6)),
            Engine.TensorGreaterThan(Engine.TensorAbs(flatTarget), NumOps.FromDouble(int.MaxValue)));
        var outOfRange = Engine.TensorMax(
            Engine.TensorMax(
                Engine.TensorLessThan(truncated, NumOps.Zero),
                Engine.TensorGreaterThan(truncated, NumOps.FromDouble(numClasses - 1))),
            Engine.TensorNotEquals(flatTarget, flatTarget));
        var invalid = Engine.TensorMax(notInteger, outOfRange);
        if (NumOps.GreaterThan(Engine.TensorMaxValue(invalid), NumOps.Zero))
        {
            // First offending position, reported exactly as the per-element check did (nonzero lists
            // positions in ascending order on every engine).
            int i = Engine.TensorNonzero(invalid).GetFlat(0);
            T rawIdx = flatTarget.GetFlat(i);
            if (NumOps.GreaterThan(notInteger.GetFlat(i), NumOps.Zero))
                throw new ArgumentException(
                    $"Target at position {i} is {rawIdx}, expected an integer class index.",
                    nameof(target));

            // NaN is reported as the out-of-range index the old (int) cast produced.
            T truncatedIdx = truncated.GetFlat(i);
            int classIdx = NumOps.Equals(truncatedIdx, truncatedIdx) ? NumOps.ToInt32(truncatedIdx) : int.MinValue;
            throw new ArgumentException(
                $"Target index {classIdx} at position {i} is out of bounds. " +
                $"Expected a value in [0, {numClasses}).",
                nameof(target));
        }

        // One one-hot row per sample: compare each class index against 0..numClasses-1.
        var rowShape = new[] { sampleCount, numClasses };
        var oneHotRows = Engine.TensorEquals(
            Engine.TensorBroadcastTo(Engine.Reshape(truncated, new[] { sampleCount, 1 }), rowShape),
            Engine.TensorBroadcastTo(Engine.TensorArange<T>(0, numClasses, 1), rowShape));

        // The one-hot is [target shape..., numClasses]. Restore higher-rank batch/sequence
        // geometry when it corresponds exactly to the prediction. For the legacy rank-1 API,
        // one prediction vector can be evaluated against several class indices. Its supervision
        // is a constant class-count mask: constructing that constant outside the tape is correct,
        // while every prediction-dependent operation below remains an IEngine operation.
        Tensor<T> oneHot;
        if (predicted.Rank == 1 && sampleCount > 1)
        {
            oneHot = Engine.ReduceSum(oneHotRows, new[] { 0 }, keepDims: false);
        }
        else
        {
            var oneHotShape = target.Shape.ToArray().Concat(new[] { numClasses }).ToArray();
            oneHot = Engine.Reshape(oneHotRows, oneHotShape);
            if (predicted.Rank > 1 && !oneHot.Shape.ToArray().SequenceEqual(predicted.Shape.ToArray()))
                oneHot = Engine.Reshape(oneHot, predicted.Shape.ToArray());
        }

        var selectedLogP = Engine.TensorMultiply(oneHot, logP);
        var sum = Engine.ReduceSum(selectedLogP, null, keepDims: false);
        return Engine.TensorNegate(
            Engine.TensorDivideScalar(sum, NumOps.FromDouble(sampleCount)));
    }
}
