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

        T sum = NumOps.Zero;
        int sampleCount = 0;

        // Process each sample
        for (int i = 0; i < actual.Length; i++)
        {
            // Extract class index from actual (convert T to int)
            int classIndex = NumOps.ToInt32(actual[i]);

            // Validate class index
            if (classIndex < 0 || classIndex >= predicted.Length)
            {
                throw new ArgumentException(
                    $"Class index {classIndex} at position {i} is out of bounds. " +
                    $"Expected value between 0 and {predicted.Length - 1}.");
            }

            // Get predicted probability for the true class
            T predictedProb = predicted[classIndex];

            // Clamp to prevent log(0) using NumericalStabilityHelper
            predictedProb = NumericalStabilityHelper.ClampProbability(predictedProb, NumericalStabilityHelper.SmallEpsilon);

            // Compute -log(predicted_probability) with safe log
            sum = NumOps.Add(sum, NumOps.Negate(NumericalStabilityHelper.SafeLog(predictedProb, NumericalStabilityHelper.SmallEpsilon)));
            sampleCount++;
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

        var classIndices = new int[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            double rawIdx = NumOps.ToDouble(target[i]);
            int classIdx = (int)rawIdx;
            if (Math.Abs(rawIdx - classIdx) > 1e-6)
                throw new ArgumentException(
                    $"Target at position {i} is {rawIdx}, expected an integer class index.",
                    nameof(target));
            if (classIdx < 0 || classIdx >= numClasses)
                throw new ArgumentException(
                    $"Target index {classIdx} at position {i} is out of bounds. " +
                    $"Expected a value in [0, {numClasses}).",
                    nameof(target));
            classIndices[i] = classIdx;
        }

        // TensorOneHot returns [sampleCount, numClasses]. Restore higher-rank batch/sequence
        // geometry when it corresponds exactly to the prediction. For the legacy rank-1 API,
        // one prediction vector can be evaluated against several class indices. Its supervision
        // is a constant class-count mask: constructing that constant outside the tape is correct,
        // while every prediction-dependent operation below remains an IEngine operation.
        Tensor<T> oneHot;
        if (predicted.Rank == 1 && sampleCount > 1)
        {
            var classCounts = new T[numClasses];
            for (int i = 0; i < classIndices.Length; i++)
                classCounts[classIndices[i]] = NumOps.Add(classCounts[classIndices[i]], NumOps.One);
            oneHot = new Tensor<T>(classCounts, new[] { numClasses });
        }
        else
        {
            var indexTensor = new Tensor<int>(classIndices, target.Shape.ToArray());
            oneHot = Engine.TensorOneHot<T>(indexTensor, numClasses);
            if (predicted.Rank > 1 && !oneHot.Shape.ToArray().SequenceEqual(predicted.Shape.ToArray()))
                oneHot = Engine.Reshape(oneHot, predicted.Shape.ToArray());
        }

        var selectedLogP = Engine.TensorMultiply(oneHot, logP);
        var sum = Engine.ReduceSum(selectedLogP, null, keepDims: false);
        return Engine.TensorNegate(
            Engine.TensorDivideScalar(sum, NumOps.FromDouble(sampleCount)));
    }
}
