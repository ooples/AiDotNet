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
public class SparseCategoricalCrossEntropyLoss<T> : LossFunctionBase<T>
{
    /// <summary>
    /// Small value to prevent numerical instability with log(0).
    /// </summary>
    private readonly T _epsilon;

    /// <summary>
    /// Initializes a new instance of the SparseCategoricalCrossEntropyLoss class.
    /// </summary>
    public SparseCategoricalCrossEntropyLoss()
    {
        _epsilon = NumOps.FromDouble(1e-15);
    }

    /// <summary>
    /// Calculates the Sparse Categorical Cross Entropy loss between predicted probabilities and class indices.
    /// </summary>
    /// <param name="predicted">The predicted probability values for all classes.</param>
    /// <param name="actual">The actual class indices as floating-point values.</param>
    /// <returns>The sparse categorical cross entropy loss value.</returns>
    /// <remarks>
    /// For single-sample usage, if predicted has N classes and actual[0] = k (class index k),
    /// the loss is -log(predicted[k]).
    ///
    /// For consistency with the ILossFunction interface, both vectors should have compatible lengths.
    /// A common pattern is: predicted contains all class probabilities, and actual[0] contains the class index.
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when vectors have incompatible lengths or class indices are invalid.</exception>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

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

            // Clamp to prevent log(0)
            predictedProb = MathHelper.Clamp(predictedProb, _epsilon, NumOps.Subtract(NumOps.One, _epsilon));

            // Compute -log(predicted_probability)
            sum = NumOps.Add(sum, NumOps.Negate(NumOps.Log(predictedProb)));
            sampleCount++;
        }

        // Return average loss
        return NumOps.Divide(sum, NumOps.FromDouble(sampleCount));
    }

    /// <summary>
    /// Calculates the derivative of the Sparse Categorical Cross Entropy loss function.
    /// </summary>
    /// <param name="predicted">The predicted probability values for all classes.</param>
    /// <param name="actual">The actual class indices as floating-point values.</param>
    /// <returns>A vector containing the derivatives for each class probability.</returns>
    /// <remarks>
    /// The derivative is:
    /// - For the correct class: -1 / predicted[correct_class]
    /// - For all other classes: 0
    ///
    /// When used with softmax activation, this combines with the softmax derivative
    /// to produce the simplified gradient (predicted - one_hot_actual).
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when vectors have incompatible lengths or class indices are invalid.</exception>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        ValidateVectorLengths(predicted, actual);

        if (predicted.Length == 0)
        {
            throw new ArgumentException("Predicted vector cannot be empty.");
        }

        // Initialize gradient vector with zeros
        var gradient = new Vector<T>(predicted.Length);

        // Process each sample
        for (int i = 0; i < actual.Length; i++)
        {
            // Extract class index from actual
            int classIndex = NumOps.ToInt32(actual[i]);

            // Validate class index
            if (classIndex < 0 || classIndex >= predicted.Length)
            {
                throw new ArgumentException(
                    $"Class index {classIndex} at position {i} is out of bounds. " +
                    $"Expected value between 0 and {predicted.Length - 1}.");
            }

            // Clamp to prevent division by zero
            T predictedProb = MathHelper.Clamp(
                predicted[classIndex],
                _epsilon,
                NumOps.Subtract(NumOps.One, _epsilon));

            // Derivative for the correct class: -1 / predicted[correct_class]
            T derivative = NumOps.Negate(NumOps.Divide(NumOps.One, predictedProb));

            // Accumulate gradient (in case multiple samples point to the same class)
            gradient[classIndex] = NumOps.Add(gradient[classIndex], derivative);
        }

        // Average the gradients
        return gradient.Divide(NumOps.FromDouble(actual.Length));
    }
}
