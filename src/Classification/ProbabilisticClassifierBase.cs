namespace AiDotNet.Classification;

/// <summary>
/// Provides a base implementation for probabilistic classification algorithms that output
/// class probability estimates.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class extends ClassifierBase to add probabilistic prediction capabilities.
/// Probabilistic classifiers can output not just the predicted class, but also the probability
/// of each class. This is useful for understanding model confidence and making threshold-based
/// decisions.
/// </para>
/// <para>
/// The default Predict() method uses argmax of the probabilities to determine the class.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Probabilistic classifiers don't just say "this is category A" - they tell you how confident
/// they are. For example, instead of just "spam", they might say "92% spam, 8% not spam."
///
/// This additional information is valuable because:
/// - You can see when the model is uncertain (close to 50%/50%)
/// - You can adjust the decision threshold for your specific needs
/// - You can combine predictions from multiple models more effectively
/// </para>
/// </remarks>
public abstract class ProbabilisticClassifierBase<T> : ClassifierBase<T>, IProbabilisticClassifier<T>
{
    /// <summary>
    /// Initializes a new instance of the ProbabilisticClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier model. If null, default options will be used.</param>
    /// <param name="regularization">Regularization method to prevent overfitting. If null, no regularization will be applied.</param>
    /// <param name="lossFunction">Loss function for gradient computation. If null, defaults to Cross Entropy.</param>
    protected ProbabilisticClassifierBase(ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options, regularization, lossFunction)
    {
    }

    /// <summary>
    /// Predicts class labels for the given input data by taking the argmax of probabilities.
    /// </summary>
    /// <param name="input">The input features matrix where each row is an example and each column is a feature.</param>
    /// <returns>A vector of predicted class indices for each input example.</returns>
    /// <remarks>
    /// <para>
    /// This implementation uses the argmax of the probability distribution to determine
    /// the predicted class. For binary classification with a custom decision threshold,
    /// you may want to use PredictProbabilities() directly and apply your own threshold.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method picks the class with the highest probability for each sample.
    ///
    /// For example, if the probabilities are [0.1, 0.7, 0.2] for classes [A, B, C],
    /// this method returns class B (index 1) because it has the highest probability (0.7).
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (NumClasses <= 0)
        {
            throw new InvalidOperationException("Model has not been trained or has no classes.");
        }

        var probabilities = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Find the class with the highest probability (argmax)
            int bestClass = 0;
            T bestProb = probabilities[i, 0];

            for (int j = 1; j < NumClasses; j++)
            {
                if (NumOps.Compare(probabilities[i, j], bestProb) > 0)
                {
                    bestProb = probabilities[i, j];
                    bestClass = j;
                }
            }

            predictions[i] = NumOps.FromDouble(bestClass);
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities for each sample in the input.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <returns>
    /// A matrix where each row corresponds to an input sample and each column corresponds to a class.
    /// The values represent the probability of the sample belonging to each class.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This abstract method must be implemented by derived classes to compute
    /// class probabilities. The output matrix should have shape [num_samples, num_classes],
    /// and each row should sum to 1.0.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This method computes the probability of each sample belonging to each class.
    /// Each row in the output represents one sample, and each column represents one class.
    /// The values in each row sum to 1.0 (100% total probability).
    /// </para>
    /// </remarks>
    public abstract Matrix<T> PredictProbabilities(Matrix<T> input);

    /// <summary>
    /// Predicts log-probabilities for each class.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <returns>
    /// A matrix where each row corresponds to an input sample and each column corresponds to a class.
    /// The values are the natural logarithm of the class probabilities.
    /// </returns>
    /// <remarks>
    /// <para>
    /// The default implementation computes log(PredictProbabilities(input)).
    /// Subclasses that compute log-probabilities directly (like Naive Bayes) should
    /// override this method for better numerical stability.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Log-probabilities are probabilities transformed by the natural logarithm.
    /// They're useful for numerical stability when working with very small probabilities.
    ///
    /// For example:
    /// - Probability 0.9 → Log-probability -0.105
    /// - Probability 0.1 → Log-probability -2.303
    /// - Probability 0.001 → Log-probability -6.908
    ///
    /// Log-probabilities are always negative (since probabilities are between 0 and 1).
    /// Higher (less negative) values mean higher probability.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probabilities = PredictProbabilities(input);
        var logProbabilities = new Matrix<T>(probabilities.Rows, probabilities.Columns);

        for (int i = 0; i < probabilities.Rows; i++)
        {
            for (int j = 0; j < probabilities.Columns; j++)
            {
                // Compute log(p) with a small epsilon to avoid log(0)
                T p = probabilities[i, j];
                T epsilon = NumOps.FromDouble(1e-15);
                T clampedP = NumOps.Compare(p, epsilon) < 0 ? epsilon : p;
                logProbabilities[i, j] = NumOps.Log(clampedP);
            }
        }

        return logProbabilities;
    }

    /// <summary>
    /// Applies softmax normalization to convert raw scores to probabilities.
    /// </summary>
    /// <param name="scores">A matrix of raw scores [num_samples, num_classes].</param>
    /// <returns>A matrix of probabilities where each row sums to 1.0.</returns>
    /// <remarks>
    /// <para>
    /// The softmax function converts arbitrary real-valued scores into a probability
    /// distribution. It's defined as: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Softmax is a way to convert any set of numbers into probabilities.
    ///
    /// It has two key properties:
    /// 1. All output values are between 0 and 1
    /// 2. All output values sum to 1.0 (so they form a valid probability distribution)
    ///
    /// For example:
    /// - Scores [2.0, 1.0, 0.1] → Probabilities [0.659, 0.242, 0.099]
    /// - Notice: 0.659 + 0.242 + 0.099 = 1.0
    ///
    /// Higher scores result in higher probabilities, but the relationship is
    /// exponential (not linear), so the highest score gets a disproportionately
    /// large probability.
    /// </para>
    /// </remarks>
    protected Matrix<T> ApplySoftmax(Matrix<T> scores)
    {
        var probabilities = new Matrix<T>(scores.Rows, scores.Columns);

        for (int i = 0; i < scores.Rows; i++)
        {
            // Find max for numerical stability (subtract max before exp)
            T max = scores[i, 0];
            for (int j = 1; j < scores.Columns; j++)
            {
                if (NumOps.Compare(scores[i, j], max) > 0)
                {
                    max = scores[i, j];
                }
            }

            // Compute exp(score - max) and sum
            T sum = NumOps.Zero;
            var expScores = new T[scores.Columns];
            for (int j = 0; j < scores.Columns; j++)
            {
                expScores[j] = NumOps.Exp(NumOps.Subtract(scores[i, j], max));
                sum = NumOps.Add(sum, expScores[j]);
            }

            // Normalize to get probabilities
            for (int j = 0; j < scores.Columns; j++)
            {
                probabilities[i, j] = NumOps.Divide(expScores[j], sum);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Applies sigmoid function for binary classification probabilities.
    /// </summary>
    /// <param name="scores">A vector of raw scores.</param>
    /// <returns>A vector of probabilities (values between 0 and 1).</returns>
    /// <remarks>
    /// <para>
    /// The sigmoid function σ(x) = 1 / (1 + exp(-x)) converts any real number
    /// to a value between 0 and 1, making it suitable for binary classification.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Sigmoid squashes any number into the range [0, 1].
    ///
    /// - Very negative numbers → close to 0
    /// - Zero → 0.5
    /// - Very positive numbers → close to 1
    ///
    /// This is perfect for binary classification where you need a probability
    /// for the positive class. For example:
    /// - Score -3.0 → Probability 0.047 (unlikely positive)
    /// - Score 0.0 → Probability 0.500 (uncertain)
    /// - Score 3.0 → Probability 0.953 (likely positive)
    /// </para>
    /// </remarks>
    protected Vector<T> ApplySigmoid(Vector<T> scores)
    {
        var probabilities = new Vector<T>(scores.Length);

        for (int i = 0; i < scores.Length; i++)
        {
            // sigmoid(x) = 1 / (1 + exp(-x))
            // For numerical stability, use: sigmoid(x) = exp(x) / (1 + exp(x)) when x >= 0
            T x = scores[i];
            T negX = NumOps.Negate(x);

            if (NumOps.Compare(x, NumOps.Zero) >= 0)
            {
                // Use standard formula for positive x
                T expNegX = NumOps.Exp(negX);
                probabilities[i] = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
            }
            else
            {
                // Use exp(x) / (1 + exp(x)) for negative x
                T expX = NumOps.Exp(x);
                probabilities[i] = NumOps.Divide(expX, NumOps.Add(NumOps.One, expX));
            }
        }

        return probabilities;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["IsProbabilistic"] = true;
        return metadata;
    }
}
