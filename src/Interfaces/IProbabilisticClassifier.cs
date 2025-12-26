namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for classifiers that can output probability estimates for each class.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Probabilistic classifiers extend the basic classification interface by providing
/// methods to obtain class probability estimates. This is useful for understanding
/// the model's confidence in its predictions and for decision-making that considers
/// uncertainty.
/// </para>
/// <para><b>For Beginners:</b> Some classifiers don't just say "this is category A" -
/// they also tell you how confident they are.
///
/// For example, when classifying an email as spam:
/// - A basic classifier might just say: "Spam"
/// - A probabilistic classifier says: "90% spam, 10% not spam"
///
/// The probability information is valuable because:
/// - You can see when the model is uncertain (50%/50% vs 99%/1%)
/// - You can adjust the decision threshold (e.g., only mark as spam if >95% confident)
/// - You can combine predictions from multiple models more effectively
///
/// Common probabilistic classifiers include:
/// - Naive Bayes (naturally outputs probabilities)
/// - Logistic Regression (outputs probabilities via sigmoid/softmax)
/// - Random Forest (outputs probabilities via vote counting)
/// </para>
/// </remarks>
public interface IProbabilisticClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Predicts class probabilities for each sample in the input.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample and each column is a feature.</param>
    /// <returns>
    /// A matrix where each row corresponds to an input sample and each column corresponds to a class.
    /// The values represent the probability of the sample belonging to each class.
    /// For each row, the probabilities sum to 1.0.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method computes the probability of each sample belonging to each possible class.
    /// The output matrix has shape [num_samples, num_classes], and each row sums to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This method tells you how likely each sample is to belong to each category.
    ///
    /// Example output for 3 samples with 3 classes (Cat, Dog, Bird):
    /// ```
    /// Sample 1: [0.85, 0.10, 0.05]  // 85% Cat, 10% Dog, 5% Bird
    /// Sample 2: [0.05, 0.90, 0.05]  // 5% Cat, 90% Dog, 5% Bird
    /// Sample 3: [0.40, 0.35, 0.25]  // 40% Cat, 35% Dog, 25% Bird (uncertain!)
    /// ```
    ///
    /// Notice that each row adds up to 1.0 (or 100%).
    ///
    /// The third sample shows the model is uncertain - Cat is most likely,
    /// but it's not very confident. You might want to flag such cases for human review.
    /// </para>
    /// </remarks>
    Matrix<T> PredictProbabilities(Matrix<T> input);

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
    /// Log-probabilities are useful for numerical stability when working with very small
    /// probabilities, and for certain algorithms that work in log-space.
    /// </para>
    /// <para><b>For Beginners:</b> Log-probabilities are just probabilities transformed with the
    /// natural logarithm function.
    ///
    /// Why use log-probabilities?
    /// 1. Very small probabilities (like 0.0000001) become manageable numbers (like -16.1)
    /// 2. Multiplying probabilities becomes adding log-probabilities (simpler and more stable)
    /// 3. Many algorithms internally work in log-space for numerical stability
    ///
    /// For example:
    /// - Probability 0.9 becomes log(0.9) = -0.105
    /// - Probability 0.1 becomes log(0.1) = -2.303
    /// - Probability 0.001 becomes log(0.001) = -6.908
    ///
    /// Log-probabilities are always negative (since probabilities are between 0 and 1).
    /// Higher (less negative) values mean higher probability.
    ///
    /// Unless you're doing advanced work, you probably want PredictProbabilities() instead.
    /// </para>
    /// </remarks>
    Matrix<T> PredictLogProbabilities(Matrix<T> input);
}
