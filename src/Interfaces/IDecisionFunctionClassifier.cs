namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for classifiers that compute a decision function for predictions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Some classifiers, particularly Support Vector Machines, make predictions based on a
/// decision function that measures the "confidence" or "distance" from the decision boundary.
/// This interface provides access to these raw decision values.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Think of classification as drawing a line (or surface) to separate classes.
/// The decision function tells you how far a point is from that line:
///
/// - Positive values: On the positive class side
/// - Negative values: On the negative class side
/// - Values near zero: Close to the decision boundary (uncertain)
///
/// For example, in spam detection:
/// - Decision value +3.5: Strongly predicted as spam
/// - Decision value +0.2: Weakly predicted as spam
/// - Decision value -0.1: Weakly predicted as not spam
/// - Decision value -2.8: Strongly predicted as not spam
///
/// This is different from probabilities (which range from 0 to 1).
/// Decision values can be any real number.
/// </para>
/// </remarks>
public interface IDecisionFunctionClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Computes the decision function for the input samples.
    /// </summary>
    /// <param name="input">The input features matrix where each row is a sample.</param>
    /// <returns>
    /// A matrix of decision values. For binary classification, this is a single column
    /// representing the signed distance to the decision boundary. For multi-class,
    /// the shape depends on the multi-class strategy (OvR vs OvO).
    /// </returns>
    /// <remarks>
    /// <para>
    /// The decision function provides the "raw" output of the classifier before
    /// any probability calibration. For SVMs, this is the signed distance to the
    /// separating hyperplane.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This gives you the classifier's "confidence" without converting to probabilities.
    ///
    /// Use this when you want to:
    /// - Apply custom thresholds for classification
    /// - Understand how confident the classifier is
    /// - Create your own probability calibration
    /// </para>
    /// </remarks>
    Matrix<T> DecisionFunction(Matrix<T> input);

    /// <summary>
    /// Gets the support vectors learned during training.
    /// </summary>
    /// <value>
    /// The matrix of support vectors, or null if not applicable or not trained.
    /// Each row is a support vector.
    /// </value>
    /// <remarks>
    /// <para>
    /// Support vectors are the training samples that lie closest to the decision
    /// boundary. They are the most "informative" samples and completely define
    /// the decision boundary.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Support vectors are the key training examples that define where the
    /// decision boundary goes. If you removed them, the classifier would change.
    /// Other training points (that are far from the boundary) don't affect
    /// the decision boundary at all.
    ///
    /// A classifier with fewer support vectors relative to training samples
    /// has learned a simpler model.
    /// </para>
    /// </remarks>
    Matrix<T>? SupportVectors { get; }

    /// <summary>
    /// Gets the number of support vectors.
    /// </summary>
    /// <value>
    /// The count of support vectors, or 0 if not trained.
    /// </value>
    int NSupportVectors { get; }
}
