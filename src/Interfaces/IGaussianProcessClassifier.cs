namespace AiDotNet.Interfaces;

/// <summary>
/// Defines an interface for Gaussian Process classification, a probabilistic approach to classification
/// that provides uncertainty estimates along with class predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Gaussian Process Classification (GPC) is a powerful method that combines
/// the flexibility of Gaussian Processes with classification tasks.
///
/// Unlike regular classifiers that just say "this is class A," a GP classifier tells you:
/// - "This is probably class A (90% confident)"
/// - "This might be class A or B (60%/40% split)"
///
/// This is incredibly useful when:
/// - You need to know how confident the model is in its predictions
/// - You want to identify ambiguous cases that might need human review
/// - Your data has complex, non-linear patterns
/// - You have limited training data but need reliable predictions
///
/// The GP classifier works by:
/// 1. Learning a latent (hidden) function over your input space
/// 2. Passing this function through a link function (like sigmoid) to get probabilities
/// 3. Using approximation techniques (like Laplace) to handle the non-Gaussian likelihood
///
/// Key differences from GP regression:
/// - Regression predicts continuous values; classification predicts discrete labels
/// - Classification requires a link function to convert latent values to probabilities
/// - The posterior is not analytically tractable, requiring approximation methods
/// </para>
/// </remarks>
public interface IGaussianProcessClassifier<T>
{
    /// <summary>
    /// Trains the Gaussian Process classifier on labeled training data.
    /// </summary>
    /// <param name="X">The input features matrix, where each row represents an observation and each column represents a feature.</param>
    /// <param name="y">The class labels vector corresponding to each observation in X. For binary classification, use 0 and 1.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the classifier using your labeled training data.
    ///
    /// The parameters:
    /// - X: Your input data as a matrix (table of numbers)
    ///   - Each row is one example (like one email)
    ///   - Each column is one feature (like word count, sender reputation, etc.)
    /// - y: Your labels as a vector (list of class labels)
    ///   - For binary classification: 0 = negative class, 1 = positive class
    ///   - For multi-class: 0, 1, 2, ... for each class
    ///
    /// During fitting, the GP classifier:
    /// 1. Computes the kernel matrix to understand relationships between training points
    /// 2. Uses an approximation method (like Laplace) to find the posterior distribution
    /// 3. Optimizes to find the best latent function values
    ///
    /// After fitting, the model can make probabilistic predictions on new data.
    /// </para>
    /// </remarks>
    void Fit(Matrix<T> X, Vector<T> y);

    /// <summary>
    /// Predicts the class probabilities for a new input point.
    /// </summary>
    /// <param name="x">The input feature vector for which to make a prediction.</param>
    /// <returns>
    /// A tuple containing:
    /// - predictedClass: The most likely class label
    /// - probability: The probability of the predicted class (between 0 and 1)
    /// - variance: The uncertainty in the latent function prediction
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method makes a prediction for a single new data point
    /// and tells you how confident the model is.
    ///
    /// The returned values:
    /// - predictedClass: The class the model thinks this point belongs to (e.g., 0 or 1)
    /// - probability: How confident the model is (0.5 = uncertain, 0.95 = very confident)
    /// - variance: Additional measure of uncertainty from the latent function
    ///
    /// Example interpretations:
    /// - (class: 1, prob: 0.95, var: 0.01) → "Very confident this is class 1"
    /// - (class: 1, prob: 0.55, var: 0.5) → "Slightly leans toward class 1, but uncertain"
    /// - (class: 0, prob: 0.80, var: 0.1) → "Fairly confident this is class 0"
    ///
    /// The probability gives you actionable confidence:
    /// - High probability (>0.9): Trust the prediction
    /// - Medium probability (0.6-0.9): Prediction likely correct but check if important
    /// - Low probability (0.5-0.6): Model is uncertain, consider human review
    /// </para>
    /// </remarks>
    (int predictedClass, T probability, T variance) Predict(Vector<T> x);

    /// <summary>
    /// Predicts class probabilities for multiple input points.
    /// </summary>
    /// <param name="X">The input features matrix where each row is a point to classify.</param>
    /// <returns>
    /// A matrix where each row contains the class probabilities for the corresponding input point.
    /// For binary classification, this is a 2-column matrix [P(class=0), P(class=1)].
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method predicts class probabilities for multiple points at once.
    ///
    /// For binary classification, the output is a matrix with 2 columns:
    /// - Column 0: Probability of class 0
    /// - Column 1: Probability of class 1
    ///
    /// Each row sums to 1.0 (since the probabilities of all classes must add up to 100%).
    ///
    /// Example output for 3 samples:
    /// ```
    /// Sample 1: [0.15, 0.85]  → 85% likely class 1
    /// Sample 2: [0.95, 0.05]  → 95% likely class 0
    /// Sample 3: [0.48, 0.52]  → Nearly 50/50, model is uncertain
    /// ```
    ///
    /// This is useful when you need to:
    /// - Classify many examples efficiently
    /// - Analyze the distribution of confidence across your dataset
    /// - Identify which predictions are reliable vs. uncertain
    /// </para>
    /// </remarks>
    Matrix<T> PredictProbabilities(Matrix<T> X);

    /// <summary>
    /// Updates the kernel function used by the classifier.
    /// </summary>
    /// <param name="kernel">The new kernel function to use.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method changes how the classifier measures similarity between data points.
    ///
    /// The kernel function is crucial for GP classification because it defines:
    /// - How influence spreads between data points
    /// - What patterns the model can recognize
    /// - How smooth the decision boundary will be
    ///
    /// Different kernels for different situations:
    /// - RBF/Gaussian kernel: Smooth decision boundaries (good default choice)
    /// - Matern kernel: Adjustable smoothness (good for real-world data)
    /// - Linear kernel: Linear decision boundaries (like logistic regression)
    /// - Polynomial kernel: Curved decision boundaries
    ///
    /// Changing the kernel can significantly affect classification performance.
    /// If you've already trained the model, it will automatically retrain with the new kernel.
    /// </para>
    /// </remarks>
    void UpdateKernel(IKernelFunction<T> kernel);

    /// <summary>
    /// Gets the log marginal likelihood of the model, useful for hyperparameter optimization.
    /// </summary>
    /// <returns>The log marginal likelihood value.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The log marginal likelihood tells you how well the model fits the training data,
    /// accounting for model complexity.
    ///
    /// This value is useful for:
    /// - Comparing different kernel configurations
    /// - Optimizing kernel hyperparameters
    /// - Model selection (choosing between different GP classifiers)
    ///
    /// Higher values (less negative) indicate a better fit. However, unlike simple accuracy,
    /// this measure automatically penalizes overly complex models, helping prevent overfitting.
    ///
    /// When tuning hyperparameters, you want to find values that maximize this quantity.
    /// </para>
    /// </remarks>
    T GetLogMarginalLikelihood();

    /// <summary>
    /// Gets the number of classes learned during training.
    /// </summary>
    /// <value>The number of distinct classes (e.g., 2 for binary classification).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many different categories the classifier learned from the training data.
    ///
    /// - For binary classification (spam/not-spam): NumClasses = 2
    /// - For multi-class (cat/dog/bird): NumClasses = 3
    ///
    /// This value is determined during training based on the unique values in your labels (y).
    /// </para>
    /// </remarks>
    int NumClasses { get; }
}
