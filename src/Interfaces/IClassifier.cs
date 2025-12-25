using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the common interface for all classification algorithms in the AiDotNet library.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Classification is a type of machine learning algorithm used to predict categorical values
/// (discrete classes) rather than continuous values. This interface extends IFullModel with
/// classification-specific functionality.
/// </para>
/// <para><b>For Beginners:</b> Classification is about putting things into categories.
///
/// For example, classification can be used to:
/// - Predict whether an email is spam or not spam (binary classification)
/// - Identify handwritten digits (0-9) from images (multi-class classification)
/// - Determine which diseases a patient might have (multi-label classification)
/// - Rate customer satisfaction as Poor/Fair/Good/Excellent (ordinal classification)
///
/// Unlike regression algorithms (which predict numbers), classification algorithms predict
/// which category or categories something belongs to.
/// </para>
/// </remarks>
public interface IClassifier<T> : IFullModel<T, Matrix<T>, Vector<T>>
{
    /// <summary>
    /// Gets the number of classes that this classifier can predict.
    /// </summary>
    /// <value>
    /// The total number of distinct classes in the classification problem.
    /// </value>
    /// <remarks>
    /// <para>
    /// For binary classification, this returns 2. For multi-class classification,
    /// this returns the number of distinct classes learned during training.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many different categories the model can predict.
    ///
    /// Examples:
    /// - Spam detection: 2 classes (spam, not spam)
    /// - Digit recognition: 10 classes (0-9)
    /// - Sentiment analysis: 3 classes (negative, neutral, positive)
    ///
    /// The number of classes is determined during training based on the target labels.
    /// </para>
    /// </remarks>
    int NumClasses { get; }

    /// <summary>
    /// Gets the type of classification task this classifier is configured for.
    /// </summary>
    /// <value>
    /// A ClassificationTaskType value indicating whether this is binary, multi-class,
    /// multi-label, or ordinal classification.
    /// </value>
    /// <remarks>
    /// <para>
    /// The task type affects how predictions are interpreted and which loss functions
    /// and metrics are appropriate for the model.
    /// </para>
    /// <para><b>For Beginners:</b> The task type tells you what kind of classification problem
    /// this model is designed to solve.
    ///
    /// - Binary: Two categories (yes/no, spam/not-spam)
    /// - MultiClass: Multiple exclusive categories (exactly one answer)
    /// - MultiLabel: Multiple overlapping categories (can have multiple answers)
    /// - Ordinal: Ordered categories (like ratings or satisfaction levels)
    ///
    /// Knowing the task type helps you understand what kind of output to expect.
    /// </para>
    /// </remarks>
    ClassificationTaskType TaskType { get; }

    /// <summary>
    /// Gets the class labels learned during training.
    /// </summary>
    /// <value>
    /// A vector containing the unique class labels, or null if not yet trained.
    /// The index corresponds to the class index used in predictions.
    /// </value>
    /// <remarks>
    /// <para>
    /// Class labels provide the mapping between numeric class indices (0, 1, 2, ...)
    /// and the original label values from the training data. This is useful for
    /// interpreting predictions.
    /// </para>
    /// <para><b>For Beginners:</b> When you train a classifier, it learns what categories exist.
    ///
    /// For example, if you train on sentiment data with labels "positive", "neutral", "negative":
    /// - The model internally uses indices: 0, 1, 2
    /// - ClassLabels stores the mapping: [positive, neutral, negative]
    /// - When the model predicts class 2, you can look up that it means "negative"
    ///
    /// This is especially important when your original labels aren't already 0, 1, 2, etc.
    /// </para>
    /// </remarks>
    Vector<T>? ClassLabels { get; }
}
