using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for classification models, which are machine learning methods used to
/// predict categorical outcomes (discrete classes) rather than continuous values.
/// </summary>
/// <typeparam name="T">The data type used in matrix operations for the classifier model.</typeparam>
/// <remarks>
/// <para>
/// Classification is a supervised learning technique where the goal is to predict which category
/// or categories a sample belongs to. This class provides base configuration options for all
/// classification models, with specific classifiers potentially extending these options.
/// </para>
/// <para><b>For Beginners:</b> Classification is about predicting categories, not numbers.
///
/// Think of examples like:
/// - Is this email spam or not? (Binary classification)
/// - What type of animal is in this picture? (Multi-class classification)
/// - What topics does this article cover? (Multi-label classification)
/// - How satisfied is this customer? (Ordinal classification)
///
/// This class lets you configure how the classification model is set up and trained.
/// </para>
/// </remarks>
public class ClassifierOptions<T> : ModelOptions
{
    /// <summary>
    /// Gets or sets the type of classification task.
    /// </summary>
    /// <value>
    /// The classification task type. Default is Binary, which assumes a two-class problem.
    /// The task type will be automatically inferred from training data if not explicitly set.
    /// </value>
    /// <remarks>
    /// <para>
    /// The task type determines how the model interprets targets and outputs predictions.
    /// While this can be set explicitly, most classifiers will automatically detect the
    /// appropriate task type based on the training data.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the model what kind of classification problem you're solving.
    ///
    /// - Binary: Two classes (yes/no, spam/not-spam)
    /// - MultiClass: Multiple exclusive classes (pick exactly one)
    /// - MultiLabel: Multiple non-exclusive labels (pick all that apply)
    /// - Ordinal: Ordered classes (poor/fair/good/excellent)
    ///
    /// Usually, you don't need to set this manually - the model will figure it out
    /// from your training data. But you can set it explicitly if needed.
    /// </para>
    /// </remarks>
    public ClassificationTaskType TaskType { get; set; } = ClassificationTaskType.Binary;

    /// <summary>
    /// Gets or sets the decision threshold for binary classification.
    /// </summary>
    /// <value>
    /// A value between 0 and 1, defaulting to 0.5. Samples with probability above this
    /// threshold are classified as the positive class.
    /// </value>
    /// <remarks>
    /// <para>
    /// The decision threshold affects the trade-off between precision and recall.
    /// A higher threshold means fewer false positives but more false negatives,
    /// while a lower threshold has the opposite effect.
    /// </para>
    /// <para><b>For Beginners:</b> This sets the "cutoff" for making yes/no decisions in binary classification.
    ///
    /// With the default threshold of 0.5:
    /// - If P(positive) = 0.6 → Predict positive (0.6 > 0.5)
    /// - If P(positive) = 0.4 → Predict negative (0.4 < 0.5)
    ///
    /// When to adjust the threshold:
    /// - Medical diagnosis: Lower threshold (0.3) to catch more potential cases
    ///   (better to have a false alarm than miss a disease)
    /// - Spam filter: Higher threshold (0.8) to avoid false positives
    ///   (better to let some spam through than delete important emails)
    ///
    /// The optimal threshold depends on your specific application and the relative
    /// costs of false positives vs. false negatives.
    /// </para>
    /// </remarks>
    public double DecisionThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to use class weights to handle imbalanced datasets.
    /// </summary>
    /// <value>
    /// True to automatically compute class weights inversely proportional to class frequencies.
    /// False (default) to treat all classes equally.
    /// </value>
    /// <remarks>
    /// <para>
    /// Class weights help address class imbalance by giving more importance to
    /// minority classes during training. When enabled, the weight for each class
    /// is typically computed as n_samples / (n_classes * n_samples_per_class).
    /// </para>
    /// <para><b>For Beginners:</b> Real-world data is often imbalanced - one category appears much
    /// more often than others.
    ///
    /// Example: In fraud detection, 99% of transactions are legitimate, 1% are fraudulent.
    /// Without class weights, the model might just predict "legitimate" for everything
    /// and still get 99% accuracy - but it would miss all the fraud!
    ///
    /// With UseClassWeights = true:
    /// - Rare classes get higher weight (fraud becomes "worth" more)
    /// - Common classes get lower weight (legitimate becomes "worth" less)
    /// - The model is incentivized to correctly classify both types
    ///
    /// Enable this when your classes have very different frequencies.
    /// </para>
    /// </remarks>
    public bool UseClassWeights { get; set; } = false;

    /// <summary>
    /// Gets or sets custom class weights for each class.
    /// </summary>
    /// <value>
    /// An array of weights corresponding to each class, or null to use default weights.
    /// When UseClassWeights is true and this is null, weights are computed automatically.
    /// </value>
    /// <remarks>
    /// <para>
    /// Custom class weights allow fine-grained control over the importance of each class.
    /// This can be useful when the cost of misclassification varies by class in ways
    /// that are not captured by simple frequency-based weighting.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes you want specific control over how much each class "counts."
    ///
    /// Example: In a medical diagnosis scenario with 3 conditions:
    /// - Class 0: Common cold (low severity)
    /// - Class 1: Flu (moderate severity)
    /// - Class 2: Serious illness (high severity)
    ///
    /// You might set ClassWeights = [1.0, 2.0, 10.0] to make the model
    /// try extra hard not to miss the serious illness.
    ///
    /// The weights don't need to sum to 1 or match any particular formula -
    /// they just indicate relative importance. A weight of 10.0 means that class
    /// is considered 10x as important as a class with weight 1.0.
    /// </para>
    /// </remarks>
    public double[]? ClassWeights { get; set; }
}
