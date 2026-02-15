namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the interface for semi-supervised classification algorithms that can learn from both labeled and unlabeled data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Semi-supervised learning is a machine learning paradigm that combines a small amount of labeled data
/// with a large amount of unlabeled data during training. This approach can significantly improve
/// learning accuracy when labeled data is scarce or expensive to obtain.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine you're teaching someone to recognize different types of flowers.
///
/// In traditional supervised learning, you'd need to label every single flower image: "This is a rose",
/// "This is a tulip", etc. But labeling thousands of images is time-consuming and expensive.
///
/// Semi-supervised learning is smarter - it uses a small set of labeled examples (maybe 100 labeled flowers)
/// combined with a large set of unlabeled images (maybe 10,000 unlabeled flower photos). The algorithm
/// learns patterns from the unlabeled data to improve its predictions.
///
/// Real-world applications include:
/// - Medical diagnosis: Only a few X-rays have expert diagnoses, but many are unlabeled
/// - Document classification: A few documents are categorized, thousands are not
/// - Speech recognition: Limited transcribed audio, abundant raw recordings
///
/// This interface extends IClassifier, meaning all semi-supervised classifiers can also be used
/// as regular classifiers and inherit all the IFullModel capabilities like serialization and checkpointing.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SemiSupervisedClassifier")]
public interface ISemiSupervisedClassifier<T> : IClassifier<T>
{
    /// <summary>
    /// Gets the number of labeled samples used in training.
    /// </summary>
    /// <value>
    /// The count of samples that had known class labels during training.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many examples with known answers were used to train the model.
    /// In semi-supervised learning, this is typically much smaller than the total number of training samples.
    /// </para>
    /// </remarks>
    int NumLabeledSamples { get; }

    /// <summary>
    /// Gets the number of unlabeled samples used in training.
    /// </summary>
    /// <value>
    /// The count of samples without class labels that were used to improve the model.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many examples without known answers were used.
    /// The algorithm uses patterns in this unlabeled data to make better predictions.
    /// A good semi-supervised algorithm can leverage thousands of unlabeled samples effectively.
    /// </para>
    /// </remarks>
    int NumUnlabeledSamples { get; }

    /// <summary>
    /// Trains the classifier using both labeled and unlabeled data.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples where each row is a sample and each column is a feature.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// This is the primary training method for semi-supervised classifiers. It uses the labeled data
    /// to learn initial patterns and then refines the model using the unlabeled data.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the model using two types of data:
    ///
    /// 1. Labeled data (labeledX, labeledY): Examples where you know the correct answer
    ///    - labeledX: The features (e.g., pixel values of flower images)
    ///    - labeledY: The labels (e.g., "rose", "tulip", "daisy")
    ///
    /// 2. Unlabeled data (unlabeledX): Examples where you don't know the answer
    ///    - The algorithm will try to figure out patterns from these on its own
    ///
    /// The magic happens when the model uses what it learned from labeled examples to make
    /// educated guesses about the unlabeled data, then uses those guesses to improve itself.
    /// </para>
    /// </remarks>
    void TrainSemiSupervised(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX);

    /// <summary>
    /// Gets the pseudo-labels assigned to the unlabeled data during training.
    /// </summary>
    /// <returns>
    /// A vector of predicted labels for the unlabeled samples, or null if not available.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> During training, the algorithm makes predictions for the unlabeled data.
    /// These predictions are called "pseudo-labels" because they're the algorithm's best guesses,
    /// not actual labels from an expert. This method lets you see what labels the algorithm assigned.
    ///
    /// This can be useful for:
    /// - Understanding how the algorithm is interpreting your unlabeled data
    /// - Finding samples that might need manual review
    /// - Debugging when the model isn't performing well
    /// </para>
    /// </remarks>
    Vector<T>? GetPseudoLabels();

    /// <summary>
    /// Gets the confidence scores for the pseudo-labels.
    /// </summary>
    /// <returns>
    /// A vector of confidence scores (0 to 1) for each pseudo-label, or null if not available.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Not all pseudo-labels are equally reliable. This method returns
    /// how confident the algorithm is about each of its guesses.
    ///
    /// A confidence of 0.95 means "I'm 95% sure this is correct"
    /// A confidence of 0.51 means "I'm barely more confident than random guessing"
    ///
    /// High-confidence pseudo-labels are more likely to be correct and can be trusted more
    /// during training. Low-confidence ones might be wrong and could hurt the model if trusted too much.
    /// </para>
    /// </remarks>
    Vector<T>? GetPseudoLabelConfidences();
}
