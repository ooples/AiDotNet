using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Classification.SemiSupervised;

/// <summary>
/// Provides a base implementation for semi-supervised classification algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class extends ClassifierBase with semi-supervised learning capabilities,
/// allowing derived classes to leverage both labeled and unlabeled data during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> This base class provides the foundation for building classifiers
/// that can learn from both labeled data (where you know the answers) and unlabeled data
/// (where you don't).
///
/// Think of it like a student learning with two types of study materials:
/// - A teacher's answer key (labeled data) - few examples but definitely correct
/// - Practice problems without answers (unlabeled data) - many examples to learn patterns from
///
/// By combining both, the student (classifier) can learn more effectively than using
/// just the answer key alone.
/// </para>
/// </remarks>
public abstract class SemiSupervisedClassifierBase<T> : ClassifierBase<T>, ISemiSupervisedClassifier<T>
{
    /// <summary>
    /// Gets or sets the number of labeled samples used in training.
    /// </summary>
    public int NumLabeledSamples { get; protected set; }

    /// <summary>
    /// Gets or sets the number of unlabeled samples used in training.
    /// </summary>
    public int NumUnlabeledSamples { get; protected set; }

    /// <summary>
    /// Stores the pseudo-labels assigned to unlabeled data during training.
    /// </summary>
    protected Vector<T>? PseudoLabels { get; set; }

    /// <summary>
    /// Stores the confidence scores for the pseudo-labels.
    /// </summary>
    protected Vector<T>? PseudoLabelConfidences { get; set; }

    /// <summary>
    /// The unlabeled feature matrix stored for prediction and analysis.
    /// </summary>
    protected Matrix<T>? UnlabeledData { get; set; }

    /// <summary>
    /// Initializes a new instance of the SemiSupervisedClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <param name="lossFunction">Loss function for gradient computation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the semi-supervised classifier with
    /// your specified settings. The options control how the algorithm behaves, regularization
    /// helps prevent the model from memorizing the training data (overfitting), and the loss
    /// function determines how prediction errors are measured.
    /// </para>
    /// </remarks>
    protected SemiSupervisedClassifierBase(
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options, regularization, lossFunction)
    {
    }

    /// <summary>
    /// Trains the classifier using both labeled and unlabeled data.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is where the semi-supervised magic happens.
    /// It first validates that your data is properly formatted, then calls the specific
    /// implementation in the derived class to actually perform the learning.
    ///
    /// The method ensures that:
    /// - Labeled features and labels have matching sample counts
    /// - All data has the same number of features
    /// - The data is stored for later analysis
    /// </para>
    /// </remarks>
    public virtual void TrainSemiSupervised(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        ValidateSemiSupervisedInput(labeledX, labeledY, unlabeledX);

        NumLabeledSamples = labeledX.Rows;
        NumUnlabeledSamples = unlabeledX.Rows;
        NumFeatures = labeledX.Columns;
        UnlabeledData = unlabeledX;

        // Extract class information from labeled data
        ClassLabels = ExtractClassLabels(labeledY);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(labeledY);

        // Call the derived class implementation
        TrainSemiSupervisedCore(labeledX, labeledY, unlabeledX);
    }

    /// <summary>
    /// Core implementation of semi-supervised training to be implemented by derived classes.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where each specific semi-supervised algorithm
    /// implements its unique learning strategy. For example:
    /// - Self-training iteratively labels high-confidence unlabeled samples
    /// - Label propagation spreads labels through a similarity graph
    /// - Co-training uses multiple views of the data
    /// </para>
    /// </remarks>
    protected abstract void TrainSemiSupervisedCore(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX);

    /// <summary>
    /// Validates the input data for semi-supervised training.
    /// </summary>
    /// <param name="labeledX">The feature matrix for labeled samples.</param>
    /// <param name="labeledY">The class labels for the labeled samples.</param>
    /// <param name="unlabeledX">The feature matrix for unlabeled samples.</param>
    /// <exception cref="ArgumentException">Thrown when the input data is invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that your data is properly formatted
    /// before training begins. It catches common mistakes like:
    /// - Having different numbers of samples and labels
    /// - Having different numbers of features in labeled vs unlabeled data
    /// - Providing empty datasets
    /// </para>
    /// </remarks>
    protected virtual void ValidateSemiSupervisedInput(Matrix<T> labeledX, Vector<T> labeledY, Matrix<T> unlabeledX)
    {
        if (labeledX.Rows == 0)
        {
            throw new ArgumentException("Labeled feature matrix cannot be empty.", nameof(labeledX));
        }

        if (labeledX.Rows != labeledY.Length)
        {
            throw new ArgumentException(
                $"Number of labeled samples ({labeledX.Rows}) must match number of labels ({labeledY.Length}).",
                nameof(labeledY));
        }

        if (unlabeledX.Rows == 0)
        {
            throw new ArgumentException("Unlabeled feature matrix cannot be empty.", nameof(unlabeledX));
        }

        if (labeledX.Columns != unlabeledX.Columns)
        {
            throw new ArgumentException(
                $"Labeled data has {labeledX.Columns} features but unlabeled data has {unlabeledX.Columns} features. They must match.",
                nameof(unlabeledX));
        }
    }

    /// <summary>
    /// Gets the pseudo-labels assigned to the unlabeled data during training.
    /// </summary>
    /// <returns>A vector of predicted labels for the unlabeled samples, or null if not available.</returns>
    public virtual Vector<T>? GetPseudoLabels()
    {
        return PseudoLabels;
    }

    /// <summary>
    /// Gets the confidence scores for the pseudo-labels.
    /// </summary>
    /// <returns>A vector of confidence scores for each pseudo-label, or null if not available.</returns>
    public virtual Vector<T>? GetPseudoLabelConfidences()
    {
        return PseudoLabelConfidences;
    }

    /// <summary>
    /// Trains the classifier using only labeled data (standard supervised training).
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Semi-supervised classifiers can also be trained using only
    /// labeled data, just like regular classifiers. This is useful when you don't have
    /// any unlabeled data or want to compare performance with and without unlabeled data.
    ///
    /// When you call this method, the classifier treats all data as labeled and ignores
    /// the semi-supervised capabilities.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // For standard training, we can either:
        // 1. Train as a regular classifier (if the derived class supports it)
        // 2. Create an empty unlabeled set and use semi-supervised training
        NumLabeledSamples = x.Rows;
        NumUnlabeledSamples = 0;
        NumFeatures = x.Columns;

        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        TrainSupervisedCore(x, y);
    }

    /// <summary>
    /// Core implementation of standard supervised training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target class labels.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows the semi-supervised classifier to also
    /// work as a regular classifier when no unlabeled data is available.
    /// </para>
    /// </remarks>
    protected abstract void TrainSupervisedCore(Matrix<T> x, Vector<T> y);
}
