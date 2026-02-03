using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Provides a base implementation for multi-label classification algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class extends ClassifierBase with multi-label classification capabilities,
/// allowing derived classes to predict multiple labels per sample.
/// </para>
/// <para>
/// <b>For Beginners:</b> Multi-label classification is different from regular (multi-class) classification:
///
/// - In multi-class: Each item gets exactly ONE label (e.g., an email is either spam OR not spam)
/// - In multi-label: Each item can get ZERO, ONE, or MANY labels (e.g., a movie can be action AND comedy AND romance)
///
/// Real-world examples of multi-label problems:
/// - Document tagging: An article might be about "politics", "economy", AND "environment"
/// - Image classification: A photo might contain "person", "dog", AND "outdoor"
/// - Music categorization: A song might be "rock", "electronic", AND "80s"
///
/// This base class provides common functionality for all multi-label classifiers in the library.
/// </para>
/// </remarks>
public abstract class MultiLabelClassifierBase<T> : ClassifierBase<T>, IMultiLabelClassifier<T>
{
    /// <summary>
    /// Gets or sets the number of labels that can be predicted.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the total number of possible labels in your problem.
    /// For example, if you're classifying movies into 10 different genre categories,
    /// NumLabels would be 10.
    /// </para>
    /// </remarks>
    public int NumLabels { get; protected set; }

    /// <summary>
    /// The label names or indices for each label column.
    /// </summary>
    protected T[]? LabelNames { get; set; }

    /// <summary>
    /// Threshold for converting probabilities to binary predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When the model outputs probabilities (like 0.7 for "comedy"),
    /// we need to decide if that's high enough to say "yes, this is a comedy".
    /// The threshold determines the cutoff point. A threshold of 0.5 means any probability
    /// above 50% will be predicted as present.
    /// </para>
    /// </remarks>
    protected T PredictionThreshold { get; set; }

    /// <summary>
    /// Initializes a new instance of the MultiLabelClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <param name="lossFunction">Loss function for gradient computation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the multi-label classifier with your
    /// specified settings. The threshold defaults to 0.5, meaning any label with probability
    /// above 50% will be predicted as present.
    /// </para>
    /// </remarks>
    protected MultiLabelClassifierBase(
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null,
        ILossFunction<T>? lossFunction = null)
        : base(options, regularization, lossFunction)
    {
        PredictionThreshold = NumOps.FromDouble(0.5);
    }

    /// <summary>
    /// Trains the multi-label classifier.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix where each column is a binary label indicator.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the model using your labeled data.
    /// Unlike regular classification where y is a vector (one label per sample),
    /// here y is a matrix where each column represents a different label.
    ///
    /// For example, if you have 100 samples and 5 possible labels:
    /// - x would be 100 rows × (number of features) columns
    /// - y would be 100 rows × 5 columns
    /// - y[i,j] = 1 means sample i has label j, y[i,j] = 0 means it doesn't
    /// </para>
    /// </remarks>
    public virtual void TrainMultiLabel(Matrix<T> x, Matrix<T> y)
    {
        ValidateMultiLabelInput(x, y);

        NumFeatures = x.Columns;
        NumLabels = y.Columns;

        // For base classifier compatibility, we use the number of labels as classes
        // This is a simplification - the actual prediction is multi-label
        NumClasses = 2; // Binary for each label
        TaskType = ClassificationTaskType.Binary;

        TrainMultiLabelCore(x, y);
    }

    /// <summary>
    /// Core implementation of multi-label training to be implemented by derived classes.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where each specific multi-label algorithm implements
    /// its unique learning strategy. For example:
    /// - Binary Relevance trains one classifier per label independently
    /// - Classifier Chains trains classifiers in a chain, using previous predictions as features
    /// - Label Powerset treats each unique label combination as a separate class
    /// </para>
    /// </remarks>
    protected abstract void TrainMultiLabelCore(Matrix<T> x, Matrix<T> y);

    /// <summary>
    /// Validates the input data for multi-label training.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix.</param>
    /// <exception cref="ArgumentException">Thrown when the input data is invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that your data is properly formatted before
    /// training begins. It catches common mistakes like having different numbers of samples
    /// in your features and labels.
    /// </para>
    /// </remarks>
    protected virtual void ValidateMultiLabelInput(Matrix<T> x, Matrix<T> y)
    {
        if (x.Rows == 0)
        {
            throw new ArgumentException("Feature matrix cannot be empty.", nameof(x));
        }

        if (x.Rows != y.Rows)
        {
            throw new ArgumentException(
                $"Number of samples in features ({x.Rows}) must match number of samples in labels ({y.Rows}).",
                nameof(y));
        }

        if (y.Columns == 0)
        {
            throw new ArgumentException("Label matrix must have at least one label column.", nameof(y));
        }
    }

    /// <summary>
    /// Predicts binary indicators for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A binary matrix where each row is a sample and each column is a label indicator.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method predicts which labels apply to each input sample.
    /// The output is a binary matrix where:
    /// - 1 means the label is predicted to be present
    /// - 0 means the label is predicted to be absent
    ///
    /// For example, if you're predicting movie genres and the output for a movie is [1, 0, 1, 0, 0],
    /// it means the first and third genres apply to this movie.
    /// </para>
    /// </remarks>
    public virtual Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        var probabilities = PredictMultiLabelProbabilities(input);
        var predictions = new Matrix<T>(input.Rows, NumLabels);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int j = 0; j < NumLabels; j++)
            {
                predictions[i, j] = NumOps.Compare(probabilities[i, j], PredictionThreshold) >= 0
                    ? NumOps.One
                    : NumOps.Zero;
            }
        }

        return predictions;
    }

    /// <summary>
    /// Predicts probabilities for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A probability matrix where each row is a sample and each column is the probability of that label.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method predicts the probability that each label applies
    /// to each input sample. Values range from 0 to 1:
    /// - 0.9 means the model is 90% confident this label applies
    /// - 0.1 means the model is only 10% confident
    ///
    /// This is useful when you want to see the model's confidence rather than just
    /// yes/no predictions.
    /// </para>
    /// </remarks>
    public abstract Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input);

    /// <summary>
    /// Standard classification prediction - returns the most likely single label per sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A vector of the most confident label index for each sample.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-label classifiers can also work as regular classifiers
    /// by returning just the most likely label. This method is provided for compatibility
    /// with the base IClassifier interface, but for true multi-label prediction you should
    /// use PredictMultiLabel() instead.
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probabilities = PredictMultiLabelProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Find the label with highest probability
            int maxIndex = 0;
            T maxProb = probabilities[i, 0];

            for (int j = 1; j < NumLabels; j++)
            {
                if (NumOps.Compare(probabilities[i, j], maxProb) > 0)
                {
                    maxProb = probabilities[i, j];
                    maxIndex = j;
                }
            }

            predictions[i] = NumOps.FromDouble(maxIndex);
        }

        return predictions;
    }

    /// <summary>
    /// Standard training method - converts vector labels to multi-label matrix.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The target labels as a vector (will be converted to binary indicators).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to use the multi-label classifier
    /// with traditional single-label data. It converts the label vector into a multi-label
    /// matrix format internally.
    /// </para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        // Convert single-label to multi-label format
        // Determine unique labels
        var uniqueLabels = new HashSet<double>();
        for (int i = 0; i < y.Length; i++)
        {
            uniqueLabels.Add(NumOps.ToDouble(y[i]));
        }

        int numUniqueLabels = uniqueLabels.Count;
        var labelList = uniqueLabels.OrderBy(l => l).ToList();

        // Create multi-label matrix (one-hot encoding)
        var multiLabelY = new Matrix<T>(y.Length, numUniqueLabels);
        for (int i = 0; i < y.Length; i++)
        {
            int labelIndex = labelList.IndexOf(NumOps.ToDouble(y[i]));
            multiLabelY[i, labelIndex] = NumOps.One;
        }

        TrainMultiLabel(x, multiLabelY);
    }

    /// <summary>
    /// Sets the threshold for converting probabilities to binary predictions.
    /// </summary>
    /// <param name="threshold">The threshold value (typically between 0 and 1).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you adjust how confident the model needs to be
    /// before predicting a label is present.
    /// - Higher threshold (e.g., 0.8): More conservative, fewer false positives
    /// - Lower threshold (e.g., 0.3): More liberal, fewer false negatives
    /// </para>
    /// </remarks>
    public void SetPredictionThreshold(T threshold)
    {
        PredictionThreshold = threshold;
    }
}
