using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Classification.MultiLabel;

/// <summary>
/// Implements the Binary Relevance approach for multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Binary Relevance is the simplest multi-label classification method. It transforms the
/// multi-label problem into multiple independent binary classification problems, one for each label.
/// </para>
/// <para>
/// <b>For Beginners:</b> Binary Relevance takes the "divide and conquer" approach:
///
/// Instead of trying to predict all labels at once, it trains a separate binary classifier for
/// each possible label. For example, if you're classifying movies into 5 genres:
///
/// 1. Train a classifier that asks: "Is this movie action?" (yes/no)
/// 2. Train a classifier that asks: "Is this movie comedy?" (yes/no)
/// 3. Train a classifier that asks: "Is this movie drama?" (yes/no)
/// 4. And so on for each genre...
///
/// To predict labels for a new movie, we run all 5 classifiers and combine their answers.
///
/// Pros:
/// - Simple to understand and implement
/// - Can use any binary classifier
/// - Parallelizable (each label classifier can train independently)
///
/// Cons:
/// - Ignores correlations between labels (e.g., "horror" and "thriller" often appear together)
/// - May produce inconsistent predictions (e.g., predicting "sequel" without "action")
///
/// For problems where labels are correlated, consider using Classifier Chains or Label Powerset instead.
/// </para>
/// </remarks>
public class BinaryRelevance<T> : MultiLabelClassifierBase<T>
{
    #region Fields

    /// <summary>
    /// Factory function to create binary classifiers for each label.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a function that creates new binary classifiers.
    /// Binary Relevance needs to create one classifier per label, so it uses this factory
    /// to create identical classifiers for each label. You provide this so you can choose
    /// what type of classifier to use (logistic regression, SVM, etc.).
    /// </para>
    /// </remarks>
    private readonly Func<IClassifier<T>> _classifierFactory;

    /// <summary>
    /// The trained binary classifiers, one per label.
    /// </summary>
    private IClassifier<T>[]? _labelClassifiers;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the BinaryRelevance class with a classifier factory.
    /// </summary>
    /// <param name="classifierFactory">A function that creates binary classifiers. Called once per label.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Regularization method to prevent overfitting.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To create a BinaryRelevance classifier, you need to tell it
    /// what kind of binary classifier to use for each label. You do this by providing a
    /// factory function that creates new classifier instances.
    ///
    /// Example usage:
    /// <code>
    /// // Using logistic regression for each label
    /// var br = new BinaryRelevance&lt;double&gt;(() => new LogisticRegression&lt;double&gt;());
    ///
    /// // Using SVM for each label
    /// var br = new BinaryRelevance&lt;double&gt;(() => new SupportVectorClassifier&lt;double&gt;());
    /// </code>
    /// </para>
    /// </remarks>
    public BinaryRelevance(
        Func<IClassifier<T>> classifierFactory,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _classifierFactory = classifierFactory ?? throw new ArgumentNullException(nameof(classifierFactory));
    }

    #endregion

    #region Training

    /// <summary>
    /// Core implementation of multi-label training using Binary Relevance.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="y">The multi-label target matrix.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method trains one binary classifier for each label column.
    /// For each label, it creates a separate classification problem where:
    /// - Positive samples are those with that label (y[i,j] = 1)
    /// - Negative samples are those without that label (y[i,j] = 0)
    ///
    /// Each classifier learns to predict whether a sample has that specific label,
    /// completely ignoring all other labels.
    /// </para>
    /// </remarks>
    protected override void TrainMultiLabelCore(Matrix<T> x, Matrix<T> y)
    {
        _labelClassifiers = new IClassifier<T>[NumLabels];

        for (int labelIndex = 0; labelIndex < NumLabels; labelIndex++)
        {
            // Create a binary classifier for this label
            var classifier = _classifierFactory();

            // Extract the binary labels for this label column
            var binaryLabels = new Vector<T>(y.Rows);
            for (int i = 0; i < y.Rows; i++)
            {
                binaryLabels[i] = y[i, labelIndex];
            }

            // Train the classifier
            classifier.Train(x, binaryLabels);
            _labelClassifiers[labelIndex] = classifier;
        }
    }

    #endregion

    #region Prediction

    /// <summary>
    /// Predicts probabilities for each label for each sample.
    /// </summary>
    /// <param name="input">The input feature matrix.</param>
    /// <returns>A probability matrix where each row is a sample and each column is the probability of that label.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs each label's binary classifier on the input
    /// and collects their probability predictions. Each classifier independently predicts
    /// the probability that its label is present.
    ///
    /// The output is a matrix where:
    /// - Row i contains probabilities for sample i
    /// - Column j contains the probability that label j is present
    /// </para>
    /// </remarks>
    public override Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_labelClassifiers is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var probabilities = new Matrix<T>(input.Rows, NumLabels);

        for (int labelIndex = 0; labelIndex < NumLabels; labelIndex++)
        {
            var classifier = _labelClassifiers[labelIndex];

            // Get predictions for this label
            if (classifier is IProbabilisticClassifier<T> probabilisticClassifier)
            {
                // Get probability of the positive class (label present)
                var labelProbs = probabilisticClassifier.PredictProbabilities(input);

                for (int i = 0; i < input.Rows; i++)
                {
                    // Assuming binary classification, column 1 is probability of class 1 (label present)
                    probabilities[i, labelIndex] = labelProbs.Columns > 1
                        ? labelProbs[i, 1]
                        : labelProbs[i, 0];
                }
            }
            else
            {
                // Fall back to binary predictions
                var predictions = classifier.Predict(input);
                for (int i = 0; i < input.Rows; i++)
                {
                    probabilities[i, labelIndex] = NumOps.Compare(predictions[i], NumOps.One) >= 0
                        ? NumOps.One
                        : NumOps.Zero;
                }
            }
        }

        return probabilities;
    }

    #endregion

    #region Abstract Method Implementations

    /// <summary>
    /// Gets the model type identifier.
    /// </summary>
    /// <returns>The ModelType enum value for Binary Relevance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This identifies what kind of model this is within the
    /// AiDotNet library's type system.
    /// </para>
    /// </remarks>
    protected override ModelType GetModelType()
    {
        return ModelType.BinaryRelevanceClassifier;
    }

    /// <summary>
    /// Gets all learnable parameters of the model as a single vector.
    /// </summary>
    /// <returns>A concatenated vector of all label classifiers' parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Binary Relevance contains multiple classifiers, each with
    /// its own parameters. This method collects all parameters from all classifiers
    /// into a single vector for operations like serialization or optimization.
    /// </para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        if (_labelClassifiers is null || _labelClassifiers.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Collect all parameters from all classifiers
        var allParams = new List<T>();
        foreach (var classifier in _labelClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            for (int i = 0; i < classifierParams.Length; i++)
            {
                allParams.Add(classifierParams[i]);
            }
        }

        return new Vector<T>(allParams.ToArray());
    }

    /// <summary>
    /// Creates a new instance of the model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters to use.</param>
    /// <returns>A new instance of the classifier.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a new Binary Relevance classifier and distributes
    /// the provided parameters among its label classifiers.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newClassifier = new BinaryRelevance<T>(_classifierFactory, Options, Regularization);
        newClassifier.NumLabels = NumLabels;
        newClassifier.NumFeatures = NumFeatures;
        newClassifier.SetParameters(parameters);
        return newClassifier;
    }

    /// <summary>
    /// Sets the parameters of this model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This distributes the provided parameters among all label classifiers.
    /// The parameters must be in the same order they were retrieved by GetParameters().
    /// </para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        if (_labelClassifiers is null || _labelClassifiers.Length == 0)
        {
            return;
        }

        int paramIndex = 0;
        foreach (var classifier in _labelClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            var newParams = new Vector<T>(classifierParams.Length);

            for (int i = 0; i < classifierParams.Length && paramIndex < parameters.Length; i++)
            {
                newParams[i] = parameters[paramIndex++];
            }

            classifier.SetParameters(newParams);
        }
    }

    /// <summary>
    /// Computes gradients for gradient-based optimization.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target labels (expected to be multi-label format).</param>
    /// <param name="lossFunction">The loss function (optional).</param>
    /// <returns>A concatenated gradient vector from all label classifiers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes gradients for each label classifier independently
    /// and concatenates them into a single vector.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_labelClassifiers is null || _labelClassifiers.Length == 0)
        {
            return new Vector<T>(0);
        }

        var allGradients = new List<T>();
        foreach (var classifier in _labelClassifiers)
        {
            var gradients = classifier.ComputeGradients(input, target, lossFunction);
            for (int i = 0; i < gradients.Length; i++)
            {
                allGradients.Add(gradients[i]);
            }
        }

        return new Vector<T>(allGradients.ToArray());
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">The gradients to apply.</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This distributes the gradient updates to each label classifier.
    /// </para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        if (_labelClassifiers is null || _labelClassifiers.Length == 0)
        {
            return;
        }

        int gradIndex = 0;
        foreach (var classifier in _labelClassifiers)
        {
            var classifierParams = classifier.GetParameters();
            var classifierGradients = new Vector<T>(classifierParams.Length);

            for (int i = 0; i < classifierParams.Length && gradIndex < gradients.Length; i++)
            {
                classifierGradients[i] = gradients[gradIndex++];
            }

            classifier.ApplyGradients(classifierGradients, learningRate);
        }
    }

    /// <summary>
    /// Creates a new instance of this classifier with default configuration.
    /// </summary>
    /// <returns>A new BinaryRelevance instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is used internally for operations like cloning or serialization.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new BinaryRelevance<T>(_classifierFactory, Options, Regularization);
    }

    /// <summary>
    /// Creates a deep copy of this classifier.
    /// </summary>
    /// <returns>A new instance with the same parameters and state.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cloning creates an independent copy of the classifier,
    /// including all its internal label classifiers.
    /// </para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new BinaryRelevance<T>(_classifierFactory, Options, Regularization);
        clone.NumLabels = NumLabels;
        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;

        if (_labelClassifiers is not null)
        {
            clone._labelClassifiers = new IClassifier<T>[_labelClassifiers.Length];
            for (int i = 0; i < _labelClassifiers.Length; i++)
            {
                clone._labelClassifiers[i] = (IClassifier<T>)_labelClassifiers[i].Clone();
            }
        }

        return clone;
    }

    #endregion
}
