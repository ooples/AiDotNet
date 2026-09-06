using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
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
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for binaryrelevance classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BinaryRelevance&lt;double&gt;(() => new GaussianNaiveBayes&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("A Review on Multi-Label Learning Algorithms", "https://doi.org/10.1109/TKDE.2013.39")]
public class BinaryRelevance<T> : MultiLabelClassifierBase<T>
{

    /// <inheritdoc />
    /// <remarks>One independent binary classifier per label, in label order -- the order the hand-written concatenation walked them. Returning early when the array is null keeps registration from latching before training has built them: the base only marks the job done once something was actually registered.</remarks>
    protected override void RegisterComponents()
    {
        if (_labelClassifiers is null) return;
        foreach (var classifier in _labelClassifiers)
        {
            RegisterParameterComponent(classifier);
        }
    }
    #region Fields

    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier.
    /// </summary>
    public BinaryRelevance()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

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
    /// var br2 = new BinaryRelevance&lt;double&gt;(() => new SupportVectorClassifier&lt;double&gt;());
    /// </code>
    /// </para>
    /// </remarks>
    public BinaryRelevance(
        Func<IClassifier<T>> classifierFactory,
        ClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        Guard.NotNull(classifierFactory);
        _classifierFactory = classifierFactory;
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

    #region Serialization

    #endregion

    #region Abstract Method Implementations

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
    public override IFullModel<T, Matrix<T>, Matrix<T>> WithParameters(Vector<T> parameters)
    {
        var newClassifier = new BinaryRelevance<T>(_classifierFactory, Options, Regularization);
        newClassifier.NumLabels = NumLabels;
        newClassifier.NumFeatures = NumFeatures;
        newClassifier.SetParameters(parameters);
        return newClassifier;
    }

    /// <summary>
    /// Computes gradients for gradient-based optimization.
    /// </summary>
    /// <param name="input">The input features.</param>
    /// <param name="target">The target label matrix.</param>
    /// <param name="lossFunction">The loss function (optional).</param>
    /// <returns>A concatenated gradient vector from all label classifiers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes gradients for each label classifier independently
    /// and concatenates them into a single vector.
    /// </para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Matrix<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (_labelClassifiers is null || _labelClassifiers.Length == 0)
        {
            return new Vector<T>(0);
        }

        var allGradients = new List<T>();
        for (int labelIndex = 0; labelIndex < NumLabels && labelIndex < _labelClassifiers.Length; labelIndex++)
        {
            var classifier = _labelClassifiers[labelIndex];

            // Extract binary labels for this label column
            var binaryTarget = new Vector<T>(target.Rows);
            for (int i = 0; i < target.Rows; i++)
            {
                binaryTarget[i] = target[i, labelIndex];
            }

            var gradients = ((IGradientComputable<T, Matrix<T>, Vector<T>>)classifier).ComputeGradients(input, binaryTarget, lossFunction);
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
            var classifierParams = ((IParameterizable<T, Matrix<T>, Vector<T>>)classifier).GetParameters();
            var classifierGradients = new Vector<T>(classifierParams.Length);

            for (int i = 0; i < classifierParams.Length && gradIndex < gradients.Length; i++)
            {
                classifierGradients[i] = gradients[gradIndex++];
            }

            ((IGradientComputable<T, Matrix<T>, Vector<T>>)classifier).ApplyGradients(classifierGradients, learningRate);
        }
    }

    #endregion
}
