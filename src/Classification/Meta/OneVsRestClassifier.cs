using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// One-vs-Rest (also called One-vs-All) classifier for multi-class and multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Trains one binary classifier per class, treating it as the positive class
/// and all other classes as the negative class. This approach works for both
/// multi-class (exactly one label per sample) and multi-label (zero or more labels per sample) problems.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// One-vs-Rest is a simple strategy for multi-class and multi-label classification:
///
/// For 3 classes (A, B, C):
/// - Classifier 1: Is it A vs not-A?
/// - Classifier 2: Is it B vs not-B?
/// - Classifier 3: Is it C vs not-C?
///
/// For prediction, the class whose classifier gives the highest score wins.
///
/// Advantages:
/// - Simple and effective
/// - Trains K classifiers for K classes
/// - Easily parallelizable
///
/// Disadvantages:
/// - Class imbalance (one class vs all others)
/// - Classifiers don't see inter-class relationships
/// </para>
/// <para>
/// <b>Multi-label usage:</b>
/// This classifier also supports multi-label classification via
/// <see cref="TrainMultiLabel(Matrix{T}, Matrix{T})"/>, along with the
/// <see cref="NumLabels"/> and <see cref="LabelNames"/> properties.
/// When using this class in a multi-label setting, prefer
/// <see cref="TrainMultiLabel(Matrix{T}, Matrix{T})"/> over
/// <see cref="Train(Matrix{T}, Vector{T})"/>.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for onevsrestclassifier classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OneVsRestClassifier&lt;double&gt;(() => new GaussianNaiveBayes&lt;double&gt;()));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("In Defense of One-Vs-All Classification", "https://jmlr.org/papers/v5/rifkin04a.html")]
public partial class OneVsRestClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator.
    /// </summary>
    public OneVsRestClassifier()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// The binary classifiers, one per class.
    /// </summary>
    private IClassifier<T>[]? _estimators;

    /// <summary>
    /// Gets the number of labels (same as number of classes for multi-label).
    /// </summary>
    public int NumLabels => NumClasses;

    /// <summary>
    /// Gets or sets the label names if available.
    /// </summary>
    public string[]? LabelNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the OneVsRestClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public OneVsRestClassifier(
        Func<IClassifier<T>> estimatorFactory,
        MetaClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the One-vs-Rest classifier on multi-label data.
    /// </summary>
    /// <param name="x">The input features matrix.</param>
    /// <param name="yMultiLabel">The multi-label target matrix (rows=samples, cols=labels).</param>
    public void TrainMultiLabel(Matrix<T> x, Matrix<T> yMultiLabel)
    {
        if (x.Rows != yMultiLabel.Rows)
        {
            throw new ArgumentException("Number of samples in X must match number of rows in Y.");
        }

        NumFeatures = x.Columns;
        NumClasses = yMultiLabel.Columns;
        TaskType = ClassificationTaskType.MultiLabel;

        // Create class labels (0 to NumClasses-1)
        ClassLabels = NumClasses == 0 ? new Vector<T>(0) : Engine.TensorArange<T>(0, NumClasses, 1).ToVector();

        // Train one binary classifier per label
        _estimators = new IClassifier<T>[NumClasses];

        for (int c = 0; c < NumClasses; c++)
        {
            _estimators[c] = CreateBaseEstimator();
            _estimators[c].Train(x, yMultiLabel.GetColumn(c));
        }
    }

    /// <summary>
    /// Trains the One-vs-Rest classifier on the provided data.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        // Train one binary classifier per class; binary labels are 1 for this class, 0 for all others
        _estimators = new IClassifier<T>[NumClasses];
        var membership = ClassMembership(y).ToMatrix();

        for (int c = 0; c < NumClasses; c++)
        {
            _estimators[c] = CreateBaseEstimator();
            _estimators[c].Train(x, membership.GetColumn(c));
        }
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var probs = PredictProbabilities(input);
        return input.Rows == 0 ? new Vector<T>(0) : ClassIndicesToLabels(ArgMaxRows(probs));
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Decision scores from each binary classifier, normalized to probabilities using softmax
        return ApplySoftmax(GetAllEstimatorScores(input));
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        // log(max(p, 1e-15))
        return base.PredictLogProbabilities(input);
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // For multi-label, each classifier's prediction is independent; threshold at 0.5
        var scores = GetAllEstimatorScores(input);
        if (scores.Rows == 0 || scores.Columns == 0)
        {
            return scores;
        }

        var engine = Engine;
        var scoreTensor = Tensor<T>.FromMatrix(scores);
        var half = new Tensor<T>(new[] { scores.Rows, scores.Columns });
        engine.TensorFill(half, NumOps.FromDouble(0.5));
        return engine.TensorGreaterEqual(scoreTensor, half).ToMatrix();
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_estimators is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        return GetAllEstimatorScores(input);
    }

    /// <summary>
    /// Stacks every estimator's scores into an [input.Rows, NumClasses] matrix.
    /// </summary>
    private Matrix<T> GetAllEstimatorScores(Matrix<T> input)
    {
        if (_estimators is null || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        var columns = new Vector<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            columns[c] = GetEstimatorScores(_estimators[c], input);
        }

        return StackColumns(columns);
    }

    /// <summary>
    /// Gets decision scores from an estimator.
    /// </summary>
    private Vector<T> GetEstimatorScores(IClassifier<T> estimator, Matrix<T> input)
    {
        // Try to get probabilities if available
        if (estimator is IProbabilisticClassifier<T> probClassifier)
        {
            // Get probability of positive class (class 1)
            var probs = probClassifier.PredictProbabilities(input);
            return probs.GetColumn(probs.Columns > 1 ? 1 : 0);
        }

        // Fall back to binary predictions
        return estimator.Predict(input);
    }
}
