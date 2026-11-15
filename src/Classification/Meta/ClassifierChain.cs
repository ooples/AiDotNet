using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Classifier Chain for multi-label classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Classifier Chain transforms a multi-label problem into a chain of binary
/// classification problems, where each classifier uses the predictions of
/// previous classifiers as additional features.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Classifier Chain captures label dependencies:
///
/// For labels A, B, C:
/// - Classifier 1: Predict A using features X
/// - Classifier 2: Predict B using features X + prediction of A
/// - Classifier 3: Predict C using features X + predictions of A and B
///
/// Benefits:
/// - Captures dependencies between labels
/// - Better than independent binary classifiers
///
/// Trade-offs:
/// - Order of chain matters (can use random order or learned order)
/// - Error propagation (early mistakes affect later predictions)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create classifier chain for multi-label classification with dependencies
/// var options = new ClassifierChainOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train chained classifiers where each uses previous outputs as features
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ClassifierChain&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict using chained label dependencies
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Classifier)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.MultiClassClassification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Classifier Chains for Multi-label Classification", "https://doi.org/10.1007/s10994-011-5256-5")]
public partial class ClassifierChain<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator.
    /// </summary>
    public ClassifierChain()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// Gets the chain-specific options.
    /// </summary>
    protected new ClassifierChainOptions<T> Options => (ClassifierChainOptions<T>)base.Options;

    /// <summary>
    /// The classifiers in the chain, one per label.
    /// </summary>
    private IClassifier<T>[]? _classifiers;

    /// <summary>
    /// The order of labels in the chain.
    /// </summary>
    private int[]? _order;

    /// <summary>
    /// Gets the number of labels (same as number of classes for multi-label).
    /// </summary>
    public int NumLabels => NumClasses;

    /// <summary>
    /// Gets or sets the label names if available.
    /// </summary>
    public string[]? LabelNames { get; set; }

    /// <summary>
    /// Initializes a new instance of the ClassifierChain class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ClassifierChain(
        Func<IClassifier<T>> estimatorFactory,
        ClassifierChainOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new ClassifierChainOptions<T>(), estimatorFactory, regularization)
    {
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

    /// <summary>
    /// Trains the Classifier Chain on multi-label data.
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

        // Only set class labels to 0..N-1 indices if not already set by Train()
        // This preserves caller labels when Train(x, y) calls TrainMultiLabel
        if (ClassLabels is null || ClassLabels.Length != NumClasses)
        {
            ClassLabels = NumClasses == 0 ? new Vector<T>(0) : Engine.TensorArange<T>(0, NumClasses, 1).ToVector();
        }

        // Determine chain order
        _order = DetermineOrder();
        _classifiers = new IClassifier<T>[NumClasses];
        var labelColumns = new Vector<T>[NumClasses];
        for (int c = 0; c < NumClasses; c++)
        {
            labelColumns[c] = yMultiLabel.GetColumn(c);
        }

        // Train classifiers in chain order
        var previous = new List<Vector<T>>(NumClasses);
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];

            // Augmented features: original features + previous labels in chain order
            var xAugmented = AugmentWithLabels(x, previous);

            // Train classifier
            _classifiers[labelIdx] = CreateBaseEstimator();
            _classifiers[labelIdx].Train(xAugmented, labelColumns[labelIdx]);
            previous.Add(labelColumns[labelIdx]);
        }
    }

    /// <summary>
    /// Standard training method - converts single labels to multi-label format.
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

        // Convert to multi-label format (one-hot encoding)
        TrainMultiLabel(x, ClassMembership(y).ToMatrix());
    }

    /// <summary>
    /// Determines the order of labels in the chain.
    /// </summary>
    private int[] DetermineOrder()
    {
        var order = new int[NumClasses];

        if (Options.Order is not null && Options.Order.Length == NumClasses)
        {
            // Use specified order
            Array.Copy(Options.Order, order, NumClasses);
        }
        else if (Options.RandomOrder)
        {
            // Random order
            for (int i = 0; i < NumClasses; i++)
            {
                order[i] = i;
            }

            var random = Options.Seed.HasValue
                ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
                : RandomHelper.CreateSeededRandom(42);

            for (int i = NumClasses - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
        }
        else
        {
            // Default order (0, 1, 2, ...)
            for (int i = 0; i < NumClasses; i++)
            {
                order[i] = i;
            }
        }

        return order;
    }

    /// <summary>
    /// Creates augmented features by adding previous predictions.
    /// </summary>
    private Matrix<T> AugmentWithLabels(Matrix<T> x, IReadOnlyList<Vector<T>> previous)
    {
        var features = LeadingColumns(x, NumFeatures);
        if (previous.Count == 0)
        {
            return features;
        }

        var parts = new Tensor<T>[previous.Count + 1];
        parts[0] = Tensor<T>.FromMatrix(features);
        for (int p = 0; p < previous.Count; p++)
        {
            parts[p + 1] = previous[p].AsTensor().Reshape(x.Rows, 1);
        }

        return Engine.TensorConcatenate(parts, 1).ToMatrix();
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        // For single-label prediction, return the class with highest probability
        var probs = PredictProbabilities(input);
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels has not been initialized.");
        }

        return input.Rows == 0 ? new Vector<T>(0) : ClassIndicesToLabels(ArgMaxRows(probs));
    }

    /// <summary>
    /// Predicts multi-label output for the given input.
    /// </summary>
    /// <param name="input">The input features matrix.</param>
    /// <returns>A matrix of binary predictions for each label.</returns>
    public Matrix<T> PredictMultiLabel(Matrix<T> input)
    {
        if (_classifiers is null || _order is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var half = NumOps.FromDouble(0.5);
        var predictions = Enumerable.Repeat(new Vector<T>(input.Rows), NumClasses).ToArray();
        var previous = new List<Vector<T>>(NumClasses);

        // Predict in chain order, feeding previous predictions as features
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];
            var labelPreds = _classifiers[labelIdx].Predict(AugmentWithLabels(input, previous));

            // Threshold at 0.5 for binary
            predictions[labelIdx] = AtLeast(labelPreds, half);
            previous.Add(predictions[labelIdx]);
        }

        return NumClasses == 0 ? new Matrix<T>(input.Rows, 0) : StackColumns(predictions);
    }

    /// <inheritdoc/>
    public Matrix<T> PredictMultiLabelProbabilities(Matrix<T> input)
    {
        if (_classifiers is null || _order is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var half = NumOps.FromDouble(0.5);
        var probs = Enumerable.Repeat(new Vector<T>(input.Rows), NumClasses).ToArray();
        var previous = new List<Vector<T>>(NumClasses);

        // Predict in chain order, feeding previous thresholded predictions as features
        for (int i = 0; i < NumClasses; i++)
        {
            int labelIdx = _order[i];
            var xAugmented = AugmentWithLabels(input, previous);

            Vector<T> labelPrediction;
            if (_classifiers[labelIdx] is IProbabilisticClassifier<T> probClassifier)
            {
                // Get probabilities if available
                var labelProbs = probClassifier.PredictProbabilities(xAugmented);
                probs[labelIdx] = labelProbs.GetColumn(labelProbs.Columns > 1 ? 1 : 0);
                labelPrediction = AtLeast(probs[labelIdx], half);
            }
            else
            {
                labelPrediction = AtLeast(_classifiers[labelIdx].Predict(xAugmented), half);
                probs[labelIdx] = labelPrediction;
            }

            previous.Add(labelPrediction);
        }

        return NumClasses == 0 ? new Matrix<T>(input.Rows, 0) : StackColumns(probs);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        return PredictMultiLabelProbabilities(input);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        // log(max(p, 1e-15))
        return base.PredictLogProbabilities(input);
    }
}

/// <summary>
/// Configuration options for Classifier Chain.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class ClassifierChainOptions<T> : MetaClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the explicit order of labels in the chain.
    /// </summary>
    /// <value>Array of label indices defining chain order.</value>
    public int[]? Order { get; set; }

    /// <summary>
    /// Gets or sets whether to use random chain order.
    /// </summary>
    /// <value>True for random order. Default is false.</value>
    public bool RandomOrder { get; set; } = false;

}
