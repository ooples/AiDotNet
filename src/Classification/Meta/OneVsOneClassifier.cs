using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// One-vs-One classifier for multi-class classification.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Trains one binary classifier for each pair of classes.
/// Uses voting to determine the final prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// One-vs-One trains a classifier for every pair of classes:
///
/// For 3 classes (A, B, C):
/// - Classifier 1: A vs B
/// - Classifier 2: A vs C
/// - Classifier 3: B vs C
///
/// For K classes, this requires K*(K-1)/2 classifiers.
///
/// For prediction, each classifier votes for one class, and
/// the class with the most votes wins.
///
/// Advantages:
/// - Each classifier is trained on balanced binary problems
/// - Works well with SVM and other pairwise classifiers
/// - Good for small to medium number of classes
///
/// Disadvantages:
/// - Requires many classifiers for large K (K*(K-1)/2)
/// - Slower training for many classes
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create One-vs-One classifier with pairwise binary classifiers
/// var options = new MetaClassifierOptions&lt;double&gt;();
///
/// // Prepare multi-class training data (3 classes, 9 samples, 2 features)
/// var features = new Matrix&lt;double&gt;(9, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 3.0; features[3, 1] = 3.1;
/// features[4, 0] = 3.2; features[4, 1] = 2.9;
/// features[5, 0] = 2.8; features[5, 1] = 3.0;
/// features[6, 0] = 5.0; features[6, 1] = 5.1;
/// features[7, 0] = 5.2; features[7, 1] = 4.9;
/// features[8, 0] = 4.8; features[8, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1, 2, 2, 2 });
///
/// // Train K*(K-1)/2 binary classifiers for each class pair
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new OneVsOneClassifier&lt;double&gt;())
///     .Build(features, labels);
///
/// // Predict using pairwise voting
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 3.1; newSample[0, 1] = 3.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Solving Multiclass Learning Problems via Error-Correcting Output Codes", "https://doi.org/10.1613/jair.105", Year = 1995, Authors = "Thomas G. Dietterich, Ghulum Bakiri")]
public partial class OneVsOneClassifier<T> : MetaClassifierBase<T>
{
    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base estimator.
    /// </summary>
    public OneVsOneClassifier()
        : this(() => new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// The binary classifiers for each pair of classes.
    /// </summary>
    private IClassifier<T>[]? _estimators;

    /// <summary>
    /// The class pair indices for each estimator.
    /// </summary>
    private (int, int)[]? _classPairs;

    /// <summary>
    /// Initializes a new instance of the OneVsOneClassifier class.
    /// </summary>
    /// <param name="estimatorFactory">Factory function to create base binary classifiers.</param>
    /// <param name="options">Configuration options for the classifier.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public OneVsOneClassifier(
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
    /// Trains the One-vs-One classifier on the provided data.
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

        // Create list of class pairs
        int numPairs = NumClasses * (NumClasses - 1) / 2;
        _classPairs = new (int, int)[numPairs];
        _estimators = new IClassifier<T>[numPairs];

        int pairIdx = 0;
        var classIndices = GetClassIndicesFromLabels(y);
        for (int i = 0; i < NumClasses - 1; i++)
        {
            for (int j = i + 1; j < NumClasses; j++)
            {
                _classPairs[pairIdx] = (i, j);

                // Get samples for this pair of classes
                var (xPair, yPair) = ExtractPairData(x, y, classIndices, i, j);

                // Train binary classifier
                _estimators[pairIdx] = CreateBaseEstimator();
                _estimators[pairIdx].Train(xPair, yPair);

                pairIdx++;
            }
        }
    }

    /// <summary>
    /// Extracts samples belonging to a pair of classes.
    /// </summary>
    private (Matrix<T> x, Vector<T> y) ExtractPairData(Matrix<T> x, Vector<T> y, int[] classIndices, int classA, int classB)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("ClassLabels has not been initialized.");
        }

        // Samples of either class, in their original order
        var rows = new List<int>();
        for (int i = 0; i < classIndices.Length; i++)
        {
            if (classIndices[i] == classA || classIndices[i] == classB)
            {
                rows.Add(i);
            }
        }

        // classA -> 0, classB -> 1
        var (xPair, yLabels) = SelectSamples(x, y, rows.ToArray());
        var yPair = yLabels.Length == 0
            ? new Vector<T>(0)
            : Engine.TensorEquals(yLabels.AsTensor(), ClassLabels[classB]).ToVector();
        return (LeadingColumns(xPair, NumFeatures), yPair);
    }

    /// <inheritdoc/>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_estimators is null || _classPairs is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        var engine = Engine;
        var samples = LeadingColumns(input, NumFeatures);
        var half = NumOps.FromDouble(0.5);
        var votes = new Tensor<T>(new[] { input.Rows, NumClasses });

        // Each classifier votes for one of its classes: 0 = classA wins, 1 = classB wins
        for (int p = 0; p < _estimators.Length; p++)
        {
            var (classA, classB) = _classPairs[p];
            var votesB = AtLeast(_estimators[p].Predict(samples), half);
            var votesA = engine.ScalarMinusTensor(NumOps.One, votesB.AsTensor()).ToVector();
            votes = engine.TensorScatterAdd(votes, new Tensor<int>(new[] { classA, classB }, new[] { 2 }),
                Tensor<T>.FromMatrix(StackColumns(new[] { votesA, votesB })), 1);
        }

        // Class with most votes (first class wins ties)
        return MajorityVoteLabels(votes);
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_estimators is null || _classPairs is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Rows == 0 || NumClasses == 0)
        {
            return new Matrix<T>(input.Rows, NumClasses);
        }

        var engine = Engine;
        var samples = LeadingColumns(input, NumFeatures);
        var pairwiseSums = new Tensor<T>(new[] { input.Rows, NumClasses });
        var partnerCounts = new int[NumClasses];

        // Accumulate pairwise probabilities: classA gets 1 - P(B), classB gets P(B)
        for (int p = 0; p < _estimators.Length; p++)
        {
            var (classA, classB) = _classPairs[p];
            Vector<T> probB;

            // Get probability from classifier
            if (_estimators[p] is IProbabilisticClassifier<T> probClassifier)
            {
                var pairProbs = probClassifier.PredictProbabilities(samples);
                probB = pairProbs.GetColumn(pairProbs.Columns > 1 ? 1 : 0);
            }
            else
            {
                probB = _estimators[p].Predict(samples);
            }

            var probA = engine.ScalarMinusTensor(NumOps.One, probB.AsTensor()).ToVector();
            pairwiseSums = engine.TensorScatterAdd(pairwiseSums, new Tensor<int>(new[] { classA, classB }, new[] { 2 }),
                Tensor<T>.FromMatrix(StackColumns(new[] { probA, probB })), 1);
            partnerCounts[classA]++;
            partnerCounts[classB]++;
        }

        // Aggregate to class probabilities: mean over partners, or 1/NumClasses for a class without partners
        var counts = CountsToValues(new Tensor<int>(partnerCounts, new[] { NumClasses }));
        var denominators = engine.TensorClampMin(counts, NumOps.One).Reshape(1, NumClasses);
        var noPartner = engine.TensorMultiplyScalar(engine.TensorEquals(counts, NumOps.Zero),
            NumOps.Divide(NumOps.One, NumOps.FromDouble(NumClasses))).Reshape(1, NumClasses);
        var classProbs = engine.TensorAdd(engine.TensorDivide(pairwiseSums, denominators), noPartner);

        // Normalize
        var rowSums = engine.TensorClampMin(engine.ReduceSum(classProbs, new[] { 1 }, true), Tolerance(1e-15));
        return engine.TensorDivide(classProbs, rowSums).ToMatrix();
    }

    /// <inheritdoc/>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        // log(max(p, 1e-15))
        return base.PredictLogProbabilities(input);
    }
}
