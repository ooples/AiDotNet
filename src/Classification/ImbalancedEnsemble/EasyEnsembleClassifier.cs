using System.Globalization;
using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.ImbalancedEnsemble;

/// <summary>
/// Easy Ensemble Classifier for extremely imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Easy Ensemble is designed for highly imbalanced data. It creates
/// multiple balanced subsets by undersampling the majority class, trains a base classifier
/// (typically AdaBoost) on each subset, and combines their predictions.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Keep all minority class samples</item>
/// <item>For each subset: randomly undersample majority to match minority</item>
/// <item>Train an AdaBoost classifier on each balanced subset</item>
/// <item>Combine predictions using majority voting or averaging</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item><b>Handles extreme imbalance:</b> Works well even with 100:1 or 1000:1 ratios</item>
/// <item><b>Uses all majority data:</b> Through multiple subsets, all majority samples contribute</item>
/// <item><b>Reduces variance:</b> Ensemble approach gives stable predictions</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Extremely imbalanced datasets (more than 10:1 ratio)</item>
/// <item>When you need high recall for the minority class</item>
/// <item>When losing some majority class accuracy is acceptable</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Liu, X.Y., Wu, J., &amp; Zhou, Z.H. (2009). "Exploratory Undersampling for Class-Imbalance Learning"</item>
/// </list>
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for easy ensemble classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new EasyEnsembleClassifier&lt;double&gt;());
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Exploratory Undersampling for Class-Imbalance Learning", "https://doi.org/10.1109/TSMCB.2008.2007853", Year = 2009, Authors = "Xu-Ying Liu, Jianxin Wu, Zhi-Hua Zhou")]
public partial class EasyEnsembleClassifier<T> : ClassifierBase<T>
{

    // Returned _subClassifiers.Count. Same as its two siblings.
    /// <summary>
    /// The ensemble of AdaBoost classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each AdaBoost classifier is trained on a different balanced
    /// subset. Their combined predictions give the final result.</para>
    /// </remarks>
    private readonly List<AdaBoostSubClassifier> _subClassifiers;

    /// <summary>
    /// Number of subset classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More subsets means more diversity and potentially better
    /// results, but also longer training time. 10-50 is typical.</para>
    /// </remarks>
    private readonly int _nSubsets;

    /// <summary>
    /// Number of AdaBoost estimators per subset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each subset classifier is an AdaBoost with this many
    /// weak learners. More estimators = stronger but slower classifiers.</para>
    /// </remarks>
    private readonly int _nEstimatorsPerSubset;

    /// <summary>
    /// Maximum depth for weak learners in AdaBoost.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shallow trees (depth 1-3) are typical for AdaBoost.
    /// Depth 1 ("decision stumps") is common and works well.</para>
    /// </remarks>
    private readonly int _maxDepth;

    /// <summary>
    /// Learning rate for AdaBoost.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller values require more estimators but can give better
    /// results. 1.0 is standard, 0.1-0.5 for more conservative boosting.</para>
    /// </remarks>
    private readonly double _learningRate;

    /// <summary>
    /// Sampling strategy for undersampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how majority class is undersampled:
    /// "auto" matches minority count, "ratio:2" uses 2x minority count, etc.</para>
    /// </remarks>
    private readonly string _samplingStrategy;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether to use soft voting (probability averaging) vs hard voting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Soft voting averages probabilities from all classifiers.
    /// Hard voting counts class votes. Soft voting often gives better results.</para>
    /// </remarks>
    private bool _softVoting;

    /// <summary>
    /// Initializes a new instance of EasyEnsembleClassifier.
    /// </summary>
    /// <param name="nSubsets">Number of balanced subsets. Default is 10.</param>
    /// <param name="nEstimatorsPerSubset">AdaBoost estimators per subset. Default is 50.</param>
    /// <param name="maxDepth">Max depth of weak learners. Default is 1 (stumps).</param>
    /// <param name="learningRate">AdaBoost learning rate. Default is 1.0.</param>
    /// <param name="samplingStrategy">Undersampling strategy. Default is "auto".</param>
    /// <param name="softVoting">Use soft voting. Default is true.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Default parameters work well for most imbalanced cases:
    /// <list type="bullet">
    /// <item>Increase nSubsets for very imbalanced data (uses more majority samples)</item>
    /// <item>Increase nEstimatorsPerSubset for complex decision boundaries</item>
    /// <item>Reduce learningRate if overfitting (with more estimators)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public EasyEnsembleClassifier(
        int nSubsets = 10,
        int nEstimatorsPerSubset = 50,
        int maxDepth = 1,
        double learningRate = 1.0,
        string samplingStrategy = "auto",
        bool softVoting = true,
        int? seed = null)
        : base()
    {
        if (nSubsets < 1)
            throw new ArgumentOutOfRangeException(nameof(nSubsets), "nSubsets must be at least 1.");
        if (nEstimatorsPerSubset < 1)
            throw new ArgumentOutOfRangeException(nameof(nEstimatorsPerSubset), "nEstimatorsPerSubset must be at least 1.");
        if (maxDepth < 1)
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "maxDepth must be at least 1.");
        if (learningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "learningRate must be greater than 0.");
        if (string.IsNullOrWhiteSpace(samplingStrategy))
            throw new ArgumentException("samplingStrategy cannot be null or whitespace.", nameof(samplingStrategy));

        _nSubsets = nSubsets;
        _nEstimatorsPerSubset = nEstimatorsPerSubset;
        _maxDepth = maxDepth;
        _learningRate = learningRate;
        _samplingStrategy = samplingStrategy;
        _softVoting = softVoting;
        _subClassifiers = [];
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.EasyEnsembleClassifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system track what type of model this is.</para>
    /// </remarks>

    /// <summary>
    /// Trains the Easy Ensemble classifier.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training creates multiple balanced subsets and trains an
    /// AdaBoost classifier on each. This ensures all majority samples contribute across subsets.</para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        if (NumClasses != 2)
        {
            throw new ArgumentException(
                "EasyEnsembleClassifier only supports binary classification. " +
                $"Found {NumClasses} classes. For multi-class problems, consider One-vs-Rest strategies.");
        }

        _subClassifiers.Clear();

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int c = 0; c < NumClasses; c++)
        {
            classSamples[c] = [];
        }

        var sampleClasses = GetClassIndicesFromLabels(y);
        for (int i = 0; i < sampleClasses.Length; i++)
        {
            int classIdx = sampleClasses[i];
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classSamples[classIdx].Add(i);
            }
        }

        // Find minority class size and majority class
        int minoritySize = classSamples.Values.Min(v => v.Count);
        if (minoritySize == 0)
        {
            throw new ArgumentException(
                "One or more classes have no samples. Cannot train on empty class data.",
                nameof(y));
        }
        int minorityClass = classSamples.First(kv => kv.Value.Count == minoritySize).Key;
        int majorityClass = NumClasses == 2 ? 1 - minorityClass :
            classSamples.First(kv => kv.Key != minorityClass).Key;

        // Train subclassifiers
        for (int s = 0; s < _nSubsets; s++)
        {
            // Create balanced subset
            var subsetIndices = CreateBalancedSubset(classSamples, minoritySize, minorityClass);

            // Extract subset data
            var (subX, subY) = SelectSamples(x, y, subsetIndices);
            if (subX.Columns != NumFeatures)
            {
                subX = subX.GetSubMatrix(0, 0, subX.Rows, NumFeatures);
            }

            // Train AdaBoost on this subset
            var adaBoost = TrainAdaBoost(subX, subY);
            _subClassifiers.Add(adaBoost);
        }
    }

    /// <summary>
    /// Creates a balanced subset by undersampling majority class.
    /// </summary>
    /// <param name="classSamples">Samples grouped by class.</param>
    /// <param name="samplesPerClass">Target samples per class.</param>
    /// <param name="minorityClass">Index of minority class.</param>
    /// <returns>Indices of selected samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps all minority samples and randomly selects an equal
    /// number from each majority class. This creates a balanced training set.</para>
    /// </remarks>
    private int[] CreateBalancedSubset(Dictionary<int, List<int>> classSamples,
        int samplesPerClass, int minorityClass)
    {
        var selected = new List<int>();

        // Determine target samples based on sampling strategy
        int targetSamples = samplesPerClass;
        if (_samplingStrategy.StartsWith("ratio:", StringComparison.OrdinalIgnoreCase))
        {
            if (double.TryParse(_samplingStrategy.Substring(6), NumberStyles.Float, CultureInfo.InvariantCulture, out double ratio) && ratio > 0)
            {
                targetSamples = (int)(samplesPerClass * ratio);
            }
        }
        // "auto" or unrecognized strategy uses samplesPerClass as-is (minority class size)

        // Guard against ratio producing zero majority samples
        if (targetSamples < 1)
        {
            throw new ArgumentException(
                "samplingStrategy results in zero samples for the majority class. " +
                "Use a ratio that yields at least one sample.");
        }

        foreach (var (classIdx, samples) in classSamples)
        {
            if (classIdx == minorityClass)
            {
                // Use all minority samples
                selected.AddRange(samples);
            }
            else
            {
                // Undersample majority class using the computed target
                var shuffled = samples.OrderBy(_ => _random.Next()).Take(targetSamples).ToList();
                selected.AddRange(shuffled);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Trains an AdaBoost classifier.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Labels.</param>
    /// <returns>Trained AdaBoost classifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> AdaBoost works by:
    /// <list type="number">
    /// <item>Start with equal weights for all samples</item>
    /// <item>Train a weak learner (shallow tree)</item>
    /// <item>Increase weights on misclassified samples</item>
    /// <item>Repeat, building a weighted ensemble</item>
    /// </list>
    /// This focuses later learners on hard-to-classify samples.</para>
    /// </remarks>
    private AdaBoostSubClassifier TrainAdaBoost(Matrix<T> x, Vector<T> y)
    {
        var engine = Engine;
        int n = x.Rows;
        var features = Tensor<T>.FromMatrix(x);
        var weights = FilledVector(n, NumOps.FromDouble(1.0 / n)).AsTensor();

        // 0/1 membership of class 0 and class 1 for every sample
        var classValues = ClassIndicesToValues(GetClassIndexTensor(y), NumClasses).AsTensor();
        var isOne = engine.TensorEquals(classValues, NumOps.One);
        var isZero = engine.TensorEquals(classValues, NumOps.Zero);

        var weakLearners = new List<WeakLearner>();
        var alphas = new List<Tensor<T>>();
        T halfLearningRate = NumOps.Multiply(NumOps.FromDouble(_learningRate), NumOps.FromDouble(0.5));
        T minError = NumOps.FromDouble(1e-10);
        T maxError = NumOps.FromDouble(1 - 1e-10);
        T two = NumOps.FromDouble(2.0);

        for (int t = 0; t < _nEstimatorsPerSubset; t++)
        {
            var learner = TrainWeakLearner(features, isOne, isZero, weights);

            // A sample is misclassified unless (prediction 1 and class 1) or (prediction 0 and class 0)
            var predictions = PredictWeakLearner(learner, features);
            var correct = engine.TensorAdd(engine.TensorMultiply(predictions, isOne),
                engine.TensorMultiply(engine.ScalarMinusTensor(NumOps.One, predictions), isZero));
            var misclassified = engine.ScalarMinusTensor(NumOps.One, correct);

            // [1] weighted error clamped to [1e-10, 1 - 1e-10]; alpha = lr * 0.5 * log((1 - err) / err)
            var weightedError = engine.TensorClamp(
                engine.ReduceSum(engine.TensorMultiply(weights, misclassified), new[] { 0 }, true), minError, maxError);
            var alpha = engine.TensorMultiplyScalar(
                engine.TensorLog(engine.TensorDivide(engine.ScalarMinusTensor(NumOps.One, weightedError), weightedError)),
                halfLearningRate);

            // w *= exp(alpha * sign) with sign = +1 for misclassified and -1 otherwise, then normalize
            var sign = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(misclassified, two), NumOps.One);
            weights = engine.TensorMultiply(weights, engine.TensorExp(engine.TensorMultiply(sign, engine.TensorBroadcastTo(alpha, new[] { n }))));
            weights = engine.TensorDivideScalar(weights, engine.TensorSum(weights));

            weakLearners.Add(learner);
            alphas.Add(alpha);
        }

        return new AdaBoostSubClassifier
        {
            WeakLearners = weakLearners,
            Alphas = alphas.Count == 0
                ? new List<T>()
                : new List<T>(engine.TensorConcatenate(alphas.ToArray(), 0).ToArray())
        };
    }

    /// <summary>
    /// Trains the decision stump with the lowest weighted error.
    /// </summary>
    /// <param name="features">The [n, features] training features.</param>
    /// <param name="isOne">1 for samples of class 1, else 0.</param>
    /// <param name="isZero">1 for samples of class 0, else 0.</param>
    /// <param name="weights">Sample weights.</param>
    /// <returns>Trained weak learner.</returns>
    /// <remarks>
    /// A stump with polarity p and threshold t predicts 1 when p * (x - t) &gt; 0. Thresholds are the
    /// midpoints between consecutive distinct values; features, thresholds and then polarities -1, +1
    /// are tried in increasing order and the first strictly lowest error wins.
    /// </remarks>
    private WeakLearner TrainWeakLearner(Tensor<T> features, Tensor<T> isOne, Tensor<T> isZero, Tensor<T> weights)
    {
        var engine = Engine;
        int n = features.Shape[0];
        var oneWeights = engine.TensorMultiply(isOne, weights);
        var zeroWeights = engine.TensorMultiply(isZero, weights);
        T oneTotal = n == 0 ? NumOps.Zero : engine.TensorSum(oneWeights);
        T zeroTotal = n == 0 ? NumOps.Zero : engine.TensorSum(zeroWeights);

        // Samples of any other class are misclassified by every stump
        T otherTotal = n == 0 ? NumOps.Zero : NumOps.Subtract(NumOps.Subtract(engine.TensorSum(weights), oneTotal), zeroTotal);

        var featureErrors = new List<Tensor<T>>();
        var featureThresholds = new List<Tensor<T>>();
        var featureIds = new List<int>();
        var featurePolarities = new List<int>();
        T two = NumOps.FromDouble(2.0);

        for (int j = 0; j < NumFeatures && n > 0; j++)
        {
            var column = features.Slice(1, j, j + 1).Reshape(n);
            var (sortedValues, order) = engine.TensorSort(column, 0, false);
            var distinct = engine.TensorUniqueConsecutive(sortedValues);
            int numDistinct = distinct.Length;
            if (numDistinct < 2) continue;

            // Weight of class 1 / class 0 samples with value <= distinct[v]
            var lastLeft = engine.TensorAddScalar(
                engine.TensorSearchSorted(sortedValues, distinct.Slice(0, 0, numDistinct - 1), true), -1);
            var oneLeft = engine.TensorGather(engine.TensorCumSum(engine.TensorGather(oneWeights, order, 0), 0), lastLeft, 0);
            var zeroLeft = engine.TensorGather(engine.TensorCumSum(engine.TensorGather(zeroWeights, order, 0), 0), lastLeft, 0);

            // Polarity -1 predicts 1 for x < t (= x <= distinct[v]); polarity +1 predicts 1 for x > t
            var negativeError = engine.TensorAddScalar(
                engine.TensorAdd(zeroLeft, engine.ScalarMinusTensor(oneTotal, oneLeft)), otherTotal);
            var positiveError = engine.TensorAddScalar(
                engine.TensorAdd(oneLeft, engine.ScalarMinusTensor(zeroTotal, zeroLeft)), otherTotal);
            var errors = engine.TensorConcatenate(new[]
            {
                negativeError.Reshape(numDistinct - 1, 1),
                positiveError.Reshape(numDistinct - 1, 1)
            }, 1).Reshape(2 * (numDistinct - 1));

            int best = engine.TensorArgMin(errors, 0).ToArray()[0];
            int v = best / 2;
            featureErrors.Add(engine.TensorSlice(errors, new[] { best }, new[] { 1 }));
            featureThresholds.Add(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorSlice(distinct, new[] { v }, new[] { 2 }), new[] { 0 }, true), two));
            featureIds.Add(j);
            featurePolarities.Add(best % 2 == 0 ? -1 : 1);
        }

        if (featureErrors.Count == 0)
            throw new InvalidOperationException(
                "No valid split found: all features appear to be constant.");

        // The first feature with the strictly lowest error wins
        int bestFeature = engine.TensorArgMin(engine.TensorConcatenate(featureErrors.ToArray(), 0), 0).ToArray()[0];
        return new WeakLearner
        {
            FeatureIndex = featureIds[bestFeature],
            Threshold = featureThresholds[bestFeature][0],
            Polarity = featurePolarities[bestFeature]
        };
    }

    /// <summary>
    /// Predicts with a weak learner: 1 where polarity * (x - threshold) &gt; 0, else 0.
    /// </summary>
    /// <param name="learner">The weak learner.</param>
    /// <param name="features">The [n, features] features.</param>
    /// <returns>The 0/1 prediction of every row.</returns>
    private Tensor<T> PredictWeakLearner(WeakLearner learner, Tensor<T> features)
    {
        var engine = Engine;
        int n = features.Shape[0];
        var column = features.Slice(1, learner.FeatureIndex, learner.FeatureIndex + 1).Reshape(n);
        var margin = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(column, learner.Threshold),
            NumOps.FromDouble(learner.Polarity));
        return engine.TensorGreaterThan(margin, NumOps.Zero);
    }

    /// <summary>
    /// Predicts class labels using the ensemble.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each AdaBoost subclassifier votes (hard voting) or contributes
    /// its probability (soft voting); class 1 wins only with strictly more support than class 0.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null || _subClassifiers.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var engine = Engine;
        int rows = input.Rows;
        if (rows == 0)
        {
            return new Vector<T>(0);
        }

        var features = Tensor<T>.FromMatrix(input);
        var classOne = new Tensor<T>(new[] { rows });
        var classZero = new Tensor<T>(new[] { rows });
        T two = NumOps.FromDouble(2.0);

        foreach (var subClassifier in _subClassifiers)
        {
            var score = PredictAdaBoostScore(subClassifier, features);
            Tensor<T> support;
            if (_softVoting)
            {
                // Average probabilities: P(class 1) = 1 / (1 + exp(-2 * score))
                support = engine.TensorSigmoid(engine.TensorMultiplyScalar(score, two));
            }
            else
            {
                // Hard voting: class 1 when score >= 0
                support = engine.TensorGreaterEqual(score, new Tensor<T>(new[] { rows }));
            }

            classOne = engine.TensorAdd(classOne, support);
            classZero = engine.TensorAdd(classZero, engine.ScalarMinusTensor(NumOps.One, support));
        }

        return ClassIndicesToLabels(ZeroOneToInt(engine.TensorGreaterThan(classOne, classZero)));
    }

    /// <summary>
    /// Computes the AdaBoost score of every row.
    /// </summary>
    /// <param name="classifier">AdaBoost classifier.</param>
    /// <param name="features">The [n, features] features.</param>
    /// <returns>sum_t alpha_t * (+1 for prediction 1, -1 for prediction 0) per row.</returns>
    private Tensor<T> PredictAdaBoostScore(AdaBoostSubClassifier classifier, Tensor<T> features)
    {
        var engine = Engine;
        var score = new Tensor<T>(new[] { features.Shape[0] });
        T two = NumOps.FromDouble(2.0);
        for (int t = 0; t < classifier.WeakLearners.Count; t++)
        {
            var sign = engine.TensorSubtractScalar(
                engine.TensorMultiplyScalar(PredictWeakLearner(classifier.WeakLearners[t], features), two),
                NumOps.One);
            score = engine.TensorAdd(score, engine.TensorMultiplyScalar(sign, classifier.Alphas[t]));
        }

        return score;
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters (limited support).</param>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new untrained model with same hyperparameters.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new EasyEnsembleClassifier<T>(_nSubsets, _nEstimatorsPerSubset, _maxDepth,
            _learningRate, _samplingStrategy, _softVoting);
    }

    /// <summary>
    /// Gets feature importance based on weak learner usage.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features used in more weak learners with higher alpha
    /// weights are considered more important. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        // Sum of |alpha| of the weak learners splitting on each feature, normalized to sum to 1
        var engine = Engine;
        var featureIndices = new List<int>();
        var weights = new List<T>();
        foreach (var subClassifier in _subClassifiers)
        {
            for (int t = 0; t < subClassifier.WeakLearners.Count; t++)
            {
                featureIndices.Add(subClassifier.WeakLearners[t].FeatureIndex);
                weights.Add(subClassifier.Alphas[t]);
            }
        }

        var importance = new Tensor<T>(new[] { NumFeatures });
        if (featureIndices.Count > 0 && NumFeatures > 0)
        {
            importance = engine.TensorScatterAdd(importance,
                new Tensor<int>(featureIndices.ToArray(), new[] { featureIndices.Count }),
                engine.TensorAbs(Vector<T>.Wrap(weights.ToArray()).AsTensor()), 0);
        }

        T total = NumFeatures == 0 ? NumOps.Zero : engine.TensorSum(importance);
        if (NumOps.Equals(total, NumOps.Zero)) total = NumOps.One;
        var values = engine.TensorDivideScalar(importance, total).ToArray();

        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = values[i];
        }

        return result;
    }

    /// <summary>
    /// Represents an AdaBoost sub-classifier.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Contains the weak learners and their weights (alphas)
    /// that make up one AdaBoost classifier.</para>
    /// </remarks>
    private class AdaBoostSubClassifier
    {
        /// <summary>
        /// List of weak learners (decision stumps).
        /// </summary>
        public List<WeakLearner> WeakLearners { get; set; } = [];

        /// <summary>
        /// Weights for each weak learner.
        /// </summary>
        public List<T> Alphas { get; set; } = [];
    }

    /// <summary>
    /// Represents a weak learner (decision stump).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A decision stump splits on a single feature at a threshold.
    /// Polarity determines which side predicts which class.</para>
    /// </remarks>
    private class WeakLearner
    {
        /// <summary>
        /// Feature index for the split.
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Threshold value for the split.
        /// </summary>
        public T Threshold { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

        /// <summary>
        /// Polarity: +1 means values above threshold predict class 1,
        /// -1 means values below threshold predict class 1.
        /// </summary>
        public int Polarity { get; set; } = 1;
    }
}
