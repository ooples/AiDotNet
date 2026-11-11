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
/// Balanced Bagging Classifier for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Bagging (Bootstrap Aggregating) creates diverse classifiers by
/// training each on a random bootstrap sample. Balanced Bagging adds undersampling to ensure
/// each bootstrap is class-balanced, improving minority class detection.</para>
///
/// <para><b>How it works:</b> For each base classifier:
/// <list type="number">
/// <item>Sample minority class with replacement (bootstrap)</item>
/// <item>Undersample majority class to match minority count</item>
/// <item>Train a base classifier (typically decision tree) on balanced bootstrap</item>
/// <item>Combine predictions using majority voting</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item><b>Reduces variance:</b> Multiple diverse classifiers give stable predictions</item>
/// <item><b>Handles imbalance:</b> Each classifier sees balanced data</item>
/// <item><b>Parallelizable:</b> Each base classifier can be trained independently</item>
/// <item><b>Flexible:</b> Works with any base classifier, not just trees</item>
/// </list>
/// </para>
///
/// <para><b>Difference from BalancedRandomForest:</b> BalancedBagging can use any base classifier
/// and each classifier sees all features, while BalancedRandomForest uses trees with random
/// feature subsets at each split.</para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you want ensemble benefits with imbalanced data</item>
/// <item>When you want to use a specific base classifier (not just trees)</item>
/// <item>When variance reduction is more important than bias reduction</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Hido, S., &amp; Kashima, H. (2009). "Roughly Balanced Bagging for Imbalanced Data"</item>
/// <item>Wang, S., &amp; Yao, X. (2009). "Diversity Analysis on Imbalanced Data Sets"</item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// var labels = new Vector&lt;double&gt;(new double[] { 0.0, 1.0, 0.0, 1.0 });
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } });
/// // Use AiModelBuilder facade for balanced bagging classification
/// var builder = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new BalancedBaggingClassifier&lt;double&gt;(nEstimators: 10));
///
/// var result = builder.Build(features, labels);
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Classifier)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.BinaryClassification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Roughly Balanced Bagging for Imbalanced Data",
    "https://doi.org/10.1007/s10115-008-0180-z",
    Year = 2009,
    Authors = "Shohei Hido, Hisashi Kashima")]
public partial class BalancedBaggingClassifier<T> : ClassifierBase<T>
{

    // Returned _baseClassifiers.Count -- the NUMBER of estimators, not a parameter. No restore
    // could rebuild an ensemble from that one value, and it pushed the count away from anything
    // the vector could mean. The base fold now reports an empty vector and a count of zero.
    /// <summary>
    /// The ensemble of base classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each base classifier is trained on a different balanced
    /// bootstrap sample. Their combined votes determine the final prediction.</para>
    /// </remarks>
    private readonly List<DecisionTreeNode> _baseClassifiers;

    /// <summary>
    /// Number of base classifiers in the ensemble.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More classifiers generally means more stable predictions
    /// but longer training. 10-50 classifiers is typical.</para>
    /// </remarks>
    private readonly int _nEstimators;

    /// <summary>
    /// Maximum depth of decision tree base classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deeper trees can learn more complex patterns but may overfit.
    /// null means unlimited depth (grow until pure leaves).</para>
    /// </remarks>
    private readonly int? _maxDepth;

    /// <summary>
    /// Minimum samples required to split a node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values prevent overfitting by requiring more
    /// samples before creating a split.</para>
    /// </remarks>
    private readonly int _minSamplesSplit;

    /// <summary>
    /// Minimum samples required in a leaf node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures every leaf has enough samples to make a
    /// reliable prediction.</para>
    /// </remarks>
    private readonly int _minSamplesLeaf;

    /// <summary>
    /// Ratio of majority samples to minority samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> 1.0 means equal samples per class. 2.0 means twice
    /// as many majority samples as minority. Use higher ratios for less aggressive balancing.</para>
    /// </remarks>
    private readonly double _samplingRatio;

    /// <summary>
    /// Whether to use bootstrap sampling for minority class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bootstrap sampling allows the same sample to be picked
    /// multiple times, creating more diverse training sets.</para>
    /// </remarks>
    private readonly bool _bootstrapMinority;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of BalancedBaggingClassifier.
    /// </summary>
    /// <param name="nEstimators">Number of base classifiers. Default is 10.</param>
    /// <param name="maxDepth">Maximum tree depth. Default is null (unlimited).</param>
    /// <param name="minSamplesSplit">Minimum samples to split. Default is 2.</param>
    /// <param name="minSamplesLeaf">Minimum samples per leaf. Default is 1.</param>
    /// <param name="samplingRatio">Majority to minority ratio. Default is 1.0.</param>
    /// <param name="bootstrapMinority">Bootstrap minority class. Default is true.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Default parameters are good starting points:
    /// <list type="bullet">
    /// <item>Increase nEstimators for more stable predictions</item>
    /// <item>Decrease maxDepth to reduce overfitting</item>
    /// <item>Increase samplingRatio for less aggressive balancing</item>
    /// </list>
    /// </para>
    /// </remarks>
    public BalancedBaggingClassifier(
        int nEstimators = 10,
        int? maxDepth = null,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1,
        double samplingRatio = 1.0,
        bool bootstrapMinority = true,
        int? seed = null)
        : base()
    {
        _nEstimators = nEstimators;
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
        _minSamplesLeaf = minSamplesLeaf;
        _samplingRatio = samplingRatio;
        _bootstrapMinority = bootstrapMinority;
        _baseClassifiers = [];
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.BalancedBaggingClassifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system track what type of model this is.</para>
    /// </remarks>

    /// <summary>
    /// Class index of every training sample (computed once per Train call).
    /// </summary>
    private int[] _sampleClasses = Array.Empty<int>();

    private Tensor<int> _sampleClassTensor = new Tensor<int>(new[] { 0 });

    /// <summary>
    /// Trains the balanced bagging classifier.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training creates multiple balanced bootstrap samples and
    /// trains a decision tree on each. This creates a diverse ensemble that handles imbalance well.</para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        _sampleClassTensor = GetClassIndexTensor(y);
        _sampleClasses = _sampleClassTensor.ToArray();
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _baseClassifiers.Clear();

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int c = 0; c < NumClasses; c++)
        {
            classSamples[c] = [];
        }

        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = _sampleClasses[i];
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classSamples[classIdx].Add(i);
            }
        }

        // Find minority class size
        int minoritySize = classSamples.Values.Min(v => v.Count);
        int majoritySampleSize = (int)(minoritySize * _samplingRatio);

        // Build base classifiers
        var features = Tensor<T>.FromMatrix(x);
        for (int t = 0; t < _nEstimators; t++)
        {
            // Create balanced bootstrap sample
            var sampleIndices = CreateBalancedBootstrap(classSamples, minoritySize, majoritySampleSize);

            // Build decision tree on balanced sample
            var tree = BuildTree(features, y, sampleIndices, 0);
            _baseClassifiers.Add(tree);
        }
    }

    /// <summary>
    /// Creates a balanced bootstrap sample.
    /// </summary>
    /// <param name="classSamples">Samples grouped by class.</param>
    /// <param name="minoritySize">Number of samples from minority class.</param>
    /// <param name="majoritySampleSize">Number of samples from majority class.</param>
    /// <returns>Indices of selected samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a training set where classes are balanced according
    /// to the sampling ratio. Uses bootstrap (with replacement) sampling.</para>
    /// </remarks>
    private int[] CreateBalancedBootstrap(Dictionary<int, List<int>> classSamples,
        int minoritySize, int majoritySampleSize)
    {
        var selected = new List<int>();
        // Find minority class (compatible with .NET Framework 4.7.1)
        int minorityClass = classSamples.OrderBy(kv => kv.Value.Count).First().Key;

        foreach (var (classIdx, samples) in classSamples)
        {
            int sampleSize = classIdx == minorityClass ? minoritySize : majoritySampleSize;

            if (_bootstrapMinority || classIdx != minorityClass)
            {
                // Bootstrap sampling (with replacement)
                for (int i = 0; i < sampleSize; i++)
                {
                    int idx = _random.Next(samples.Count);
                    selected.Add(samples[idx]);
                }
            }
            else
            {
                // Without replacement for minority if bootstrapMinority is false
                var shuffled = samples.OrderBy(_ => _random.Next()).Take(sampleSize);
                selected.AddRange(shuffled);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Builds a decision tree recursively.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Labels.</param>
    /// <param name="sampleIndices">Indices of samples to use.</param>
    /// <param name="depth">Current depth.</param>
    /// <returns>Root node of the tree.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Builds a tree by finding the best split at each node
    /// using Gini impurity, then recursively building subtrees.</para>
    /// </remarks>
    private DecisionTreeNode BuildTree(Tensor<T> x, Vector<T> y, int[] sampleIndices, int depth)
    {
        // Class distribution, majority class (the first wins ties) and class probabilities
        var engine = Engine;
        var classCounts = ClassCounts(_sampleClassTensor, sampleIndices);
        int majorityClass = NumClasses == 0 ? 0 : engine.TensorArgMax(classCounts, 0).ToArray()[0];
        int total = sampleIndices.Length;
        var classProbs = engine.TensorDivideScalar(classCounts, NumOps.FromDouble(total)).ToArray();

        // Stopping conditions
        bool isPure = engine.TensorCountNonzero(classCounts) <= 1;
        bool maxDepthReached = _maxDepth.HasValue && depth >= _maxDepth.Value;
        bool tooFewSamples = sampleIndices.Length < _minSamplesSplit;

        if (isPure || maxDepthReached || tooFewSamples)
        {
            return new DecisionTreeNode
            {
                PredictedClass = majorityClass,
                ClassProbabilities = classProbs
            };
        }

        // Find best split
        var (bestFeature, bestThreshold) = BestGiniSplit(x, _sampleClassTensor, sampleIndices, Enumerable.Range(0, NumFeatures), _minSamplesLeaf);

        // No valid split found
        if (bestFeature < 0)
        {
            return new DecisionTreeNode
            {
                PredictedClass = majorityClass,
                ClassProbabilities = classProbs
            };
        }

        var (bestLeftIndices, bestRightIndices) = PartitionByThreshold(x, sampleIndices, bestFeature, bestThreshold);

        // Recursively build children
        var leftChild = BuildTree(x, y, bestLeftIndices, depth + 1);
        var rightChild = BuildTree(x, y, bestRightIndices, depth + 1);

        return new DecisionTreeNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            LeftChild = leftChild,
            RightChild = rightChild,
            PredictedClass = majorityClass,
            ClassProbabilities = classProbs
        };
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each base classifier votes. The final prediction is the
    /// class with the most votes.</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null || _baseClassifiers.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        // Majority vote over the trees (the first class wins ties)
        var engine = Engine;
        int rows = input.Rows;
        var features = Tensor<T>.FromMatrix(input);
        var allRows = Enumerable.Range(0, rows).ToArray();
        var votes = new Tensor<T>(new[] { rows, NumClasses });
        foreach (var tree in _baseClassifiers)
        {
            var predicted = new int[rows];
            AssignPredictions(tree, features, allRows, predicted);
            votes = engine.TensorAdd(votes, engine.TensorOneHot<T>(new Tensor<int>(predicted, new[] { rows }), NumClasses));
        }

        return MajorityVoteLabels(votes);
    }

    /// <summary>
    /// Gets prediction from a single tree.
    /// </summary>
    /// <param name="node">Tree root node.</param>
    /// <param name="x">Features [n_samples, n_features].</param>
    /// <param name="rows">The rows that reach this node.</param>
    /// <param name="predicted">Receives the predicted class index of every row.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Traverses from root to leaf following the split conditions.</para>
    /// </remarks>
    private void AssignPredictions(DecisionTreeNode node, Tensor<T> x, int[] rows, int[] predicted)
    {
        if (rows.Length == 0)
        {
            return;
        }

        if (node.LeftChild is null || node.RightChild is null)
        {
            foreach (int i in rows)
            {
                predicted[i] = node.PredictedClass;
            }

            return;
        }

        var (left, right) = PartitionByThreshold(x, rows, node.FeatureIndex, node.Threshold);
        AssignPredictions(node.LeftChild, x, left, predicted);
        AssignPredictions(node.RightChild, x, right, predicted);
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
        return new BalancedBaggingClassifier<T>(_nEstimators, _maxDepth, _minSamplesSplit,
            _minSamplesLeaf, _samplingRatio, _bootstrapMinority);
    }

    /// <summary>
    /// Gets feature importance based on split usage.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features used more often for splits are more important.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        // Collect the feature of every split across all trees
        var splitFeatures = new List<int>();
        foreach (var tree in _baseClassifiers)
        {
            CountFeatureUsage(tree, splitFeatures);
        }

        // Usage count per feature, normalized to sum to 1
        var engine = Engine;
        var importance = new Tensor<T>(new[] { NumFeatures });
        if (splitFeatures.Count > 0 && NumFeatures > 0)
        {
            var counts = engine.TensorScatterAdd(importance,
                new Tensor<int>(splitFeatures.ToArray(), new[] { splitFeatures.Count }),
                FilledVector(splitFeatures.Count, NumOps.One).AsTensor(), 0);
            importance = engine.TensorDivideScalar(counts, engine.TensorSum(counts));
        }

        var values = importance.ToArray();
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
    /// Counts feature usage in a tree.
    /// </summary>
    /// <param name="node">Tree node.</param>
    /// <param name="splitFeatures">Receives the feature index of every split.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recursively traverses the tree counting feature splits.</para>
    /// </remarks>
    private void CountFeatureUsage(DecisionTreeNode node, List<int> splitFeatures)
    {
        if (node.LeftChild is null || node.RightChild is null) return;

        splitFeatures.Add(node.FeatureIndex);
        CountFeatureUsage(node.LeftChild, splitFeatures);
        CountFeatureUsage(node.RightChild, splitFeatures);
    }

    private static Dictionary<string, object?> SerializeTreeNode(DecisionTreeNode node)
    {
        var dict = new Dictionary<string, object?>
        {
            ["FeatureIndex"] = node.FeatureIndex,
            ["Threshold"] = node.Threshold,
            ["PredictedClass"] = node.PredictedClass,
            ["ClassProbabilities"] = node.ClassProbabilities
        };

        if (node.LeftChild is not null)
            dict["LeftChild"] = SerializeTreeNode(node.LeftChild);
        if (node.RightChild is not null)
            dict["RightChild"] = SerializeTreeNode(node.RightChild);

        return dict;
    }

    private static DecisionTreeNode DeserializeTreeNode(JObject jObj)
    {
        var node = new DecisionTreeNode
        {
            FeatureIndex = jObj["FeatureIndex"]?.ToObject<int>() ?? 0,
            Threshold = jObj["Threshold"] is JToken threshold
                ? threshold.ToObject<T>() ?? MathHelper.GetNumericOperations<T>().Zero
                : MathHelper.GetNumericOperations<T>().Zero,
            PredictedClass = jObj["PredictedClass"]?.ToObject<int>() ?? 0,
            ClassProbabilities = jObj["ClassProbabilities"]?.ToObject<T[]>()
        };

        if (jObj["LeftChild"] is JObject leftObj)
            node.LeftChild = DeserializeTreeNode(leftObj);
        if (jObj["RightChild"] is JObject rightObj)
            node.RightChild = DeserializeTreeNode(rightObj);

        return node;
    }

    /// <summary>
    /// Internal decision tree node structure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each node is either a split (internal) or prediction (leaf).</para>
    /// </remarks>
    private class DecisionTreeNode
    {
        /// <summary>
        /// Feature index for splitting.
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Threshold for the split.
        /// </summary>
        public T Threshold { get; set; } = MathHelper.GetNumericOperations<T>().Zero;

        /// <summary>
        /// Left child (values &lt;= threshold).
        /// </summary>
        public DecisionTreeNode? LeftChild { get; set; }

        /// <summary>
        /// Right child (values > threshold).
        /// </summary>
        public DecisionTreeNode? RightChild { get; set; }

        /// <summary>
        /// Predicted class for leaf nodes.
        /// </summary>
        public int PredictedClass { get; set; }

        /// <summary>
        /// Class probabilities for leaf nodes.
        /// </summary>
        public T[]? ClassProbabilities { get; set; }
    }
}
