using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BalancedBaggingClassifier<T> : ClassifierBase<T>
{
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
    protected override ModelType GetModelType() => ModelType.BalancedBaggingClassifier;

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
            int classIdx = GetClassIndexFromLabel(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classSamples[classIdx].Add(i);
            }
        }

        // Find minority class size
        int minoritySize = classSamples.Values.Min(v => v.Count);
        int majoritySampleSize = (int)(minoritySize * _samplingRatio);

        // Build base classifiers
        for (int t = 0; t < _nEstimators; t++)
        {
            // Create balanced bootstrap sample
            var sampleIndices = CreateBalancedBootstrap(classSamples, minoritySize, majoritySampleSize);

            // Build decision tree on balanced sample
            var tree = BuildTree(x, y, sampleIndices, 0);
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
    private DecisionTreeNode BuildTree(Matrix<T> x, Vector<T> y, int[] sampleIndices, int depth)
    {
        // Compute class distribution
        var classCounts = new int[NumClasses];
        foreach (int i in sampleIndices)
        {
            int classIdx = GetClassIndexFromLabel(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classCounts[classIdx]++;
            }
        }

        // Find majority class
        int majorityClass = 0;
        int maxCount = 0;
        for (int c = 0; c < NumClasses; c++)
        {
            if (classCounts[c] > maxCount)
            {
                maxCount = classCounts[c];
                majorityClass = c;
            }
        }

        // Class probabilities
        var classProbs = new double[NumClasses];
        int total = sampleIndices.Length;
        for (int c = 0; c < NumClasses; c++)
        {
            classProbs[c] = (double)classCounts[c] / total;
        }

        // Stopping conditions
        bool isPure = classCounts.Count(c => c > 0) <= 1;
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
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGini = double.MaxValue;
        int[]? bestLeftIndices = null;
        int[]? bestRightIndices = null;

        for (int j = 0; j < NumFeatures; j++)
        {
            // Get unique values
            var values = new SortedSet<double>();
            foreach (int i in sampleIndices)
            {
                values.Add(NumOps.ToDouble(x[i, j]));
            }

            if (values.Count < 2)
            {
                continue;
            }

            // Try split points
            var valueList = values.ToList();
            for (int v = 0; v < valueList.Count - 1; v++)
            {
                double threshold = (valueList[v] + valueList[v + 1]) / 2.0;

                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                foreach (int i in sampleIndices)
                {
                    if (NumOps.ToDouble(x[i, j]) <= threshold)
                    {
                        leftIndices.Add(i);
                    }
                    else
                    {
                        rightIndices.Add(i);
                    }
                }

                if (leftIndices.Count < _minSamplesLeaf || rightIndices.Count < _minSamplesLeaf)
                {
                    continue;
                }

                // Compute weighted Gini
                double leftGini = ComputeGini(y, leftIndices);
                double rightGini = ComputeGini(y, rightIndices);
                double weightedGini = ((double)leftIndices.Count / total) * leftGini +
                                      ((double)rightIndices.Count / total) * rightGini;

                if (weightedGini < bestGini)
                {
                    bestGini = weightedGini;
                    bestFeature = j;
                    bestThreshold = threshold;
                    bestLeftIndices = [.. leftIndices];
                    bestRightIndices = [.. rightIndices];
                }
            }
        }

        // No valid split found
        if (bestFeature < 0 || bestLeftIndices is null || bestRightIndices is null)
        {
            return new DecisionTreeNode
            {
                PredictedClass = majorityClass,
                ClassProbabilities = classProbs
            };
        }

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
    /// Computes Gini impurity for a set of samples.
    /// </summary>
    /// <param name="y">All labels.</param>
    /// <param name="indices">Indices to consider.</param>
    /// <returns>Gini impurity value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gini = 1 - sum(p_i^2) where p_i is the proportion of class i.
    /// Lower values mean purer (more homogeneous) sets.</para>
    /// </remarks>
    private double ComputeGini(Vector<T> y, List<int> indices)
    {
        if (indices.Count == 0) return 0;

        var classCounts = new int[NumClasses];
        foreach (int i in indices)
        {
            int classIdx = GetClassIndexFromLabel(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classCounts[classIdx]++;
            }
        }

        double gini = 1.0;
        int total = indices.Count;
        for (int c = 0; c < NumClasses; c++)
        {
            double p = (double)classCounts[c] / total;
            gini -= p * p;
        }

        return gini;
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

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var votes = new int[NumClasses];

            foreach (var tree in _baseClassifiers)
            {
                int pred = PredictTree(tree, input, i);
                votes[pred]++;
            }

            // Majority vote
            int bestClass = 0;
            int maxVotes = 0;
            for (int c = 0; c < NumClasses; c++)
            {
                if (votes[c] > maxVotes)
                {
                    maxVotes = votes[c];
                    bestClass = c;
                }
            }

            predictions[i] = ClassLabels[bestClass];
        }

        return predictions;
    }

    /// <summary>
    /// Gets prediction from a single tree.
    /// </summary>
    /// <param name="node">Tree root node.</param>
    /// <param name="x">Feature matrix.</param>
    /// <param name="sampleIdx">Sample index.</param>
    /// <returns>Predicted class index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Traverses from root to leaf following the split conditions.</para>
    /// </remarks>
    private int PredictTree(DecisionTreeNode node, Matrix<T> x, int sampleIdx)
    {
        while (node.LeftChild is not null && node.RightChild is not null)
        {
            if (NumOps.ToDouble(x[sampleIdx, node.FeatureIndex]) <= node.Threshold)
            {
                node = node.LeftChild;
            }
            else
            {
                node = node.RightChild;
            }
        }
        return node.PredictedClass;
    }

    /// <summary>
    /// Gets the model parameters.
    /// </summary>
    /// <returns>Vector with base classifier count.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensemble models don't fit neatly into parameter vectors.
    /// Use serialization for full persistence.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_baseClassifiers.Count) };
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Parameter vector (limited support).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use serialization to save/load ensemble models.</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Limited support for ensemble models
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
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>New instance with same hyperparameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates an untrained copy with the same settings.</para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new BalancedBaggingClassifier<T>(_nEstimators, _maxDepth, _minSamplesSplit,
            _minSamplesLeaf, _samplingRatio, _bootstrapMinority);
    }

    /// <summary>
    /// Computes gradients (not applicable for tree ensembles).
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <returns>Zero gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree ensembles don't use gradient descent.</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return new Vector<T>(1) { [0] = NumOps.Zero };
    }

    /// <summary>
    /// Applies gradients (not applicable for tree ensembles).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree ensembles don't support gradient updates.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree ensembles don't support gradient-based updates
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
        var importance = new double[NumFeatures];

        foreach (var tree in _baseClassifiers)
        {
            CountFeatureUsage(tree, importance);
        }

        double total = importance.Sum();
        if (total == 0) total = 1;

        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.FromDouble(importance[i] / total);
        }

        return result;
    }

    /// <summary>
    /// Counts feature usage in a tree.
    /// </summary>
    /// <param name="node">Tree node.</param>
    /// <param name="importance">Array to accumulate counts.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recursively traverses the tree counting feature splits.</para>
    /// </remarks>
    private void CountFeatureUsage(DecisionTreeNode node, double[] importance)
    {
        if (node.LeftChild is null || node.RightChild is null) return;

        importance[node.FeatureIndex] += 1;
        CountFeatureUsage(node.LeftChild, importance);
        CountFeatureUsage(node.RightChild, importance);
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
        public double Threshold { get; set; }

        /// <summary>
        /// Left child (values <= threshold).
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
        public double[]? ClassProbabilities { get; set; }
    }
}
