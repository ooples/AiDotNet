using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.ImbalancedEnsemble;

/// <summary>
/// Balanced Random Forest Classifier for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When you have imbalanced data (e.g., 1000 normal transactions vs
/// 10 fraudulent ones), regular classifiers often ignore the minority class. Balanced Random Forest
/// fixes this by training each tree on a balanced subset of data.</para>
///
/// <para><b>How it works:</b> For each tree in the forest:
/// <list type="number">
/// <item>Randomly sample from the minority class with replacement</item>
/// <item>Randomly undersample the majority class to match minority count</item>
/// <item>Train a decision tree on this balanced bootstrap</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item>Better detection of minority class compared to standard Random Forest</item>
/// <item>Maintains the ensemble benefits (reduced variance, robustness)</item>
/// <item>No need to manually balance your dataset</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Fraud detection where fraud cases are rare</item>
/// <item>Medical diagnosis where positive cases are uncommon</item>
/// <item>Any binary classification with significant class imbalance</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Chen, C., Liaw, A., &amp; Breiman, L. (2004). "Using Random Forest to Learn Imbalanced Data"</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BalancedRandomForestClassifier<T> : ClassifierBase<T>
{
    /// <summary>
    /// The ensemble of decision trees.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each tree is trained on a different balanced subset of data.
    /// The final prediction combines all trees' votes.</para>
    /// </remarks>
    private readonly List<DecisionTreeNode> _trees;

    /// <summary>
    /// Number of trees in the forest.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More trees generally means better and more stable predictions,
    /// but also longer training time. 100-500 trees is typical.</para>
    /// </remarks>
    private readonly int _nEstimators;

    /// <summary>
    /// Maximum depth of each tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deeper trees can capture complex patterns but may overfit.
    /// Use None (unlimited) or values like 10-20 for random forests.</para>
    /// </remarks>
    private readonly int? _maxDepth;

    /// <summary>
    /// Number of features to consider for best split.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> At each split, we randomly select this many features and find
    /// the best split among them. Using sqrt(n_features) introduces randomness for diversity.</para>
    /// </remarks>
    private readonly int? _maxFeatures;

    /// <summary>
    /// Minimum samples required to split a node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Nodes with fewer samples won't be split further. Higher values
    /// prevent overfitting.</para>
    /// </remarks>
    private readonly int _minSamplesSplit;

    /// <summary>
    /// Minimum samples required in a leaf node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Every leaf must have at least this many samples. Higher values
    /// create simpler, more generalizable trees.</para>
    /// </remarks>
    private readonly int _minSamplesLeaf;

    /// <summary>
    /// Sampling strategy for the minority class.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how minority class samples are selected:
    /// "all" uses all minority samples, "auto" samples same as majority.</para>
    /// </remarks>
    private readonly string _samplingStrategy;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether to use bootstrap sampling (with replacement).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bootstrap sampling means the same sample can be picked multiple
    /// times. This creates diverse training sets for each tree.</para>
    /// </remarks>
    private readonly bool _bootstrap;

    /// <summary>
    /// Initializes a new instance of BalancedRandomForestClassifier.
    /// </summary>
    /// <param name="nEstimators">Number of trees. Default is 100.</param>
    /// <param name="maxDepth">Maximum tree depth. Default is null (unlimited).</param>
    /// <param name="maxFeatures">Features to consider per split. Default is sqrt(n_features).</param>
    /// <param name="minSamplesSplit">Minimum samples to split. Default is 2.</param>
    /// <param name="minSamplesLeaf">Minimum samples per leaf. Default is 1.</param>
    /// <param name="samplingStrategy">How to sample. Default is "auto".</param>
    /// <param name="bootstrap">Use bootstrap sampling. Default is true.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Default parameters work well for most cases. Key tuning:
    /// <list type="bullet">
    /// <item>Increase nEstimators for better accuracy (at cost of speed)</item>
    /// <item>Adjust maxDepth if overfitting (reduce) or underfitting (increase)</item>
    /// <item>Increase minSamplesLeaf to reduce overfitting</item>
    /// </list>
    /// </para>
    /// </remarks>
    public BalancedRandomForestClassifier(
        int nEstimators = 100,
        int? maxDepth = null,
        int? maxFeatures = null,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1,
        string samplingStrategy = "auto",
        bool bootstrap = true,
        int? seed = null)
        : base()
    {
        _nEstimators = nEstimators;
        _maxDepth = maxDepth;
        _maxFeatures = maxFeatures;
        _minSamplesSplit = minSamplesSplit;
        _minSamplesLeaf = minSamplesLeaf;
        _samplingStrategy = samplingStrategy;
        _bootstrap = bootstrap;
        _trees = [];
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.BalancedRandomForestClassifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system track what type of model this is.</para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.BalancedRandomForestClassifier;

    /// <summary>
    /// Trains the balanced random forest classifier.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training creates multiple trees, each trained on a balanced
    /// bootstrap sample. This ensures minority class patterns are well-represented in all trees.</para>
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

        _trees.Clear();

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

        // Compute number of features to consider at each split
        int actualMaxFeatures = _maxFeatures ?? (int)Math.Sqrt(NumFeatures);
        actualMaxFeatures = Math.Max(1, Math.Min(actualMaxFeatures, NumFeatures));

        // Build trees
        for (int t = 0; t < _nEstimators; t++)
        {
            // Create balanced bootstrap sample
            var sampleIndices = CreateBalancedBootstrap(classSamples, minoritySize);

            // Build tree on balanced sample
            var tree = BuildTree(x, y, sampleIndices, 0, actualMaxFeatures);
            _trees.Add(tree);
        }
    }

    /// <summary>
    /// Creates a balanced bootstrap sample.
    /// </summary>
    /// <param name="classSamples">Samples grouped by class.</param>
    /// <param name="samplesPerClass">Number of samples per class.</param>
    /// <returns>Indices of selected samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a dataset where each class has the same number
    /// of samples. For minority class, we sample with replacement (might repeat samples).
    /// For majority class, we undersample (use fewer samples).</para>
    /// </remarks>
    private int[] CreateBalancedBootstrap(Dictionary<int, List<int>> classSamples, int samplesPerClass)
    {
        var selected = new List<int>();

        foreach (var (classIdx, samples) in classSamples)
        {
            for (int i = 0; i < samplesPerClass; i++)
            {
                int idx = _random.Next(samples.Count);
                selected.Add(samples[idx]);
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
    /// <param name="maxFeatures">Features to consider per split.</param>
    /// <returns>Root node of the tree.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Builds the tree by finding the best split at each node:
    /// <list type="number">
    /// <item>Randomly select maxFeatures features to consider</item>
    /// <item>For each feature, find the best split point using Gini impurity</item>
    /// <item>Split the data and recursively build child nodes</item>
    /// </list>
    /// </para>
    /// </remarks>
    private DecisionTreeNode BuildTree(Matrix<T> x, Vector<T> y, int[] sampleIndices,
        int depth, int maxFeatures)
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

        // Majority class prediction
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

        // Randomly select features to consider
        var featureIndices = Enumerable.Range(0, NumFeatures).ToArray();
        ShuffleArray(featureIndices);
        var selectedFeatures = featureIndices.Take(maxFeatures).ToArray();

        // Find best split
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGini = double.MaxValue;
        int[]? bestLeftIndices = null;
        int[]? bestRightIndices = null;

        foreach (int feature in selectedFeatures)
        {
            // Get unique values for this feature
            var values = new SortedSet<double>();
            foreach (int i in sampleIndices)
            {
                values.Add(NumOps.ToDouble(x[i, feature]));
            }

            if (values.Count < 2)
            {
                continue;
            }

            // Try split points (midpoints between consecutive values)
            var valueList = values.ToList();
            for (int v = 0; v < valueList.Count - 1; v++)
            {
                double threshold = (valueList[v] + valueList[v + 1]) / 2.0;

                var leftIndices = new List<int>();
                var rightIndices = new List<int>();

                foreach (int i in sampleIndices)
                {
                    if (NumOps.ToDouble(x[i, feature]) <= threshold)
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

                // Compute weighted Gini impurity
                double leftGini = ComputeGini(y, leftIndices);
                double rightGini = ComputeGini(y, rightIndices);
                double weightedGini = ((double)leftIndices.Count / total) * leftGini +
                                      ((double)rightIndices.Count / total) * rightGini;

                if (weightedGini < bestGini)
                {
                    bestGini = weightedGini;
                    bestFeature = feature;
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
        var leftChild = BuildTree(x, y, bestLeftIndices, depth + 1, maxFeatures);
        var rightChild = BuildTree(x, y, bestRightIndices, depth + 1, maxFeatures);

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
    /// Computes the Gini impurity for a set of samples.
    /// </summary>
    /// <param name="y">All labels.</param>
    /// <param name="indices">Indices of samples to consider.</param>
    /// <returns>Gini impurity value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gini impurity measures how "mixed" the classes are:
    /// - 0 = perfectly pure (all same class)
    /// - Higher values = more mixed (harder to predict)
    /// The formula is: 1 - sum(p_i^2) where p_i is the proportion of class i.</para>
    /// </remarks>
    private double ComputeGini(Vector<T> y, List<int> indices)
    {
        if (indices.Count == 0)
        {
            return 0;
        }

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
    /// Shuffles an array in place.
    /// </summary>
    /// <param name="array">Array to shuffle.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Fisher-Yates shuffle - creates a random permutation of the array.</para>
    /// </remarks>
    private void ShuffleArray<TArray>(TArray[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each tree votes for a class. The final prediction is the
    /// class with the most votes (majority voting).</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var votes = new int[NumClasses];

            foreach (var tree in _trees)
            {
                int prediction = PredictTree(tree, input, i);
                votes[prediction]++;
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
    /// <para><b>For Beginners:</b> Traverses the tree from root to leaf by comparing feature
    /// values to thresholds at each node.</para>
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
    /// <returns>Vector with tree count (simplified representation).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree ensembles don't fit neatly into a parameter vector.
    /// Use serialization for full model persistence.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_trees.Count) };
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Parameter vector (limited support).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree models are better loaded via serialization.</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Limited support for tree models
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
        return new BalancedRandomForestClassifier<T>(_nEstimators, _maxDepth, _maxFeatures,
            _minSamplesSplit, _minSamplesLeaf, _samplingStrategy, _bootstrap);
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
        return new BalancedRandomForestClassifier<T>(_nEstimators, _maxDepth, _maxFeatures,
            _minSamplesSplit, _minSamplesLeaf, _samplingStrategy, _bootstrap);
    }

    /// <summary>
    /// Computes gradients (not applicable for tree models).
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <returns>Zero gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree models don't use gradient descent.</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return new Vector<T>(1) { [0] = NumOps.Zero };
    }

    /// <summary>
    /// Applies gradients (not applicable for tree models).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree models don't support gradient updates.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree models don't support gradient-based updates
    }

    /// <summary>
    /// Gets feature importance based on split usage.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features that appear in more splits across all trees are
    /// considered more important. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new double[NumFeatures];

        foreach (var tree in _trees)
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
    /// <para><b>For Beginners:</b> Recursively counts how many times each feature is used for splits.</para>
    /// </remarks>
    private void CountFeatureUsage(DecisionTreeNode node, double[] importance)
    {
        if (node.LeftChild is null || node.RightChild is null)
        {
            return;
        }

        importance[node.FeatureIndex] += 1;
        CountFeatureUsage(node.LeftChild, importance);
        CountFeatureUsage(node.RightChild, importance);
    }

    /// <summary>
    /// Internal decision tree node structure.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each node either splits on a feature (internal node) or
    /// makes a prediction (leaf node).</para>
    /// </remarks>
    private class DecisionTreeNode
    {
        /// <summary>
        /// Feature index used for splitting.
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Threshold value for the split.
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
