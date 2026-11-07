using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Classification.Boosting;

/// <summary>
/// Histogram-based Gradient Boosting Classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a fast gradient boosting implementation that uses histograms
/// to speed up the tree-building process. It's similar to LightGBM and scikit-learn's
/// HistGradientBoostingClassifier.</para>
///
/// <para><b>How it works:</b> Instead of evaluating all possible split points, this algorithm:
/// <list type="number">
/// <item>Bins continuous features into a fixed number of buckets (histograms)</item>
/// <item>Builds trees by evaluating splits only at bin boundaries</item>
/// <item>Uses gradient boosting to iteratively improve predictions</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item><b>Speed:</b> Much faster than traditional gradient boosting on large datasets</item>
/// <item><b>Memory:</b> Uses less memory due to binning</item>
/// <item><b>Missing values:</b> Please impute before training (missing-value handling is not built in)</item>
/// <item><b>Scalability:</b> Scales well to millions of samples</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>When you have a large dataset (thousands to millions of samples)</item>
/// <item>When you need fast training without sacrificing much accuracy</item>
/// <item>For tabular data classification problems</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"</item>
/// <item>Scikit-learn HistGradientBoostingClassifier implementation</item>
/// </list>
/// </para>
/// </remarks>
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
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
/// // Train through the facade: the builder configures the model, runs the training pipeline and
/// // returns a result to predict on.
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new HistGradientBoostingClassifier&lt;double&gt;(
///         maxBins: 256, maxDepth: 6, nEstimators: 100, learningRate: 0.1))
///     .Build(features, labels);
///
/// // Predict class for new sample
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("LightGBM: A Highly Efficient Gradient Boosting Decision Tree", "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree")]
public partial class HistGradientBoostingClassifier<T> : ClassifierBase<T>
{

    // Its own comment: "For tree-based models, parameters do not fit the typical vector format".
    /// <summary>
    /// The ensemble of histogram-based decision trees.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model is a collection of trees where each tree tries to
    /// correct the errors of the previous trees. This is the "boosting" part.</para>
    /// </remarks>
    private readonly List<HistTree> _trees;

    /// <summary>
    /// The bin boundaries for each feature.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instead of storing exact values, features are grouped into
    /// bins. This array stores the boundaries between bins for each feature.</para>
    /// </remarks>
    private double[][]? _binBoundaries;

    /// <summary>
    /// The initial prediction (log-odds for binary, class probabilities for multiclass).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before any trees are added, the model makes a baseline
    /// prediction based on the class distribution in the training data.</para>
    /// </remarks>
    private double[]? _initialPrediction;

    /// <summary>
    /// Number of histogram bins per feature.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More bins = more precision but slower training.
    /// 256 is a common choice that balances speed and accuracy.</para>
    /// </remarks>
    private readonly int _maxBins;

    /// <summary>
    /// Maximum depth of each tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deeper trees can capture more complex patterns but
    /// may overfit. Typical values range from 3 to 10.</para>
    /// </remarks>
    private readonly int _maxDepth;

    /// <summary>
    /// Number of boosting iterations (trees).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More trees usually improve accuracy but increase training
    /// time and risk of overfitting. Use early stopping to find the optimal number.</para>
    /// </remarks>
    private readonly int _nEstimators;

    /// <summary>
    /// Learning rate shrinkage.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each tree's contribution is multiplied by this value.
    /// Smaller values require more trees but often give better results.</para>
    /// </remarks>
    private readonly double _learningRate;

    /// <summary>
    /// Minimum samples required to split a node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Nodes with fewer samples than this won't be split.
    /// Higher values prevent overfitting.</para>
    /// </remarks>
    private readonly int _minSamplesLeaf;

    /// <summary>
    /// L2 regularization strength.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Penalizes large leaf values to prevent overfitting.
    /// Higher values create more conservative predictions.</para>
    /// </remarks>
    private readonly double _l2Regularization;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Initializes a new instance of HistGradientBoostingClassifier.
    /// </summary>
    /// <param name="maxBins">Maximum number of bins per feature. Default is 256.</param>
    /// <param name="maxDepth">Maximum depth of each tree. Default is 6.</param>
    /// <param name="nEstimators">Number of boosting iterations. Default is 100.</param>
    /// <param name="learningRate">Learning rate shrinkage. Default is 0.1.</param>
    /// <param name="minSamplesLeaf">Minimum samples per leaf. Default is 20.</param>
    /// <param name="l2Regularization">L2 regularization strength. Default is 0.0.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Start with defaults and tune based on validation performance:
    /// <list type="bullet">
    /// <item>If overfitting: reduce maxDepth, increase minSamplesLeaf, reduce nEstimators</item>
    /// <item>If underfitting: increase maxDepth, increase nEstimators, reduce regularization</item>
    /// <item>For speed: reduce maxBins, reduce nEstimators</item>
    /// </list>
    /// </para>
    /// </remarks>
    public HistGradientBoostingClassifier(
        int maxBins = 256,
        int maxDepth = 6,
        int nEstimators = 100,
        double learningRate = 0.1,
        int minSamplesLeaf = 20,
        double l2Regularization = 0.0,
        int? seed = null)
        : base()
    {
        _maxBins = maxBins;
        _maxDepth = maxDepth;
        _nEstimators = nEstimators;
        _learningRate = learningRate;
        _minSamplesLeaf = minSamplesLeaf;
        _l2Regularization = l2Regularization;
        _trees = [];
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.HistGradientBoostingClassifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system track what type of model this is.</para>
    /// </remarks>

    /// <summary>
    /// Trains the histogram-based gradient boosting classifier.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves:
    /// <list type="number">
    /// <item>Binning all features into histograms</item>
    /// <item>Computing initial predictions from class frequencies</item>
    /// <item>Iteratively: compute gradients, build a tree, update predictions</item>
    /// </list>
    ///
    /// The gradient boosting formula: prediction = initial + sum(learningRate * tree[i].predict(X))
    /// </para>
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

        int n = x.Rows;
        _trees.Clear();

        // Steps 1-2: Compute each feature's bin boundaries and bin the features
        // (HistGradientBoostingClassifier.HistBuilder.cs)
        _binBoundaries = PrepareHistBuilder(x);
        var targets = new int[n];
        for (int i = 0; i < n; i++) targets[i] = GetClassIndexFromLabel(y[i]);

        // Step 3: Initialize predictions
        _initialPrediction = ComputeInitialPrediction(y);

        // For binary classification, we use a single output
        // For multiclass, we use K outputs (one vs rest)
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        var predictions = new double[n, numOutputs];

        // Initialize predictions
        for (int i = 0; i < n; i++)
        {
            for (int k = 0; k < numOutputs; k++)
            {
                predictions[i, k] = _initialPrediction[k];
            }
        }

        // Step 4: Gradient boosting iterations
        try
        {
            for (int iter = 0; iter < _nEstimators; iter++)
            {
                // Compute all gradients from the same predictions snapshot
                var grads = new double[numOutputs][];
                var hess = new double[numOutputs][];
                for (int k = 0; k < numOutputs; k++)
                {
                    (grads[k], hess[k]) = ComputeGradientsAndHessians(targets, predictions, k);
                }

                // Build each class's tree; its leaves record every sample's output for the update.
                for (int k = 0; k < numOutputs; k++)
                {
                    _trees.Add(BuildHistTreeDense(grads[k], hess[k]));
                    int output = k;
                    ForEachRowBlock(n, (from, to) =>
                    {
                        for (int i = from; i < to; i++)
                        {
                            predictions[i, output] += _learningRate * _sampleLeafValue[i];
                        }
                    });
                }
            }
        }
        finally
        {
            ReleaseHistBuilder();
        }
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prediction aggregates the outputs of all trees:
    /// <list type="number">
    /// <item>Bin the input features using the same boundaries from training</item>
    /// <item>Sum: initial_prediction + learningRate * sum(tree[i].predict(X))</item>
    /// <item>Convert to probabilities using sigmoid (binary) or softmax (multiclass)</item>
    /// <item>Return the class with highest probability</item>
    /// </list>
    /// </para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (_binBoundaries is null || _initialPrediction is null || ClassLabels is null)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        if (input.Columns != NumFeatures)
        {
            throw new ArgumentException($"Expected {NumFeatures} features but got {input.Columns}.", nameof(input));
        }

        var forest = new FlatForest(_trees, _binBoundaries);
        var initialPrediction = _initialPrediction;
        var classLabels = ClassLabels;
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        int columns = input.Columns;
        var predictions = new Vector<T>(input.Rows);

        // Rows are independent, so fixed blocks of them run in parallel.
        int jobs = ResolveHistJobs();
        int blocks = (input.Rows + PredictBlockRows - 1) / PredictBlockRows;
        Parallel.For(0, blocks, new ParallelOptions { MaxDegreeOfParallelism = jobs }, block =>
        {
            var row = new double[columns];
            var rawPred = new double[numOutputs];
            int end = Math.Min(input.Rows, (block + 1) * PredictBlockRows);
            for (int i = block * PredictBlockRows; i < end; i++)
            {
                var source = input.GetRowReadOnlySpan(i);
                for (int j = 0; j < columns; j++) row[j] = NumOps.ToDouble(source[j]);
                predictions[i] = PredictRow(forest, row, rawPred, initialPrediction, classLabels, numOutputs);
            }
        });

        return predictions;
    }

    /// <summary>Rows per block of a parallel prediction: small, as a row costs a walk down every tree.</summary>
    private const int PredictBlockRows = 128;

    /// <summary>The predicted label of one row: the trees' outputs summed in tree order, then the likeliest class.</summary>
    private T PredictRow(FlatForest forest, double[] row, double[] rawPred, double[] initialPrediction,
        Vector<T> classLabels, int numOutputs)
    {
        for (int k = 0; k < numOutputs; k++)
        {
            rawPred[k] = initialPrediction[k];
        }

        // Sum tree predictions
        for (int treeIdx = 0; treeIdx < forest.Trees; treeIdx++)
        {
            int k = treeIdx % numOutputs;
            rawPred[k] += _learningRate * forest.Evaluate(treeIdx, row);
        }

        // Convert to class prediction
        if (NumClasses == 2)
        {
            double prob = 1.0 / (1.0 + Math.Exp(-rawPred[0]));
            return prob >= 0.5 ? classLabels[1] : classLabels[0];
        }
        else
        {
            // Softmax and argmax
            double maxVal = double.MinValue;
            for (int k = 0; k < NumClasses; k++) maxVal = Math.Max(maxVal, rawPred[k]);
            double sumExp = 0;
            for (int k = 0; k < NumClasses; k++)
            {
                sumExp += Math.Exp(rawPred[k] - maxVal);
            }

            int bestClass = 0;
            double bestProb = 0;
            for (int k = 0; k < NumClasses; k++)
            {
                double prob = Math.Exp(rawPred[k] - maxVal) / sumExp;
                if (prob > bestProb)
                {
                    bestProb = prob;
                    bestClass = k;
                }
            }
            return classLabels[bestClass];
        }
    }

    /// <summary>
    /// The trees as flat arrays for prediction, each split compared against its bin's upper boundary in feature units.
    /// </summary>
    /// <remarks>
    /// A value falls in bin <c>b</c> when <c>b</c> boundaries lie below it (equal goes to the lower bin), so it is in a
    /// bin &lt;= <c>t</c> exactly when it is not above boundary <c>t</c>: prediction reads the raw value instead of
    /// binning every row first. NaN, which binning put in bin 0, is not above any boundary, so it goes left too.
    /// </remarks>
    private sealed class FlatForest
    {
        private readonly int[] _roots;
        private readonly int[] _feature;
        private readonly double[] _threshold;
        private readonly int[] _left;
        private readonly int[] _right;
        private readonly double[] _value;

        public FlatForest(List<HistTree> trees, double[][] boundaries)
        {
            int nodes = 0;
            var stack = new Stack<HistTree>();
            foreach (var tree in trees)
            {
                stack.Push(tree);
                while (stack.Count > 0)
                {
                    var node = stack.Pop();
                    nodes++;
                    if (node.LeftChild is not null && node.RightChild is not null)
                    {
                        stack.Push(node.LeftChild);
                        stack.Push(node.RightChild);
                    }
                }
            }

            _roots = new int[trees.Count];
            _feature = new int[nodes];
            _threshold = new double[nodes];
            _left = new int[nodes];
            _right = new int[nodes];
            _value = new double[nodes];
            int next = 0;
            var pending = new Stack<(HistTree Node, int Index)>();
            for (int t = 0; t < trees.Count; t++)
            {
                _roots[t] = next;
                pending.Push((trees[t], next++));
                while (pending.Count > 0)
                {
                    var (node, index) = pending.Pop();
                    if (node.LeftChild is not null && node.RightChild is not null)
                    {
                        var bounds = boundaries[node.FeatureIndex];
                        _feature[index] = node.FeatureIndex;
                        _threshold[index] = node.BinThreshold < bounds.Length
                            ? bounds[node.BinThreshold]
                            : double.PositiveInfinity;
                        _left[index] = next++;
                        _right[index] = next++;
                        pending.Push((node.LeftChild, _left[index]));
                        pending.Push((node.RightChild, _right[index]));
                    }
                    else
                    {
                        _feature[index] = -1;
                        _value[index] = node.LeafValue;
                    }
                }
            }
        }

        public int Trees => _roots.Length;

        /// <summary>The leaf value tree <paramref name="tree"/> gives <paramref name="row"/>.</summary>
        public double Evaluate(int tree, double[] row)
        {
            int node = _roots[tree];
            while (_feature[node] >= 0)
                node = row[_feature[node]] > _threshold[node] ? _right[node] : _left[node];
            return _value[node];
        }
    }

    /// <summary>
    /// Computes the initial prediction (prior).
    /// </summary>
    /// <param name="y">Target labels.</param>
    /// <returns>Initial prediction values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The initial prediction is the "baseline" before any trees are added.
    /// For binary classification, it's the log-odds of the positive class.
    /// For multiclass, it's the log-probability of each class.</para>
    /// </remarks>
    private double[] ComputeInitialPrediction(Vector<T> y)
    {
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        var classCounts = new int[NumClasses];

        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = GetClassIndexFromLabel(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classCounts[classIdx]++;
            }
        }

        var initial = new double[numOutputs];

        if (NumClasses == 2)
        {
            // Binary: log-odds
            double p1 = (double)classCounts[1] / y.Length;
            p1 = Math.Max(1e-7, Math.Min(1 - 1e-7, p1));
            initial[0] = Math.Log(p1 / (1 - p1));
        }
        else
        {
            // Multiclass: log probabilities
            for (int k = 0; k < NumClasses; k++)
            {
                double p = (double)classCounts[k] / y.Length;
                p = Math.Max(1e-7, p);
                initial[k] = Math.Log(p);
            }
        }

        return initial;
    }

    /// <summary>
    /// Computes gradients and hessians for gradient boosting.
    /// </summary>
    /// <param name="targets">Each sample's class index.</param>
    /// <param name="predictions">Current predictions.</param>
    /// <param name="classIdx">Class index for multiclass problems.</param>
    /// <returns>Tuple of gradients and hessians.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how wrong our predictions are.
    /// Hessians (second derivatives) tell us how confident we should be in our gradient estimates.
    /// Together they help us build trees that efficiently reduce the loss.</para>
    /// </remarks>
    private (double[] gradients, double[] hessians) ComputeGradientsAndHessians(
        int[] targets, double[,] predictions, int classIdx)
    {
        int n = targets.Length;
        var gradients = new double[n];
        var hessians = new double[n];

        if (NumClasses == 2)
        {
            // Binary cross-entropy gradients
            ForEachRowBlock(n, (from, to) =>
            {
                for (int i = from; i < to; i++)
                {
                    double target = targets[i] == 1 ? 1.0 : 0.0;
                    double pred = 1.0 / (1.0 + Math.Exp(-predictions[i, 0]));
                    gradients[i] = pred - target;
                    hessians[i] = pred * (1 - pred);
                    hessians[i] = Math.Max(hessians[i], 1e-7);
                }
            });
        }
        else
        {
            // Multiclass cross-entropy gradients
            ForEachRowBlock(n, (from, to) =>
            {
                for (int i = from; i < to; i++)
                {
                    // Softmax
                    double maxVal = double.MinValue;
                    for (int k = 0; k < NumClasses; k++)
                    {
                        maxVal = Math.Max(maxVal, predictions[i, k]);
                    }

                    double sumExp = 0;
                    for (int k = 0; k < NumClasses; k++)
                    {
                        sumExp += Math.Exp(predictions[i, k] - maxVal);
                    }

                    double pred = Math.Exp(predictions[i, classIdx] - maxVal) / sumExp;
                    double target = targets[i] == classIdx ? 1.0 : 0.0;

                    gradients[i] = pred - target;
                    hessians[i] = pred * (1 - pred);
                    hessians[i] = Math.Max(hessians[i], 1e-7);
                }
            });
        }

        return (gradients, hessians);
    }


    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters (limited support).</param>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new model. For tree models, the parameters
    /// don't fully capture the model state.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var model = new HistGradientBoostingClassifier<T>(_maxBins, _maxDepth, _nEstimators,
            _learningRate, _minSamplesLeaf, _l2Regularization);
        return model;
    }

    /// <summary>
    /// Gets feature importance based on total gain reduction.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features that appear in more splits or provide larger
    /// gain reductions are considered more important. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new double[NumFeatures];

        // Count feature usage across all trees
        foreach (var tree in _trees)
        {
            CountFeatureUsage(tree, importance);
        }

        // Normalize
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
    /// Counts feature usage in a tree for importance calculation.
    /// </summary>
    /// <param name="node">Tree node to process.</param>
    /// <param name="importance">Array to accumulate importance scores.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recursively traverses the tree and counts how often
    /// each feature is used for splits.</para>
    /// </remarks>
    private void CountFeatureUsage(HistTree node, double[] importance)
    {
        if (node.LeftChild is null || node.RightChild is null)
        {
            return;
        }

        importance[node.FeatureIndex] += 1;
        CountFeatureUsage(node.LeftChild, importance);
        CountFeatureUsage(node.RightChild, importance);
    }

    private static HistTree CloneHistTree(HistTree node)
    {
        var cloned = new HistTree
        {
            FeatureIndex = node.FeatureIndex,
            BinThreshold = node.BinThreshold,
            LeafValue = node.LeafValue
        };
        if (node.LeftChild is not null)
            cloned.LeftChild = CloneHistTree(node.LeftChild);
        if (node.RightChild is not null)
            cloned.RightChild = CloneHistTree(node.RightChild);
        return cloned;
    }

    private Dictionary<string, object> SerializeHistTree(HistTree node)
    {
        var data = new Dictionary<string, object>
        {
            { "FeatureIndex", node.FeatureIndex },
            { "BinThreshold", node.BinThreshold },
            { "LeafValue", node.LeafValue },
            { "IsLeaf", node.LeftChild is null && node.RightChild is null }
        };

        if (node.LeftChild is not null)
            data["Left"] = SerializeHistTree(node.LeftChild);
        if (node.RightChild is not null)
            data["Right"] = SerializeHistTree(node.RightChild);

        return data;
    }

    private HistTree DeserializeHistTree(JObject data)
    {
        var node = new HistTree
        {
            FeatureIndex = data["FeatureIndex"]?.ToObject<int>() ?? 0,
            BinThreshold = data["BinThreshold"]?.ToObject<int>() ?? 0,
            LeafValue = data["LeafValue"]?.ToObject<double>() ?? 0.0
        };

        var leftToken = data["Left"];
        if (leftToken is not null && leftToken.Type == JTokenType.Object)
            node.LeftChild = DeserializeHistTree((JObject)leftToken);

        var rightToken = data["Right"];
        if (rightToken is not null && rightToken.Type == JTokenType.Object)
            node.RightChild = DeserializeHistTree((JObject)rightToken);

        return node;
    }

    /// <summary>
    /// Internal tree node structure for histogram-based trees.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This represents a node in the decision tree. Internal nodes
    /// have a split condition (feature and bin threshold), leaf nodes have a prediction value.</para>
    /// </remarks>
    private class HistTree
    {
        /// <summary>
        /// Feature index used for splitting.
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Bin threshold for the split (values &lt;= threshold go left).
        /// </summary>
        public int BinThreshold { get; set; }

        /// <summary>
        /// Left child node (values &lt;= threshold).
        /// </summary>
        public HistTree? LeftChild { get; set; }

        /// <summary>
        /// Right child node (values > threshold).
        /// </summary>
        public HistTree? RightChild { get; set; }

        /// <summary>
        /// Prediction value for leaf nodes.
        /// </summary>
        public double LeafValue { get; set; }
    }
}
