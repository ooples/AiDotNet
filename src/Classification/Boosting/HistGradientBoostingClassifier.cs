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
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>[]? _binBoundaries;

    /// <summary>
    /// The initial prediction (log-odds for binary, class probabilities for multiclass).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before any trees are added, the model makes a baseline
    /// prediction based on the class distribution in the training data.</para>
    /// </remarks>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _initialPrediction;

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
    /// Gets or sets the number of threads that build each tree: a positive count, or negative to count back from the
    /// processor count (-1, the default, uses every processor, as scikit-learn's HistGradientBoostingClassifier
    /// does). The trained model is the same for any value.
    /// </summary>
    /// <remarks>
    /// Histograms, split gains and predictions run as engine tensor operations, which parallelise inside the engine,
    /// so this value is validated (0 is rejected) but does not change the work split.
    /// </remarks>
    public int NJobs { get; set; } = -1;

    /// <summary>
    /// Rejects an <see cref="NJobs"/> of 0, which names no thread count.
    /// </summary>
    private void ValidateJobs()
    {
        if (NJobs == 0)
        {
            throw new InvalidOperationException(
                $"{nameof(NJobs)} must be positive, or negative to count back from the processor count; got 0.");
        }
    }

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

        ValidateJobs();

        // Step 1: Compute bin boundaries for each feature
        _binBoundaries = ComputeBinBoundaries(x);

        // Step 2: Bin the features
        var binnedX = BinFeatures(x, _binBoundaries);

        // Step 3: Initialize predictions
        _initialPrediction = ComputeInitialPrediction(y);

        // For binary classification, we use a single output
        // For multiclass, we use K outputs (one vs rest)
        var engine = Engine;
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        var predictions = engine.TensorTile(_initialPrediction.AsTensor().Reshape(1, numOutputs), new[] { n, 1 });
        // One-hot targets (every training label is one of ClassLabels)
        var targets = n == 0
            ? new Tensor<T>(new[] { 0, NumClasses })
            : engine.TensorOneHot<T>(GetClassIndexTensor(y), NumClasses);
        T learningRate = NumOps.FromDouble(_learningRate);

        // Step 4: Gradient boosting iterations
        for (int iter = 0; iter < _nEstimators; iter++)
        {
            // Compute all gradients from the same predictions snapshot
            var (gradients, hessians) = ComputeGradientsAndHessians(targets, predictions);

            // Build trees for all classes, then update predictions for all classes together
            var outputs = new Tensor<T>[numOutputs];
            for (int k = 0; k < numOutputs; k++)
            {
                var tree = BuildHistTree(binnedX,
                    gradients.Slice(1, k, k + 1).Reshape(n), hessians.Slice(1, k, k + 1).Reshape(n), 0);
                _trees.Add(tree);
                outputs[k] = PredictTree(tree, binnedX, n).Reshape(n, 1);
            }

            var treeOutputs = numOutputs == 1 ? outputs[0] : engine.TensorConcatenate(outputs, 1);
            predictions = engine.TensorAdd(predictions, engine.TensorMultiplyScalar(treeOutputs, learningRate));
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

        ValidateJobs();
        var engine = Engine;
        int rows = input.Rows;
        if (rows == 0)
        {
            return new Vector<T>(0);
        }

        var binnedX = BinFeatures(input, _binBoundaries);
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        T learningRate = NumOps.FromDouble(_learningRate);

        // Sum tree predictions: tree t contributes to output t % numOutputs
        var outputs = new Tensor<T>[numOutputs];
        for (int k = 0; k < numOutputs; k++)
        {
            outputs[k] = FilledVector(rows, _initialPrediction[k]).AsTensor();
        }

        for (int treeIdx = 0; treeIdx < _trees.Count; treeIdx++)
        {
            int k = treeIdx % numOutputs;
            outputs[k] = engine.TensorAdd(outputs[k],
                engine.TensorMultiplyScalar(PredictTree(_trees[treeIdx], binnedX, rows), learningRate));
        }

        Tensor<int> bestClasses;
        if (NumClasses == 2)
        {
            // P(class 1) = sigmoid(raw) >= 0.5
            var prob = engine.TensorSigmoid(outputs[0]);
            var isPositive = engine.TensorGreaterEqual(prob, FilledVector(rows, NumOps.FromDouble(0.5)).AsTensor());
            bestClasses = ZeroOneToInt(isPositive);
        }
        else
        {
            // Softmax and argmax (the first class wins ties)
            var raw = engine.TensorConcatenate(outputs.Select(o => o.Reshape(rows, 1)).ToArray(), 1);
            bestClasses = engine.TensorArgMax(engine.TensorSoftmax(raw, 1), 1);
        }

        return ClassIndicesToLabels(bestClasses);
    }

    /// <summary>
    /// Computes histogram bin boundaries for each feature.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <returns>Array of bin boundaries for each feature.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This finds evenly-spaced quantile boundaries for each feature.
    /// For example, with 256 bins, we find the values that divide the data into 256 equal parts.</para>
    /// </remarks>
    private Vector<T>[] ComputeBinBoundaries(Matrix<T> x)
    {
        int p = x.Columns;
        int n = x.Rows;
        var boundaries = new Vector<T>[p];
        var engine = Engine;

        for (int j = 0; j < p; j++)
        {
            if (n == 0)
            {
                boundaries[j] = new Vector<T>(0);
                continue;
            }

            // Validate: every value must be finite
            var column = x.GetColumn(j).AsTensor();
            var invalid = engine.TensorWhere(engine.TensorIsFinite(column),
                new Tensor<T>(new[] { n }), FilledVector(n, NumOps.One).AsTensor());
            if (engine.TensorCountNonzero(invalid) > 0)
            {
                var badRows = engine.TensorNonzero(invalid);
                throw new ArgumentException(
                    $"Missing/NaN/Infinity values are not supported. Found at row {badRows.ToArray()[0]}, feature {j}. Please impute before training.",
                    nameof(x));
            }

            var sorted = engine.TensorSort(column, 0, false).Values;

            // Compute quantile boundaries (strictly increasing, so duplicates are dropped)
            int numBins = Math.Min(_maxBins, n);
            if (numBins <= 1)
            {
                boundaries[j] = new Vector<T>(0);
                continue;
            }

            var positions = new int[numBins - 1];
            for (int b = 1; b < numBins; b++)
            {
                positions[b - 1] = Math.Min((int)((double)b / numBins * n), n - 1);
            }

            var candidates = engine.TensorGather(sorted, new Tensor<int>(positions, new[] { positions.Length }), 0);
            boundaries[j] = engine.TensorUniqueConsecutive(candidates).ToVector();
        }

        return boundaries;
    }

    /// <summary>
    /// Bins features using precomputed boundaries.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="boundaries">Bin boundaries for each feature.</param>
    /// <returns>Binned feature matrix.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts continuous feature values to bin indices.
    /// A value of 5.3 might become bin 42 if that's where 5.3 falls in the boundaries.</para>
    /// </remarks>
    private Tensor<int>[] BinFeatures(Matrix<T> x, Vector<T>[] boundaries)
    {
        int n = x.Rows;
        int p = x.Columns;
        var binned = new Tensor<int>[p];
        var engine = Engine;
        for (int j = 0; j < p; j++)
        {
            // Bin = index of the first boundary >= value (an exact boundary match maps to its own index)
            binned[j] = n == 0 || boundaries[j].Length == 0
                ? new Tensor<int>(new[] { n })
                : engine.TensorSearchSorted(boundaries[j].AsTensor(), x.GetColumn(j).AsTensor(), false);
        }

        return binned;
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
    private Vector<T> ComputeInitialPrediction(Vector<T> y)
    {
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        var engine = Engine;
        var counts = new Tensor<T>(new[] { NumClasses });
        if (y.Length > 0)
        {
            // Unknown labels map to -1; shifting by one sends them to bin 0, which is dropped.
            var shifted = engine.TensorAddScalar(GetClassIndexTensor(y), 1);
            var binCounts = engine.TensorScatterAdd(new Tensor<T>(new[] { NumClasses + 1 }), shifted,
                FilledVector(y.Length, NumOps.One).AsTensor(), 0);
            counts = binCounts.Slice(0, 1, NumClasses + 1);
        }

        var fractions = engine.TensorDivideScalar(counts, NumOps.FromDouble(y.Length));

        if (NumClasses == 2)
        {
            // Binary: log-odds
            T p1 = fractions[1];
            p1 = NumOps.Compare(p1, NumOps.FromDouble(1 - 1e-7)) > 0 ? NumOps.FromDouble(1 - 1e-7) : p1;
            p1 = NumOps.Compare(p1, NumOps.FromDouble(1e-7)) < 0 ? NumOps.FromDouble(1e-7) : p1;
            return Vector<T>.Wrap(new[] { NumOps.Log(NumOps.Divide(p1, NumOps.Subtract(NumOps.One, p1))) });
        }

        // Multiclass: log probabilities
        return numOutputs == 0
            ? new Vector<T>(0)
            : engine.TensorLog(engine.TensorClampMin(fractions, NumOps.FromDouble(1e-7))).ToVector();
    }

    /// <summary>
    /// Computes gradients and hessians for gradient boosting.
    /// </summary>
    /// <param name="targets">One-hot targets [n_samples, n_classes].</param>
    /// <param name="predictions">Current raw predictions [n_samples, n_outputs].</param>
    /// <returns>Gradients and hessians [n_samples, n_outputs].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how wrong our predictions are.
    /// Hessians (second derivatives) tell us how confident we should be in our gradient estimates.
    /// Together they help us build trees that efficiently reduce the loss.</para>
    /// </remarks>
    private (Tensor<T> Gradients, Tensor<T> Hessians) ComputeGradientsAndHessians(Tensor<T> targets, Tensor<T> predictions)
    {
        var engine = Engine;
        int n = predictions.Shape[0];
        if (n == 0)
        {
            return (predictions, predictions);
        }

        // Binary cross-entropy uses sigmoid(raw) against the class-1 indicator; multiclass uses softmax
        Tensor<T> probabilities;
        Tensor<T> target;
        if (NumClasses == 2)
        {
            probabilities = engine.TensorSigmoid(predictions);
            target = targets.Slice(1, 1, 2);
        }
        else
        {
            probabilities = engine.TensorSoftmax(predictions, 1);
            target = targets;
        }

        var gradients = engine.TensorSubtract(probabilities, target);
        var hessians = engine.TensorClampMin(
            engine.TensorMultiply(probabilities, engine.ScalarMinusTensor(NumOps.One, probabilities)), NumOps.FromDouble(1e-7));
        return (gradients, hessians);
    }

    /// <summary>
    /// Builds a histogram-based decision tree.
    /// </summary>
    /// <param name="binnedX">Binned feature matrix.</param>
    /// <param name="gradients">Gradient values.</param>
    /// <param name="hessians">Hessian values.</param>
    /// <param name="depth">Current depth.</param>
    /// <param name="sampleIndices">Indices of samples in this node.</param>
    /// <returns>Root node of the tree.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This builds a tree by:
    /// <list type="number">
    /// <item>Building histograms of gradients and hessians for each feature</item>
    /// <item>Finding the best split by scanning histogram bins</item>
    /// <item>Recursively splitting until stopping criteria are met</item>
    /// </list>
    /// The key speedup is that we only evaluate O(bins) splits per feature instead of O(samples).
    /// </para>
    /// </remarks>
    private HistTree BuildHistTree(Tensor<int>[] binnedX, Tensor<T> gradients, Tensor<T> hessians,
        int depth, int[]? sampleIndices = null)
    {
        var engine = Engine;
        int n = gradients.Length;
        sampleIndices ??= Enumerable.Range(0, n).ToArray();
        int count = sampleIndices.Length;
        var indexTensor = new Tensor<int>(sampleIndices, new[] { count });
        var nodeGradients = count == n ? gradients : engine.TensorGather(gradients, indexTensor, 0);
        var nodeHessians = count == n ? hessians : engine.TensorGather(hessians, indexTensor, 0);

        // Compute leaf value
        T lambda = NumOps.FromDouble(_l2Regularization);
        T sumGrad = count == 0 ? NumOps.Zero : engine.TensorSum(nodeGradients);
        T sumHess = count == 0 ? NumOps.Zero : engine.TensorSum(nodeHessians);
        T leafValue = NumOps.Negate(NumOps.Divide(sumGrad, NumOps.Add(sumHess, lambda)));

        // Stopping conditions
        if (depth >= _maxDepth || count < 2 * _minSamplesLeaf)
        {
            return new HistTree { LeafValue = leafValue };
        }

        // Find best split (strictly positive gain; the first feature, then the first bin, wins ties)
        int bestFeature = -1;
        int bestBin = -1;
        var featureGains = new List<Tensor<T>>();
        var featureIds = new List<int>();
        var bestBinPerFeature = new List<int>();
        T countValue = NumOps.FromDouble(count);
        T parentVal = NumOps.Divide(NumOps.Multiply(sumGrad, sumGrad), NumOps.Add(sumHess, lambda));
        T minLeaf = NumOps.FromDouble(_minSamplesLeaf);
        T half = NumOps.FromDouble(0.5);

        for (int j = 0; j < NumFeatures; j++)
        {
            // Histogram for this feature
            int numBins = _binBoundaries is not null ? _binBoundaries[j].Length + 1 : _maxBins;
            if (numBins < 2)
            {
                continue;
            }

            var bins = count == n ? binnedX[j] : engine.TensorGather(binnedX[j], indexTensor, 0);
            var binGradSum = engine.TensorScatterAdd(new Tensor<T>(new[] { numBins }), bins, nodeGradients, 0);
            var binHessSum = engine.TensorScatterAdd(new Tensor<T>(new[] { numBins }), bins, nodeHessians, 0);
            var binCount = engine.TensorScatterAdd(new Tensor<T>(new[] { numBins }), bins, FilledVector(count, NumOps.One).AsTensor(), 0);

            // Left statistics for splits after bins 0..numBins-2
            var leftGrad = engine.TensorCumSum(binGradSum, 0).Slice(0, 0, numBins - 1);
            var leftHess = engine.TensorCumSum(binHessSum, 0).Slice(0, 0, numBins - 1);
            var leftCount = engine.TensorCumSum(binCount, 0).Slice(0, 0, numBins - 1);
            var rightGrad = engine.ScalarMinusTensor(sumGrad, leftGrad);
            var rightHess = engine.ScalarMinusTensor(sumHess, leftHess);
            var rightCount = engine.ScalarMinusTensor(countValue, leftCount);

            // gain = 0.5 * (L^2 / (LH + lambda) + R^2 / (RH + lambda) - P^2 / (PH + lambda))
            var leftVal = engine.TensorDivide(engine.TensorMultiply(leftGrad, leftGrad), engine.TensorAddScalar(leftHess, lambda));
            var rightVal = engine.TensorDivide(engine.TensorMultiply(rightGrad, rightGrad), engine.TensorAddScalar(rightHess, lambda));
            var gains = engine.TensorMultiplyScalar(engine.TensorSubtractScalar(engine.TensorAdd(leftVal, rightVal), parentVal), half);

            // Splits leaving fewer than minSamplesLeaf on either side are not allowed
            var allowed = engine.TensorMultiply(
                engine.TensorGreaterEqual(leftCount, FilledVector(numBins - 1, minLeaf).AsTensor()),
                engine.TensorGreaterEqual(rightCount, FilledVector(numBins - 1, minLeaf).AsTensor()));
            var candidates = engine.TensorWhere(allowed, gains, new Tensor<T>(new[] { numBins - 1 }));
            int bin = engine.TensorArgMax(candidates, 0).ToArray()[0];
            featureGains.Add(engine.TensorSlice(candidates, new[] { bin }, new[] { 1 }));
            featureIds.Add(j);
            bestBinPerFeature.Add(bin);
        }

        // First feature with the largest gain wins, provided that gain is strictly positive
        if (featureGains.Count > 0)
        {
            var allGains = engine.TensorConcatenate(featureGains.ToArray(), 0);
            int bestPosition = engine.TensorArgMax(allGains, 0).ToArray()[0];
            if (NumOps.GreaterThan(allGains[bestPosition], NumOps.Zero))
            {
                bestFeature = featureIds[bestPosition];
                bestBin = bestBinPerFeature[bestPosition];
            }
        }

        // No good split found
        if (bestFeature < 0)
        {
            return new HistTree { LeafValue = leafValue };
        }

        // Split samples
        var featureBins = binnedX[bestFeature].ToArray();
        var leftIndices = sampleIndices.Where(i => featureBins[i] <= bestBin).ToArray();
        var rightIndices = sampleIndices.Where(i => featureBins[i] > bestBin).ToArray();

        // Build child trees
        var leftTree = BuildHistTree(binnedX, gradients, hessians, depth + 1, leftIndices);
        var rightTree = BuildHistTree(binnedX, gradients, hessians, depth + 1, rightIndices);

        return new HistTree
        {
            FeatureIndex = bestFeature,
            BinThreshold = bestBin,
            LeftChild = leftTree,
            RightChild = rightTree
        };
    }

    /// <summary>
    /// Gets the prediction from a tree for a single sample.
    /// </summary>
    /// <param name="tree">The tree to evaluate.</param>
    /// <param name="binnedX">Binned features, one bin index tensor per feature.</param>
    /// <param name="rows">The number of samples.</param>
    /// <returns>The tree output for every sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Traverses the tree from root to leaf by comparing
    /// the sample's binned feature values to the split thresholds.</para>
    /// </remarks>
    private Tensor<T> PredictTree(HistTree tree, Tensor<int>[] binnedX, int rows)
    {
        // Route every sample to its leaf, then look the leaf values up
        var leafValues = new List<T>();
        var leafOfSample = new int[rows];
        AssignLeaves(tree, binnedX, Enumerable.Range(0, rows).ToArray(), leafValues, leafOfSample);
        if (rows == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        return Engine.TensorGather(Vector<T>.Wrap(leafValues.ToArray()).AsTensor(),
            new Tensor<int>(leafOfSample, new[] { rows }), 0);
    }

    /// <summary>
    /// Records, for every sample in <paramref name="samples"/>, the index of the leaf it reaches.
    /// </summary>
    private static void AssignLeaves(HistTree node, Tensor<int>[] binnedX, int[] samples, List<T> leafValues, int[] leafOfSample)
    {
        if (node.LeftChild is null || node.RightChild is null)
        {
            int leaf = leafValues.Count;
            leafValues.Add(node.LeafValue);
            foreach (int i in samples)
            {
                leafOfSample[i] = leaf;
            }

            return;
        }

        var featureBins = binnedX[node.FeatureIndex].ToArray();
        int threshold = node.BinThreshold;
        AssignLeaves(node.LeftChild, binnedX, samples.Where(i => featureBins[i] <= threshold).ToArray(), leafValues, leafOfSample);
        AssignLeaves(node.RightChild, binnedX, samples.Where(i => featureBins[i] > threshold).ToArray(), leafValues, leafOfSample);
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
        // Collect the feature of every split across all trees
        var splitFeatures = new List<int>();
        foreach (var tree in _trees)
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
    /// Counts feature usage in a tree for importance calculation.
    /// </summary>
    /// <param name="node">Tree node to process.</param>
    /// <param name="splitFeatures">Receives the feature index of every split.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Recursively traverses the tree and counts how often
    /// each feature is used for splits.</para>
    /// </remarks>
    private void CountFeatureUsage(HistTree node, List<int> splitFeatures)
    {
        if (node.LeftChild is null || node.RightChild is null)
        {
            return;
        }

        splitFeatures.Add(node.FeatureIndex);
        CountFeatureUsage(node.LeftChild, splitFeatures);
        CountFeatureUsage(node.RightChild, splitFeatures);
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
            { "LeafValue", (object?)node.LeafValue ?? throw new InvalidOperationException("Tree leaf value is null.") },
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
            LeafValue = data["LeafValue"] is JToken leaf ? leaf.ToObject<T>() ?? NumOps.Zero : NumOps.Zero
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
        public T LeafValue { get; set; } = MathHelper.GetNumericOperations<T>().Zero;
    }
}
