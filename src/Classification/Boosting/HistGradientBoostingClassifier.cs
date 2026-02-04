using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

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
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HistGradientBoostingClassifier<T> : ClassifierBase<T>
{
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
    protected override ModelType GetModelType() => ModelType.HistGradientBoostingClassifier;

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

        // Step 1: Compute bin boundaries for each feature
        _binBoundaries = ComputeBinBoundaries(x);

        // Step 2: Bin the features
        var binnedX = BinFeatures(x, _binBoundaries);

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
        for (int iter = 0; iter < _nEstimators; iter++)
        {
            // Compute all gradients from the same predictions snapshot
            var grads = new double[numOutputs][];
            var hess = new double[numOutputs][];
            for (int k = 0; k < numOutputs; k++)
            {
                (grads[k], hess[k]) = ComputeGradientsAndHessians(y, predictions, k);
            }

            // Build trees for all classes
            var treesThisIter = new HistTree[numOutputs];
            for (int k = 0; k < numOutputs; k++)
            {
                var tree = BuildHistTree(binnedX, grads[k], hess[k], 0);
                treesThisIter[k] = tree;
                _trees.Add(tree);
            }

            // Update predictions for all classes together
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < numOutputs; k++)
                {
                    double treeOutput = PredictTree(treesThisIter[k], binnedX, i);
                    predictions[i, k] += _learningRate * treeOutput;
                }
            }
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

        var binnedX = BinFeatures(input, _binBoundaries);
        int numOutputs = NumClasses == 2 ? 1 : NumClasses;
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var rawPred = new double[numOutputs];
            for (int k = 0; k < numOutputs; k++)
            {
                rawPred[k] = _initialPrediction[k];
            }

            // Sum tree predictions
            for (int treeIdx = 0; treeIdx < _trees.Count; treeIdx++)
            {
                int k = treeIdx % numOutputs;
                double treeOutput = PredictTree(_trees[treeIdx], binnedX, i);
                rawPred[k] += _learningRate * treeOutput;
            }

            // Convert to class prediction
            if (NumClasses == 2)
            {
                double prob = 1.0 / (1.0 + Math.Exp(-rawPred[0]));
                predictions[i] = prob >= 0.5 ? ClassLabels[1] : ClassLabels[0];
            }
            else
            {
                // Softmax and argmax
                double maxVal = rawPred.Max();
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
                predictions[i] = ClassLabels[bestClass];
            }
        }

        return predictions;
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
    private double[][] ComputeBinBoundaries(Matrix<T> x)
    {
        int p = x.Columns;
        var boundaries = new double[p][];

        for (int j = 0; j < p; j++)
        {
            // Collect unique values with validation
            var values = new List<double>();
            for (int i = 0; i < x.Rows; i++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                if (double.IsNaN(val) || double.IsInfinity(val))
                {
                    throw new ArgumentException(
                        $"Missing/NaN/Infinity values are not supported. Found at row {i}, feature {j}. Please impute before training.",
                        nameof(x));
                }
                values.Add(val);
            }
            values.Sort();

            // Compute quantile boundaries
            int numBins = Math.Min(_maxBins, values.Count);
            var binBounds = new List<double>();

            for (int b = 1; b < numBins; b++)
            {
                int idx = (int)((double)b / numBins * values.Count);
                idx = Math.Min(idx, values.Count - 1);
                if (binBounds.Count == 0 || values[idx] > binBounds[^1])
                {
                    binBounds.Add(values[idx]);
                }
            }

            boundaries[j] = [.. binBounds];
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
    private int[,] BinFeatures(Matrix<T> x, double[][] boundaries)
    {
        int n = x.Rows;
        int p = x.Columns;
        var binned = new int[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double val = NumOps.ToDouble(x[i, j]);
                int bin = Array.BinarySearch(boundaries[j], val);
                if (bin < 0) bin = ~bin;
                binned[i, j] = bin;
            }
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
    /// <param name="y">Target labels.</param>
    /// <param name="predictions">Current predictions.</param>
    /// <param name="classIdx">Class index for multiclass problems.</param>
    /// <returns>Tuple of gradients and hessians.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradients tell us how wrong our predictions are.
    /// Hessians (second derivatives) tell us how confident we should be in our gradient estimates.
    /// Together they help us build trees that efficiently reduce the loss.</para>
    /// </remarks>
    private (double[] gradients, double[] hessians) ComputeGradientsAndHessians(
        Vector<T> y, double[,] predictions, int classIdx)
    {
        int n = y.Length;
        var gradients = new double[n];
        var hessians = new double[n];

        if (NumClasses == 2)
        {
            // Binary cross-entropy gradients
            for (int i = 0; i < n; i++)
            {
                double target = GetClassIndexFromLabel(y[i]) == 1 ? 1.0 : 0.0;
                double pred = 1.0 / (1.0 + Math.Exp(-predictions[i, 0]));
                gradients[i] = pred - target;
                hessians[i] = pred * (1 - pred);
                hessians[i] = Math.Max(hessians[i], 1e-7);
            }
        }
        else
        {
            // Multiclass cross-entropy gradients
            for (int i = 0; i < n; i++)
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
                double target = GetClassIndexFromLabel(y[i]) == classIdx ? 1.0 : 0.0;

                gradients[i] = pred - target;
                hessians[i] = pred * (1 - pred);
                hessians[i] = Math.Max(hessians[i], 1e-7);
            }
        }

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
    private HistTree BuildHistTree(int[,] binnedX, double[] gradients, double[] hessians,
        int depth, int[]? sampleIndices = null)
    {
        int n = binnedX.GetLength(0);
        sampleIndices ??= Enumerable.Range(0, n).ToArray();

        // Compute leaf value
        double sumGrad = 0;
        double sumHess = 0;
        foreach (int i in sampleIndices)
        {
            sumGrad += gradients[i];
            sumHess += hessians[i];
        }

        double leafValue = -sumGrad / (sumHess + _l2Regularization);

        // Stopping conditions
        if (depth >= _maxDepth || sampleIndices.Length < 2 * _minSamplesLeaf)
        {
            return new HistTree { LeafValue = leafValue };
        }

        // Find best split
        int bestFeature = -1;
        int bestBin = -1;
        double bestGain = 0;

        for (int j = 0; j < NumFeatures; j++)
        {
            // Build histogram for this feature
            int numBins = _binBoundaries is not null ? _binBoundaries[j].Length + 1 : _maxBins;
            var binGradSum = new double[numBins];
            var binHessSum = new double[numBins];
            var binCount = new int[numBins];

            foreach (int i in sampleIndices)
            {
                int bin = binnedX[i, j];
                binGradSum[bin] += gradients[i];
                binHessSum[bin] += hessians[i];
                binCount[bin]++;
            }

            // Scan for best split
            double leftGradSum = 0;
            double leftHessSum = 0;
            int leftCount = 0;

            for (int b = 0; b < numBins - 1; b++)
            {
                leftGradSum += binGradSum[b];
                leftHessSum += binHessSum[b];
                leftCount += binCount[b];

                double rightGradSum = sumGrad - leftGradSum;
                double rightHessSum = sumHess - leftHessSum;
                int rightCount = sampleIndices.Length - leftCount;

                if (leftCount < _minSamplesLeaf || rightCount < _minSamplesLeaf)
                {
                    continue;
                }

                // Compute gain
                double leftVal = leftGradSum * leftGradSum / (leftHessSum + _l2Regularization);
                double rightVal = rightGradSum * rightGradSum / (rightHessSum + _l2Regularization);
                double parentVal = sumGrad * sumGrad / (sumHess + _l2Regularization);
                double gain = 0.5 * (leftVal + rightVal - parentVal);

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = j;
                    bestBin = b;
                }
            }
        }

        // No good split found
        if (bestFeature < 0)
        {
            return new HistTree { LeafValue = leafValue };
        }

        // Split samples
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        foreach (int i in sampleIndices)
        {
            if (binnedX[i, bestFeature] <= bestBin)
            {
                leftIndices.Add(i);
            }
            else
            {
                rightIndices.Add(i);
            }
        }

        // Build child trees
        var leftTree = BuildHistTree(binnedX, gradients, hessians, depth + 1, [.. leftIndices]);
        var rightTree = BuildHistTree(binnedX, gradients, hessians, depth + 1, [.. rightIndices]);

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
    /// <param name="binnedX">Binned feature matrix.</param>
    /// <param name="sampleIdx">Index of the sample.</param>
    /// <returns>Tree prediction value.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Traverses the tree from root to leaf by comparing
    /// the sample's binned feature values to the split thresholds.</para>
    /// </remarks>
    private double PredictTree(HistTree tree, int[,] binnedX, int sampleIdx)
    {
        var node = tree;
        while (node.LeftChild is not null && node.RightChild is not null)
        {
            if (binnedX[sampleIdx, node.FeatureIndex] <= node.BinThreshold)
            {
                node = node.LeftChild;
            }
            else
            {
                node = node.RightChild;
            }
        }
        return node.LeafValue;
    }

    /// <summary>
    /// Gets the model parameters.
    /// </summary>
    /// <returns>Vector containing serialized tree parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trees are complex structures that don't fit neatly into a
    /// parameter vector. This returns a simplified representation for compatibility.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        // For tree-based models, parameters don't fit the typical vector format
        // Return a placeholder with tree count
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_trees.Count) };
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Parameter vector (not fully supported for tree models).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree-based models are better loaded via serialization
    /// than through a parameter vector. This is a limited implementation.</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Limited support for tree models - use serialization instead
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
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>New instance with same hyperparameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates an untrained copy with the same settings.</para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new HistGradientBoostingClassifier<T>(_maxBins, _maxDepth, _nEstimators,
            _learningRate, _minSamplesLeaf, _l2Regularization);
    }

    /// <summary>
    /// Computes gradients for the model parameters.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <returns>Gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gradient boosting models are trained iteratively, not by
    /// computing a single gradient over all parameters. This returns a placeholder gradient.</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Tree models don't use gradient-based parameter updates in the traditional sense
        return new Vector<T>(1) { [0] = NumOps.Zero };
    }

    /// <summary>
    /// Applies gradients to update model parameters.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Tree models are not updated via gradient descent on parameters.
    /// They're trained by building trees iteratively.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree models don't support gradient-based parameter updates
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
        /// Bin threshold for the split (values <= threshold go left).
        /// </summary>
        public int BinThreshold { get; set; }

        /// <summary>
        /// Left child node (values <= threshold).
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
