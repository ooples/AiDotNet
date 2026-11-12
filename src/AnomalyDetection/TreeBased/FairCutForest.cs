using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TreeBased;

/// <summary>
/// Implements Fair-Cut Forest (FCF) for anomaly detection with balanced tree construction.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Fair-Cut Forest improves on Isolation Forest by using balanced
/// tree construction. Instead of random splits, it selects splits that more evenly
/// divide the data, leading to more consistent isolation of anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Building trees with "fair" splits that balance the data more evenly
/// 2. Using importance-weighted feature selection based on data spread
/// 3. Computing anomaly scores based on average path length to isolation
/// </para>
/// <para>
/// <b>When to use:</b>
/// - High-dimensional data where standard Isolation Forest struggles
/// - When you need more consistent anomaly rankings
/// - Datasets with features of varying importance
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of trees: 100
/// - Max samples: 256
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Inspired by improvements to Isolation Forest for fair/balanced splits.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Robust Random Cut Forest Based Anomaly Detection on Streams", "https://doi.org/10.1145/2806416.2806568")]
public partial class FairCutForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private List<FCFTree>? _trees;
    private int _inputDim;
    private int _effectiveMaxSamples;

    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    public int NumTrees => _numTrees;

    /// <summary>
    /// Gets the maximum samples used per tree.
    /// </summary>
    public int MaxSamples => _maxSamples;

    /// <summary>
    /// Creates a new Fair-Cut Forest anomaly detector.
    /// </summary>
    /// <param name="numTrees">Number of trees in the forest. Default is 100.</param>
    /// <param name="maxSamples">Maximum samples per tree. Default is 256.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public FairCutForest(int numTrees = 100, int maxSamples = 256,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numTrees < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numTrees),
                "Number of trees must be at least 1. Recommended is 100.");
        }

        if (maxSamples < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamples),
                "Max samples must be at least 2. Recommended is 256.");
        }

        _numTrees = numTrees;
        _maxSamples = maxSamples;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;
        _effectiveMaxSamples = Math.Min(_maxSamples, n);

        if (_effectiveMaxSamples < 2)
        {
            throw new ArgumentException(
                $"Fair-Cut Forest requires at least 2 samples. Got {n} samples.",
                nameof(X));
        }

        // Compute feature importances based on spread
        var featureImportances = ComputeFeatureImportances(X);

        // Build trees
        int maxDepth = (int)Math.Ceiling(Math.Log(_effectiveMaxSamples) / Math.Log(2));
        _trees = new List<FCFTree>(_numTrees);
        var dataTensor = AnomalyTensorOps.ToTensor(X);

        for (int t = 0; t < _numTrees; t++)
        {
            // Sample data
            var sampleIndices = SampleIndices(n, _effectiveMaxSamples);
            var sample = AnomalyTensorOps.SelectRows(dataTensor, sampleIndices).ToMatrix();

            // Build tree with fair cuts
            var tree = BuildFairCutTree(sample, featureImportances, maxDepth);
            _trees.Add(tree);
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Vector<T> ComputeFeatureImportances(Matrix<T> data)
    {
        int d = data.Columns;
        var engine = Engine;
        var values = AnomalyTensorOps.ToTensor(data);

        // Spread (max - min) of each feature
        var importances = engine.TensorSubtract(
            engine.ReduceMax(values, new[] { 0 }, false), engine.TensorAmin(values, new[] { 0 }, false));

        // Normalize importances
        T sum = engine.TensorSum(importances);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            return engine.TensorDivideScalar(importances, sum).ToVector();
        }

        return engine.Fill(d, NumOps.Divide(NumOps.One, NumOps.FromDouble(d)));
    }

    private int[] SampleIndices(int n, int sampleSize)
    {
        var indices = Enumerable.Range(0, n).ToArray();

        // Fisher-Yates shuffle for first sampleSize elements
        for (int i = 0; i < sampleSize; i++)
        {
            int j = _random.Next(i, n);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices.Take(sampleSize).ToArray();
    }

    private FCFTree BuildFairCutTree(Matrix<T> data, Vector<T> featureImportances, int maxDepth)
    {
        return BuildFairCutTreeRecursive(data, featureImportances, 0, maxDepth);
    }

    private FCFTree BuildFairCutTreeRecursive(Matrix<T> data, Vector<T> featureImportances,
        int currentDepth, int maxDepth)
    {
        var node = new FCFTree();

        if (data.Rows <= 1 || currentDepth >= maxDepth)
        {
            node.IsLeaf = true;
            node.Size = data.Rows;
            return node;
        }

        // Select feature based on importance-weighted probability
        int selectedFeature = SelectFeatureByImportance(featureImportances);

        // Find min and max for selected feature
        var engine = Engine;
        var dataTensor = AnomalyTensorOps.ToTensor(data);
        var column = engine.TensorSliceAxis(dataTensor, 1, selectedFeature);
        T min = engine.TensorMinValue(column);
        T max = engine.TensorMaxValue(column);

        T range = NumOps.Subtract(max, min);
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(NumOps.Abs(range), eps))
        {
            node.IsLeaf = true;
            node.Size = data.Rows;
            return node;
        }

        // Fair cut: use median instead of random split for more balanced trees
        var (sortedColumn, _) = engine.TensorSort(column, 0, false);
        T splitValue = sortedColumn[data.Rows / 2]; // Median

        // Add small random perturbation to avoid identical splits
        T perturbation = NumOps.Multiply(NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1), range);
        splitValue = NumOps.Add(splitValue, perturbation);
        if (NumOps.LessThan(splitValue, min)) splitValue = min;
        if (NumOps.GreaterThan(splitValue, max)) splitValue = max;

        node.SplitFeature = selectedFeature;
        node.SplitValue = splitValue;

        // Split data: values below the split go left (order preserved)
        var goesLeft = engine.TensorLessThan(column, splitValue);
        var leftIndices = AnomalyTensorOps.NonzeroIndices(goesLeft);
        var rightIndices = AnomalyTensorOps.NonzeroIndices(engine.ScalarMinusTensor(NumOps.One, goesLeft));

        if (leftIndices.Length == 0 || rightIndices.Length == 0)
        {
            node.IsLeaf = true;
            node.Size = data.Rows;
            return node;
        }

        var leftData = AnomalyTensorOps.SelectRows(dataTensor, leftIndices).ToMatrix();
        var rightData = AnomalyTensorOps.SelectRows(dataTensor, rightIndices).ToMatrix();

        node.Left = BuildFairCutTreeRecursive(leftData, featureImportances, currentDepth + 1, maxDepth);
        node.Right = BuildFairCutTreeRecursive(rightData, featureImportances, currentDepth + 1, maxDepth);

        return node;
    }

    private int SelectFeatureByImportance(Vector<T> importances)
    {
        // Feature selection uses Random.NextDouble() - this is a double boundary
        double r = _random.NextDouble();

        // The first feature whose running importance total reaches r (the last one if none does)
        var engine = Engine;
        var cumulative = engine.TensorCumSum(importances.AsTensor(), 0);
        var target = Vector<T>.FromArray(new[] { NumOps.FromDouble(r) }).AsTensor();
        int position = engine.TensorSearchSorted(cumulative, target, false).ToArray()[0];
        return Math.Min(position, importances.Length - 1);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var trees = _trees;
        if (trees == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        T cn = NumOps.FromDouble(ComputeCn(_effectiveMaxSamples));
        T nTreesT = NumOps.FromDouble(trees.Count);

        if (!NumOps.GreaterThan(cn, NumOps.Zero))
        {
            return Engine.Fill(X.Rows, NumOps.FromDouble(0.5));
        }

        // Average path length across all trees (summed in tree order for every point)
        var engine = Engine;
        var columns = AnomalyTensorOps.Columns(X);
        var allRows = AnomalyTensorOps.Range(X.Rows);
        var totalPathLength = Tensor<T>.CreateZeros(new[] { X.Rows });
        foreach (var tree in trees)
        {
            var leafRows = new List<Tensor<int>>();
            var leafValues = new List<Tensor<T>>();
            RoutePathLengths(columns, allRows, tree, 0, leafRows, leafValues);
            totalPathLength = engine.TensorAdd(totalPathLength, AnomalyTensorOps.AssembleByRow(leafRows, leafValues));
        }

        var avgPathLength = engine.TensorDivideScalar(totalPathLength, nTreesT);

        // Anomaly score: 2^(-avgPathLength / c(n))
        return engine.TensorExp2(engine.TensorNegate(engine.TensorDivideScalar(avgPathLength, cn))).ToVector();
    }

    // Routes the given rows down a tree as index sets; each leaf contributes depth + c(size) for the
    // rows that reach it.
    private void RoutePathLengths(Tensor<T>[] columns, Tensor<int> rows, FCFTree node, int currentDepth,
        List<Tensor<int>> leafRows, List<Tensor<T>> leafValues)
    {
        if (rows.Length == 0)
        {
            return;
        }

        if (node.IsLeaf)
        {
            leafRows.Add(rows);
            leafValues.Add(Engine.Fill(rows.Length, NumOps.FromDouble(currentDepth + ComputeCn(node.Size))).AsTensor());
            return;
        }

        var engine = Engine;
        var goesLeft = engine.TensorLessThan(engine.TensorGather(columns[node.SplitFeature], rows, 0), node.SplitValue);
        var leftRows = AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.One);
        var rightRows = AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.Zero);
        if (leftRows.Length > 0)
        {
            RoutePathLengths(columns, leftRows,
                node.Left ?? throw new InvalidOperationException("Corrupt tree: non-leaf node missing left child."),
                currentDepth + 1, leafRows, leafValues);
        }

        if (rightRows.Length > 0)
        {
            RoutePathLengths(columns, rightRows,
                node.Right ?? throw new InvalidOperationException("Corrupt tree: non-leaf node missing right child."),
                currentDepth + 1, leafRows, leafValues);
        }
    }

    private static double ComputeCn(int n)
    {
        if (n <= 1) return 0;
        if (n == 2) return 1;

        double h = Math.Log(n - 1) + 0.5772156649; // Euler-Mascheroni constant
        return 2 * h - 2.0 * (n - 1) / n;
    }

    private class FCFTree
    {
        public bool IsLeaf { get; set; }
        public int Size { get; set; }
        public int SplitFeature { get; set; }
        public T SplitValue { get; set; } = MathHelper.GetNumericOperations<T>().Zero;
        public FCFTree? Left { get; set; }
        public FCFTree? Right { get; set; }
    }
}
