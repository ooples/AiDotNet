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
    [ModelPaper("Robust Random Cut Forest Based Anomaly Detection on Streams", "https://doi.org/10.1145/2806416.2806568")]
public class FairCutForest<T> : AnomalyDetectorBase<T>
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

        for (int t = 0; t < _numTrees; t++)
        {
            // Sample data
            var sampleIndices = SampleIndices(n, _effectiveMaxSamples);
            var sample = new Matrix<T>(sampleIndices.Length, _inputDim);
            for (int si = 0; si < sampleIndices.Length; si++)
            {
                for (int j = 0; j < _inputDim; j++)
                {
                    sample[si, j] = X[sampleIndices[si], j];
                }
            }

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
        var importances = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            T min = NumOps.MaxValue;
            T max = NumOps.MinValue;

            for (int i = 0; i < data.Rows; i++)
            {
                if (NumOps.LessThan(data[i, j], min)) min = data[i, j];
                if (NumOps.GreaterThan(data[i, j], max)) max = data[i, j];
            }

            importances[j] = NumOps.Subtract(max, min);
        }

        // Normalize importances
        T sum = NumOps.Zero;
        for (int j = 0; j < d; j++) sum = NumOps.Add(sum, importances[j]);

        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int j = 0; j < d; j++) importances[j] = NumOps.Divide(importances[j], sum);
        }
        else
        {
            T uniform = NumOps.Divide(NumOps.One, NumOps.FromDouble(d));
            for (int j = 0; j < d; j++) importances[j] = uniform;
        }

        return importances;
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
        T min = NumOps.MaxValue;
        T max = NumOps.MinValue;

        for (int i = 0; i < data.Rows; i++)
        {
            if (NumOps.LessThan(data[i, selectedFeature], min)) min = data[i, selectedFeature];
            if (NumOps.GreaterThan(data[i, selectedFeature], max)) max = data[i, selectedFeature];
        }

        T range = NumOps.Subtract(max, min);
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(NumOps.Abs(range), eps))
        {
            node.IsLeaf = true;
            node.Size = data.Rows;
            return node;
        }

        // Fair cut: use median instead of random split for more balanced trees
        var values = Enumerable.Range(0, data.Rows)
            .Select(i => data[i, selectedFeature])
            .OrderBy(v => NumOps.ToDouble(v))
            .ToArray();
        T splitValue = values[values.Length / 2]; // Median

        // Add small random perturbation to avoid identical splits
        T perturbation = NumOps.Multiply(NumOps.FromDouble((_random.NextDouble() - 0.5) * 0.1), range);
        splitValue = NumOps.Add(splitValue, perturbation);
        if (NumOps.LessThan(splitValue, min)) splitValue = min;
        if (NumOps.GreaterThan(splitValue, max)) splitValue = max;

        node.SplitFeature = selectedFeature;
        node.SplitValue = splitValue;

        // Split data
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();
        for (int i = 0; i < data.Rows; i++)
        {
            if (NumOps.LessThan(data[i, selectedFeature], splitValue))
                leftIndices.Add(i);
            else
                rightIndices.Add(i);
        }

        if (leftIndices.Count == 0 || rightIndices.Count == 0)
        {
            node.IsLeaf = true;
            node.Size = data.Rows;
            return node;
        }

        var leftData = ExtractRows(data, leftIndices);
        var rightData = ExtractRows(data, rightIndices);

        node.Left = BuildFairCutTreeRecursive(leftData, featureImportances, currentDepth + 1, maxDepth);
        node.Right = BuildFairCutTreeRecursive(rightData, featureImportances, currentDepth + 1, maxDepth);

        return node;
    }

    private static Matrix<T> ExtractRows(Matrix<T> data, List<int> indices)
    {
        var result = new Matrix<T>(indices.Count, data.Columns);
        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[indices[i], j];
            }
        }
        return result;
    }

    private int SelectFeatureByImportance(Vector<T> importances)
    {
        // Feature selection uses Random.NextDouble() - this is a double boundary
        double r = _random.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < importances.Length; i++)
        {
            cumulative += NumOps.ToDouble(importances[i]);
            if (r <= cumulative)
            {
                return i;
            }
        }

        return importances.Length - 1;
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

        var scores = new Vector<T>(X.Rows);
        T cn = NumOps.FromDouble(ComputeCn(_effectiveMaxSamples));
        T two = NumOps.FromDouble(2);
        T nTreesT = NumOps.FromDouble(trees.Count);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Average path length across all trees
            T avgPathLength = NumOps.Zero;
            foreach (var tree in trees)
            {
                avgPathLength = NumOps.Add(avgPathLength, ComputePathLength(point, tree, 0));
            }
            avgPathLength = NumOps.Divide(avgPathLength, nTreesT);

            // Anomaly score: 2^(-avgPathLength / c(n))
            T score = NumOps.GreaterThan(cn, NumOps.Zero)
                ? NumOps.Power(two, NumOps.Negate(NumOps.Divide(avgPathLength, cn)))
                : NumOps.FromDouble(0.5);

            scores[i] = score;
        }

        return scores;
    }

    private T ComputePathLength(Vector<T> point, FCFTree node, int currentDepth)
    {
        if (node.IsLeaf)
        {
            return NumOps.FromDouble(currentDepth + ComputeCn(node.Size));
        }

        if (NumOps.LessThan(point[node.SplitFeature], node.SplitValue))
        {
            if (node.Left is null)
                throw new InvalidOperationException("Corrupt tree: non-leaf node missing left child.");
            return ComputePathLength(point, node.Left, currentDepth + 1);
        }
        else
        {
            if (node.Right is null)
                throw new InvalidOperationException("Corrupt tree: non-leaf node missing right child.");
            return ComputePathLength(point, node.Right, currentDepth + 1);
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
