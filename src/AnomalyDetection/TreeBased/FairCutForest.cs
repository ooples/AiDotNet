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
public class FairCutForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private List<FCFTree>? _trees;
    private int _inputDim;

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
        int effectiveMaxSamples = Math.Min(_maxSamples, n);

        if (effectiveMaxSamples < 2)
        {
            throw new ArgumentException(
                $"Fair-Cut Forest requires at least 2 samples. Got {n} samples.",
                nameof(X));
        }

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Compute feature importances based on spread
        var featureImportances = ComputeFeatureImportances(data);

        // Build trees
        int maxDepth = (int)Math.Ceiling(Math.Log(effectiveMaxSamples) / Math.Log(2));
        _trees = new List<FCFTree>(_numTrees);

        for (int t = 0; t < _numTrees; t++)
        {
            // Sample data
            var sampleIndices = SampleIndices(n, effectiveMaxSamples);
            var sample = sampleIndices.Select(i => data[i]).ToArray();

            // Build tree with fair cuts
            var tree = BuildFairCutTree(sample, featureImportances, maxDepth);
            _trees.Add(tree);
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[] ComputeFeatureImportances(double[][] data)
    {
        int d = data[0].Length;
        var importances = new double[d];

        for (int j = 0; j < d; j++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;

            foreach (var row in data)
            {
                if (row[j] < min) min = row[j];
                if (row[j] > max) max = row[j];
            }

            // Importance is the range (spread) of the feature
            importances[j] = max - min;
        }

        // Normalize importances
        double sum = importances.Sum();
        if (sum > 0)
        {
            for (int j = 0; j < d; j++)
            {
                importances[j] /= sum;
            }
        }
        else
        {
            // Uniform if all features have zero range
            for (int j = 0; j < d; j++)
            {
                importances[j] = 1.0 / d;
            }
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

    private FCFTree BuildFairCutTree(double[][] data, double[] featureImportances, int maxDepth)
    {
        return BuildFairCutTreeRecursive(data, featureImportances, 0, maxDepth);
    }

    private FCFTree BuildFairCutTreeRecursive(double[][] data, double[] featureImportances,
        int currentDepth, int maxDepth)
    {
        var node = new FCFTree();

        if (data.Length <= 1 || currentDepth >= maxDepth)
        {
            node.IsLeaf = true;
            node.Size = data.Length;
            return node;
        }

        int d = data[0].Length;

        // Select feature based on importance-weighted probability
        int selectedFeature = SelectFeatureByImportance(featureImportances);

        // Find min and max for selected feature
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (var row in data)
        {
            if (row[selectedFeature] < min) min = row[selectedFeature];
            if (row[selectedFeature] > max) max = row[selectedFeature];
        }

        if (Math.Abs(max - min) < 1e-10)
        {
            node.IsLeaf = true;
            node.Size = data.Length;
            return node;
        }

        // Fair cut: use median instead of random split for more balanced trees
        var values = data.Select(row => row[selectedFeature]).OrderBy(v => v).ToArray();
        double splitValue = values[values.Length / 2]; // Median

        // Add small random perturbation to avoid identical splits
        splitValue += (_random.NextDouble() - 0.5) * (max - min) * 0.1;
        splitValue = Math.Max(min, Math.Min(max, splitValue));

        node.SplitFeature = selectedFeature;
        node.SplitValue = splitValue;

        // Split data
        var leftData = data.Where(row => row[selectedFeature] < splitValue).ToArray();
        var rightData = data.Where(row => row[selectedFeature] >= splitValue).ToArray();

        // Handle edge case where all data goes to one side
        if (leftData.Length == 0 || rightData.Length == 0)
        {
            node.IsLeaf = true;
            node.Size = data.Length;
            return node;
        }

        node.Left = BuildFairCutTreeRecursive(leftData, featureImportances, currentDepth + 1, maxDepth);
        node.Right = BuildFairCutTreeRecursive(rightData, featureImportances, currentDepth + 1, maxDepth);

        return node;
    }

    private int SelectFeatureByImportance(double[] importances)
    {
        double r = _random.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < importances.Length; i++)
        {
            cumulative += importances[i];
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

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Average path length across all trees
            double avgPathLength = 0;
            foreach (var tree in trees)
            {
                avgPathLength += ComputePathLength(point, tree, 0);
            }
            avgPathLength /= trees.Count;

            // Anomaly score: 2^(-avgPathLength / c(n))
            // where c(n) is the average path length in an unsuccessful search in a BST
            double cn = ComputeCn(_maxSamples);
            double score = Math.Pow(2, -avgPathLength / cn);

            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private double ComputePathLength(double[] point, FCFTree node, int currentDepth)
    {
        if (node.IsLeaf)
        {
            return currentDepth + ComputeCn(node.Size);
        }

        if (point[node.SplitFeature] < node.SplitValue)
        {
            var left = node.Left;
            return left != null
                ? ComputePathLength(point, left, currentDepth + 1)
                : currentDepth + 1;
        }
        else
        {
            var right = node.Right;
            return right != null
                ? ComputePathLength(point, right, currentDepth + 1)
                : currentDepth + 1;
        }
    }

    private static double ComputeCn(int n)
    {
        if (n <= 1) return 0;
        if (n == 2) return 1;

        // c(n) = 2 * H(n-1) - 2*(n-1)/n
        // where H(i) is the harmonic number
        double h = Math.Log(n - 1) + 0.5772156649; // Euler-Mascheroni constant
        return 2 * h - 2.0 * (n - 1) / n;
    }

    private class FCFTree
    {
        public bool IsLeaf { get; set; }
        public int Size { get; set; }
        public int SplitFeature { get; set; }
        public double SplitValue { get; set; }
        public FCFTree? Left { get; set; }
        public FCFTree? Right { get; set; }
    }
}
