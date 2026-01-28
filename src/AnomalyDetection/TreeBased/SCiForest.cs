using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TreeBased;

/// <summary>
/// Detects anomalies using SCiForest (Sparse Clustering-Integrated Isolation Forest).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SCiForest improves on Isolation Forest by using sparse splits that
/// combine multiple features. This makes it better at detecting anomalies in subspaces and
/// handles high-dimensional data more effectively.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Build trees using sparse random projections for splits
/// 2. Each split uses a weighted combination of features
/// 3. Anomalies have shorter average path lengths
/// </para>
/// <para>
/// <b>When to use:</b>
/// - High-dimensional data
/// - When anomalies hide in subspaces
/// - When standard Isolation Forest doesn't perform well
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of trees: 100
/// - Max samples: 256
/// - Sparsity: 0.2 (20% of features per split)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Liu, F.T., Ting, K.M., Zhou, Z.H. (2012). "Isolation-Based Anomaly Detection."
/// TKDD 6(1), with sparse extensions.
/// </para>
/// </remarks>
public class SCiForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private readonly double _sparsity;
    private List<SCiTree>? _trees;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    public int NumTrees => _numTrees;

    /// <summary>
    /// Gets the maximum samples per tree.
    /// </summary>
    public int MaxSamples => _maxSamples;

    /// <summary>
    /// Gets the sparsity level for projections.
    /// </summary>
    public double Sparsity => _sparsity;

    /// <summary>
    /// Creates a new SCiForest anomaly detector.
    /// </summary>
    /// <param name="numTrees">Number of trees in the forest. Default is 100.</param>
    /// <param name="maxSamples">Maximum samples per tree. Default is 256.</param>
    /// <param name="sparsity">Fraction of features to use per split. Default is 0.2.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SCiForest(int numTrees = 100, int maxSamples = 256, double sparsity = 0.2,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (numTrees < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numTrees),
                "NumTrees must be at least 1. Recommended is 100.");
        }

        if (maxSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamples),
                "MaxSamples must be at least 1. Recommended is 256.");
        }

        if (sparsity <= 0 || sparsity > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(sparsity),
                "Sparsity must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.2.");
        }

        _numTrees = numTrees;
        _maxSamples = maxSamples;
        _sparsity = sparsity;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        int nSamples = Math.Min(_maxSamples, X.Rows);
        int maxDepth = (int)Math.Ceiling(Math.Log(nSamples, 2));

        // Convert data to double array
        var data = new double[X.Rows][];
        for (int i = 0; i < X.Rows; i++)
        {
            data[i] = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        _trees = new List<SCiTree>();

        // Build trees
        for (int t = 0; t < _numTrees; t++)
        {
            var random = new Random(_randomSeed + t);

            // Sample data
            var sampleIndices = SampleIndices(X.Rows, nSamples, random);
            var sampleData = sampleIndices.Select(i => data[i]).ToArray();

            // Build tree
            var tree = new SCiTree(_nFeatures, _sparsity, random);
            tree.Build(sampleData, 0, maxDepth);
            _trees.Add(tree);
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var scores = new Vector<T>(X.Rows);
        int nSamples = Math.Min(_maxSamples, X.Rows);

        // Expected path length for BST
        double c = nSamples > 2
            ? 2 * (Math.Log(nSamples - 1) + 0.5772156649) - (2.0 * (nSamples - 1) / nSamples)
            : nSamples == 2 ? 1 : 0;

        var trees = _trees;
        if (trees == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Average path length across all trees
            double avgPathLength = trees.Average(tree => tree.PathLength(point, 0));

            // Anomaly score: s = 2^(-avgPathLength/c)
            double score = c > 0 ? Math.Pow(2, -avgPathLength / c) : 0.5;
            scores[i] = NumOps.FromDouble(score);
        }

        return scores;
    }

    private int[] SampleIndices(int total, int sampleSize, Random random)
    {
        var indices = Enumerable.Range(0, total).ToList();
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        return indices.Take(sampleSize).ToArray();
    }

    private class SCiTree
    {
        private readonly int _nFeatures;
        private readonly double _sparsity;
        private readonly Random _random;
        private double[]? _sparseWeights;
        private double _threshold;
        private SCiTree? _left;
        private SCiTree? _right;
        private int _size;
        private bool _isLeaf;

        public SCiTree(int nFeatures, double sparsity, Random random)
        {
            _nFeatures = nFeatures;
            _sparsity = sparsity;
            _random = random;
            _isLeaf = true;
        }

        public void Build(double[][] data, int currentDepth, int maxDepth)
        {
            _size = data.Length;

            if (currentDepth >= maxDepth || data.Length <= 1)
            {
                _isLeaf = true;
                return;
            }

            // Generate sparse random projection
            _sparseWeights = new double[_nFeatures];
            int numActive = Math.Max(1, (int)(_nFeatures * _sparsity));

            var activeFeatures = Enumerable.Range(0, _nFeatures)
                .OrderBy(_ => _random.NextDouble())
                .Take(numActive)
                .ToArray();

            foreach (int f in activeFeatures)
            {
                // Random weight: +1 or -1
                _sparseWeights[f] = _random.NextDouble() < 0.5 ? 1 : -1;
            }

            // Compute projections
            var projections = data.Select(p => DotProduct(p, _sparseWeights)).ToArray();

            double minProj = projections.Min();
            double maxProj = projections.Max();

            if (Math.Abs(maxProj - minProj) < 1e-10)
            {
                _isLeaf = true;
                return;
            }

            _threshold = minProj + _random.NextDouble() * (maxProj - minProj);

            // Split data
            var leftData = new List<double[]>();
            var rightData = new List<double[]>();

            for (int i = 0; i < data.Length; i++)
            {
                if (projections[i] < _threshold)
                    leftData.Add(data[i]);
                else
                    rightData.Add(data[i]);
            }

            if (leftData.Count == 0 || rightData.Count == 0)
            {
                _isLeaf = true;
                return;
            }

            _isLeaf = false;
            _left = new SCiTree(_nFeatures, _sparsity, _random);
            _right = new SCiTree(_nFeatures, _sparsity, _random);

            _left.Build(leftData.ToArray(), currentDepth + 1, maxDepth);
            _right.Build(rightData.ToArray(), currentDepth + 1, maxDepth);
        }

        public double PathLength(double[] point, int currentDepth)
        {
            if (_isLeaf)
            {
                return currentDepth + EstimateC(_size);
            }

            var sparseWeights = _sparseWeights;
            var left = _left;
            var right = _right;

            if (sparseWeights == null || left == null || right == null)
            {
                return currentDepth + EstimateC(_size);
            }

            double projection = DotProduct(point, sparseWeights);

            if (projection < _threshold)
                return left.PathLength(point, currentDepth + 1);
            else
                return right.PathLength(point, currentDepth + 1);
        }

        private static double DotProduct(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        private static double EstimateC(int n)
        {
            if (n > 2)
                return 2 * (Math.Log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n);
            return n == 2 ? 1 : 0;
        }
    }
}
