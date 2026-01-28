using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TreeBased;

/// <summary>
/// Detects anomalies using Extended Isolation Forest (EIF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Extended Isolation Forest improves on the original Isolation Forest by
/// using hyperplane cuts instead of axis-parallel cuts. This eliminates biases that occur when
/// anomalies align with the axes, making detection more robust.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Build trees using random hyperplane cuts (instead of axis-parallel)
/// 2. Each cut is defined by a random normal vector and intercept
/// 3. Anomalies still have shorter path lengths on average
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Same use cases as Isolation Forest
/// - When anomalies may align with coordinate axes
/// - When you need more robust splits
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Extension level: Full (use all dimensions)
/// - Number of trees: 100
/// - Max samples: 256
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Hariri, S., Kind, M.C., Brunner, R.J. (2019). "Extended Isolation Forest." IEEE TKDE.
/// </para>
/// </remarks>
public class ExtendedIsolationForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private readonly int _extensionLevel;
    private List<ExtendedIsolationTree>? _trees;
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
    /// Gets the extension level (number of dimensions for hyperplane cuts).
    /// </summary>
    public int ExtensionLevel => _extensionLevel;

    /// <summary>
    /// Creates a new Extended Isolation Forest anomaly detector.
    /// </summary>
    /// <param name="numTrees">Number of trees in the forest. Default is 100.</param>
    /// <param name="maxSamples">Maximum samples per tree. Default is 256.</param>
    /// <param name="extensionLevel">
    /// Number of dimensions for hyperplane cuts. 0 means axis-parallel (standard IF),
    /// -1 means full extension (all dimensions). Default is -1.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ExtendedIsolationForest(int numTrees = 100, int maxSamples = 256, int extensionLevel = -1,
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

        _numTrees = numTrees;
        _maxSamples = maxSamples;
        _extensionLevel = extensionLevel;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        int effectiveExtension = _extensionLevel == -1 ? _nFeatures - 1 : Math.Min(_extensionLevel, _nFeatures - 1);

        int nSamples = Math.Min(_maxSamples, X.Rows);
        int maxDepth = (int)Math.Ceiling(Math.Log(nSamples, 2));

        _trees = new List<ExtendedIsolationTree>();

        // Convert data to double array for tree building
        var data = new double[X.Rows][];
        for (int i = 0; i < X.Rows; i++)
        {
            data[i] = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Build trees
        for (int t = 0; t < _numTrees; t++)
        {
            var random = new Random(_randomSeed + t);

            // Sample data
            var sampleIndices = SampleIndices(X.Rows, nSamples, random);
            var sampleData = sampleIndices.Select(i => data[i]).ToArray();

            // Build tree
            var tree = new ExtendedIsolationTree(effectiveExtension, random);
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

        var trees = _trees;
        if (trees == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int nSamples = Math.Min(_maxSamples, X.Rows);

        // Expected path length for BST
        double c = nSamples > 2
            ? 2 * (Math.Log(nSamples - 1) + 0.5772156649) - (2.0 * (nSamples - 1) / nSamples)
            : nSamples == 2 ? 1 : 0;

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
            // Higher is more anomalous (we want this)
            double score = c > 0 ? Math.Pow(2, -avgPathLength / c) : 0.5;

            // Invert to match our convention (higher = more anomalous)
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

    private class ExtendedIsolationTree
    {
        private readonly int _extensionLevel;
        private readonly Random _random;
        private double[]? _normal;
        private double _intercept;
        private ExtendedIsolationTree? _left;
        private ExtendedIsolationTree? _right;
        private int _size;
        private bool _isLeaf;

        public ExtendedIsolationTree(int extensionLevel, Random random)
        {
            _extensionLevel = extensionLevel;
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

            int nFeatures = data[0].Length;

            // Generate random hyperplane normal
            _normal = new double[nFeatures];

            if (_extensionLevel == 0)
            {
                // Axis-parallel (standard IF)
                int splitDim = _random.Next(nFeatures);
                _normal[splitDim] = 1;
            }
            else
            {
                // Extended: random direction
                // Pick extensionLevel + 1 random dimensions
                var dims = Enumerable.Range(0, nFeatures)
                    .OrderBy(_ => _random.NextDouble())
                    .Take(Math.Min(_extensionLevel + 1, nFeatures))
                    .ToArray();

                foreach (int d in dims)
                {
                    // Random Gaussian component
                    double u1 = 1.0 - _random.NextDouble();
                    double u2 = 1.0 - _random.NextDouble();
                    _normal[d] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                }

                // Normalize
                double norm = Math.Sqrt(_normal.Sum(n => n * n));
                if (norm > 1e-10)
                {
                    for (int i = 0; i < nFeatures; i++)
                    {
                        _normal[i] /= norm;
                    }
                }
            }

            // Compute projections
            var projections = data.Select(p => DotProduct(p, _normal)).ToArray();

            // Random intercept between min and max projection
            double minProj = projections.Min();
            double maxProj = projections.Max();

            if (Math.Abs(maxProj - minProj) < 1e-10)
            {
                _isLeaf = true;
                return;
            }

            _intercept = minProj + _random.NextDouble() * (maxProj - minProj);

            // Split data
            var leftData = new List<double[]>();
            var rightData = new List<double[]>();

            for (int i = 0; i < data.Length; i++)
            {
                if (projections[i] < _intercept)
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
            _left = new ExtendedIsolationTree(_extensionLevel, _random);
            _right = new ExtendedIsolationTree(_extensionLevel, _random);

            _left.Build(leftData.ToArray(), currentDepth + 1, maxDepth);
            _right.Build(rightData.ToArray(), currentDepth + 1, maxDepth);
        }

        public double PathLength(double[] point, int currentDepth)
        {
            if (_isLeaf)
            {
                // Approximate remaining path length for unbuilt tree
                return currentDepth + EstimateC(_size);
            }

            var normal = _normal;
            var left = _left;
            var right = _right;

            if (normal == null || left == null || right == null)
            {
                return currentDepth + EstimateC(_size);
            }

            double projection = DotProduct(point, normal);

            if (projection < _intercept)
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
