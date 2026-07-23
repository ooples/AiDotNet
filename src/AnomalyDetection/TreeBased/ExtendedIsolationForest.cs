using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Extended Isolation Forest", "https://doi.org/10.1109/TKDE.2019.2947676", Year = 2019, Authors = "Sahand Hariri, Matias Carrasco Kind, Robert J. Brunner")]
public class ExtendedIsolationForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private readonly int _extensionLevel;
    private List<ExtendedIsolationTree>? _trees;
    private int _nFeatures;
    private int _trainingSampleSize;

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

        if (extensionLevel < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(extensionLevel),
                "ExtensionLevel must be -1 (full extension) or >= 0. Use 0 for standard Isolation Forest behavior.");
        }

        _numTrees = numTrees;
        _maxSamples = maxSamples;
        _extensionLevel = extensionLevel;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);
        if (X.Rows < 2)
            throw new ArgumentException("At least 2 samples are required for isolation forest.", nameof(X));

        _nFeatures = X.Columns;
        int effectiveExtension = _extensionLevel == -1 ? _nFeatures - 1 : Math.Min(_extensionLevel, _nFeatures - 1);

        int nSamples = Math.Min(_maxSamples, X.Rows);
        _trainingSampleSize = nSamples;
        int maxDepth = (int)Math.Ceiling(Math.Log(nSamples, 2));

        _trees = new List<ExtendedIsolationTree>();

        // Build trees using Matrix<T> directly
        for (int t = 0; t < _numTrees; t++)
        {
            var random = RandomHelper.CreateSeededRandom(_randomSeed + t);

            // Sample data
            var sampleIndices = SampleIndices(X.Rows, nSamples, random);
            var sampleData = new Matrix<T>(sampleIndices.Length, X.Columns);
            for (int si = 0; si < sampleIndices.Length; si++)
            {
                for (int j = 0; j < X.Columns; j++)
                {
                    sampleData[si, j] = X[sampleIndices[si], j];
                }
            }

            // Build tree
            var tree = new ExtendedIsolationTree(effectiveExtension, random, NumOps);
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var trees = _trees;
        if (trees == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int nSamples = _trainingSampleSize;

        // Expected path length for BST (must use training subsample size, not test size)
        T c = NumOps.FromDouble(nSamples > 2
            ? 2 * (Math.Log(nSamples - 1) + 0.5772156649) - (2.0 * (nSamples - 1) / nSamples)
            : nSamples == 2 ? 1 : 0);

        T two = NumOps.FromDouble(2);
        T half = NumOps.FromDouble(0.5);
        T nTreesT = NumOps.FromDouble(trees.Count);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Average path length across all trees
            T avgPathLength = NumOps.Zero;
            foreach (var tree in trees)
            {
                avgPathLength = NumOps.Add(avgPathLength, tree.PathLength(point, 0));
            }
            avgPathLength = NumOps.Divide(avgPathLength, nTreesT);

            // Anomaly score: s = 2^(-avgPathLength/c)
            T score = NumOps.GreaterThan(c, NumOps.Zero)
                ? NumOps.Power(two, NumOps.Negate(NumOps.Divide(avgPathLength, c)))
                : half;

            scores[i] = score;
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
        private readonly INumericOperations<T> _numOps;
        private Vector<T>? _normal;

        private T _intercept;

        private ExtendedIsolationTree? _left;
        private ExtendedIsolationTree? _right;
        private int _size;
        private bool _isLeaf;

        public ExtendedIsolationTree(int extensionLevel, Random random, INumericOperations<T> numOps)
        {
            _extensionLevel = extensionLevel;
            _random = random;
            _numOps = numOps;
            _intercept = numOps.Zero;
            _isLeaf = true;
        }

        public void Build(Matrix<T> data, int currentDepth, int maxDepth)
        {
            _size = data.Rows;

            if (currentDepth >= maxDepth || data.Rows <= 1)
            {
                _isLeaf = true;
                return;
            }

            int nFeatures = data.Columns;

            // Generate random hyperplane normal
            _normal = new Vector<T>(nFeatures);

            if (_extensionLevel == 0)
            {
                // Axis-parallel (standard IF)
                int splitDim = _random.Next(nFeatures);
                _normal[splitDim] = _numOps.One;
            }
            else
            {
                // Extended: random direction
                var dims = Enumerable.Range(0, nFeatures)
                    .OrderBy(_ => _random.NextDouble())
                    .Take(Math.Min(_extensionLevel + 1, nFeatures))
                    .ToArray();

                foreach (int d in dims)
                {
                    // Random Gaussian component (Box-Muller at Random.NextDouble boundary)
                    double u1 = 1.0 - _random.NextDouble();
                    double u2 = 1.0 - _random.NextDouble();
                    double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    _normal[d] = _numOps.FromDouble(z);
                }

                // Normalize using Engine-style manual dot product
                T normSq = _numOps.Zero;
                for (int i = 0; i < nFeatures; i++)
                {
                    normSq = _numOps.Add(normSq, _numOps.Multiply(_normal[i], _normal[i]));
                }
                T norm = _numOps.Sqrt(normSq);
                T eps = _numOps.FromDouble(1e-10);
                if (_numOps.GreaterThan(norm, eps))
                {
                    for (int i = 0; i < nFeatures; i++)
                    {
                        _normal[i] = _numOps.Divide(_normal[i], norm);
                    }
                }
            }

            // Compute projections
            var projections = new Vector<T>(data.Rows);
            for (int i = 0; i < data.Rows; i++)
            {
                T proj = _numOps.Zero;
                for (int j = 0; j < nFeatures; j++)
                {
                    proj = _numOps.Add(proj, _numOps.Multiply(data[i, j], _normal[j]));
                }
                projections[i] = proj;
            }

            // Random intercept between min and max projection
            T minProj = projections[0];
            T maxProj = projections[0];
            for (int i = 1; i < data.Rows; i++)
            {
                if (_numOps.LessThan(projections[i], minProj)) minProj = projections[i];
                if (_numOps.GreaterThan(projections[i], maxProj)) maxProj = projections[i];
            }

            T range = _numOps.Subtract(maxProj, minProj);
            T eps2 = _numOps.FromDouble(1e-10);
            if (NumericOpsLessThanAbs(range, eps2))
            {
                _isLeaf = true;
                return;
            }

            _intercept = _numOps.Add(minProj, _numOps.Multiply(_numOps.FromDouble(_random.NextDouble()), range));

            // Split data
            var leftIndices = new List<int>();
            var rightIndices = new List<int>();

            for (int i = 0; i < data.Rows; i++)
            {
                if (_numOps.LessThan(projections[i], _intercept))
                    leftIndices.Add(i);
                else
                    rightIndices.Add(i);
            }

            if (leftIndices.Count == 0 || rightIndices.Count == 0)
            {
                _isLeaf = true;
                return;
            }

            // Extract submatrices
            var leftData = ExtractRows(data, leftIndices);
            var rightData = ExtractRows(data, rightIndices);

            _isLeaf = false;
            _left = new ExtendedIsolationTree(_extensionLevel, _random, _numOps);
            _right = new ExtendedIsolationTree(_extensionLevel, _random, _numOps);

            _left.Build(leftData, currentDepth + 1, maxDepth);
            _right.Build(rightData, currentDepth + 1, maxDepth);
        }

        public T PathLength(Vector<T> point, int currentDepth)
        {
            if (_isLeaf)
            {
                return _numOps.FromDouble(currentDepth + EstimateC(_size));
            }

            var normal = _normal;
            var left = _left;
            var right = _right;

            if (normal == null || left == null || right == null)
            {
                return _numOps.FromDouble(currentDepth + EstimateC(_size));
            }

            T projection = _numOps.Zero;
            for (int i = 0; i < point.Length; i++)
            {
                projection = _numOps.Add(projection, _numOps.Multiply(point[i], normal[i]));
            }

            if (_numOps.LessThan(projection, _intercept))
                return left.PathLength(point, currentDepth + 1);
            else
                return right.PathLength(point, currentDepth + 1);
        }

        private bool NumericOpsLessThanAbs(T value, T threshold)
        {
            return _numOps.LessThan(_numOps.Abs(value), threshold);
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

        private static double EstimateC(int n)
        {
            if (n > 2)
                return 2 * (Math.Log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n);
            return n == 2 ? 1 : 0;
        }
    }
}
