using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("Isolation-Based Anomaly Detection", "https://doi.org/10.1145/2133360.2133363", Year = 2012, Authors = "Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou")]
public class SCiForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private readonly double _sparsity;
    private List<SCiTree>? _trees;
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

        NumericGuard.RejectIntegerTypes<T>("SCiForest");

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
        _trainingSampleSize = nSamples;
        int maxDepth = (int)Math.Ceiling(Math.Log(nSamples, 2));

        _trees = new List<SCiTree>();

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
            var tree = new SCiTree(_nFeatures, _sparsity, random, NumOps);
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

        var scores = new Vector<T>(X.Rows);

        // Expected path length for BST — must use the training subsample size, not the scoring batch size
        T c = _trainingSampleSize > 2
            ? NumOps.FromDouble(2 * (Math.Log(_trainingSampleSize - 1) + 0.5772156649) - (2.0 * (_trainingSampleSize - 1) / _trainingSampleSize))
            : NumOps.FromDouble(_trainingSampleSize == 2 ? 1 : 0);

        var trees = _trees;
        if (trees is null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new Vector<T>(X.GetRowReadOnlySpan(i).ToArray());

            // Average path length across all trees
            T totalPathLength = NumOps.Zero;
            for (int t = 0; t < trees.Count; t++)
            {
                totalPathLength = NumOps.Add(totalPathLength, trees[t].PathLength(point, 0));
            }
            T avgPathLength = NumOps.Divide(totalPathLength, NumOps.FromDouble(trees.Count));

            // Anomaly score: s = 2^(-avgPathLength/c)
            if (NumOps.GreaterThan(c, NumOps.Zero))
            {
                T exponent = NumOps.Negate(NumOps.Divide(avgPathLength, c));
                T ln2 = NumOps.FromDouble(Math.Log(2));
                scores[i] = NumOps.Exp(NumOps.Multiply(exponent, ln2));
            }
            else
            {
                scores[i] = NumOps.FromDouble(0.5);
            }
        }

        return scores;
    }

    private static int[] SampleIndices(int total, int sampleSize, Random random)
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
        private readonly INumericOperations<T> _ops;

        /// <summary>Minimum range threshold below which a feature dimension is considered constant.</summary>
        private readonly T _minRangeTolerance;

        private readonly int _nFeatures;
        private readonly double _sparsity;
        private readonly Random _random;
        private Vector<T>? _sparseWeights;
        private T _threshold;
        private SCiTree? _left;
        private SCiTree? _right;
        private int _size;
        private bool _isLeaf;

        public SCiTree(int nFeatures, double sparsity, Random random, INumericOperations<T> numOps)
        {
            _ops = numOps;
            _minRangeTolerance = _ops.FromDouble(1e-10);
            _nFeatures = nFeatures;
            _sparsity = sparsity;
            _random = random;
            _isLeaf = true;
            _threshold = _ops.Zero;
        }

        public void Build(Matrix<T> data, int currentDepth, int maxDepth)
        {
            _size = data.Rows;

            if (currentDepth >= maxDepth || data.Rows <= 1)
            {
                _isLeaf = true;
                return;
            }

            // Generate sparse random projection
            _sparseWeights = new Vector<T>(_nFeatures);

            int numActive = Math.Max(1, (int)(_nFeatures * _sparsity));

            var activeFeatures = Enumerable.Range(0, _nFeatures)
                .OrderBy(_ => _random.NextDouble())
                .Take(numActive)
                .ToArray();

            foreach (int f in activeFeatures)
            {
                _sparseWeights[f] = _random.NextDouble() < 0.5 ? _ops.One : _ops.Negate(_ops.One);
            }

            // Compute projections
            var projections = new Vector<T>(data.Rows);
            for (int i = 0; i < data.Rows; i++)
            {
                var row = new Vector<T>(data.GetRowReadOnlySpan(i).ToArray());
                T proj = _ops.Zero;
                for (int j = 0; j < _nFeatures; j++)
                {
                    proj = _ops.Add(proj, _ops.Multiply(row[j], _sparseWeights[j]));
                }
                projections[i] = proj;
            }

            T minProj = projections[0];
            T maxProj = projections[0];
            for (int i = 1; i < data.Rows; i++)
            {
                if (_ops.LessThan(projections[i], minProj)) minProj = projections[i];
                if (_ops.GreaterThan(projections[i], maxProj)) maxProj = projections[i];
            }

            T range = _ops.Subtract(maxProj, minProj);
            if (_ops.LessThan(range, _minRangeTolerance))
            {
                _isLeaf = true;
                return;
            }

            _threshold = _ops.Add(minProj, _ops.Multiply(_ops.FromDouble(_random.NextDouble()), range));

            // Split data
            var leftIndices = new List<int>();
            var rightIndices = new List<int>();

            for (int i = 0; i < data.Rows; i++)
            {
                if (_ops.LessThan(projections[i], _threshold))
                    leftIndices.Add(i);
                else
                    rightIndices.Add(i);
            }

            if (leftIndices.Count == 0 || rightIndices.Count == 0)
            {
                _isLeaf = true;
                return;
            }

            _isLeaf = false;
            _left = new SCiTree(_nFeatures, _sparsity, _random, _ops);
            _right = new SCiTree(_nFeatures, _sparsity, _random, _ops);

            _left.Build(ExtractRows(data, leftIndices), currentDepth + 1, maxDepth);
            _right.Build(ExtractRows(data, rightIndices), currentDepth + 1, maxDepth);
        }

        public T PathLength(Vector<T> point, int currentDepth)
        {
            if (_isLeaf)
            {
                return _ops.Add(_ops.FromDouble(currentDepth), EstimateC(_size));
            }

            var sparseWeights = _sparseWeights;
            var left = _left;
            var right = _right;

            if (sparseWeights is null || left is null || right is null)
            {
                throw new InvalidOperationException(
                    "Non-leaf SCiTree node has uninitialized children or weights. " +
                    "This indicates a corrupted tree structure.");
            }

            T projection = _ops.Zero;
            for (int i = 0; i < point.Length; i++)
            {
                projection = _ops.Add(projection, _ops.Multiply(point[i], sparseWeights[i]));
            }

            if (_ops.LessThan(projection, _threshold))
                return left.PathLength(point, currentDepth + 1);
            else
                return right.PathLength(point, currentDepth + 1);
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

        private T EstimateC(int n)
        {
            if (n > 2)
                return _ops.FromDouble(2 * (Math.Log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n));
            return _ops.FromDouble(n == 2 ? 1 : 0);
        }
    }
}
