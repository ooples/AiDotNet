using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TreeBased;

/// <summary>
/// Implements the Isolation Forest algorithm for anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Isolation Forest is an efficient algorithm for detecting anomalies.
/// The key insight is that anomalies are "few and different" - they are easier to isolate
/// from the rest of the data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Building a "forest" of random isolation trees
/// 2. Each tree randomly partitions the data by selecting random features and split points
/// 3. Anomalies tend to be isolated in fewer steps (closer to the root of the tree)
/// 4. Normal points require more splits to isolate (deeper in the tree)
/// </para>
/// <para>
/// <b>When to use:</b> Isolation Forest is particularly effective for:
/// - High-dimensional data
/// - Large datasets (it scales linearly)
/// - When you don't know the distribution of your data
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Number of trees: 100 (provides stable results)
/// - Max samples: 256 (balances speed and accuracy)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Liu, F. T., Ting, K. M., and Zhou, Z. H. (2008). "Isolation Forest."
/// In: ICDM '08. Eighth IEEE International Conference on Data Mining.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Isolation Forest", "https://doi.org/10.1109/ICDM.2008.17", Year = 2008, Authors = "Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou")]
public partial class IsolationForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private List<IsolationTree>? _trees;

    private T _averagePathLength;

    private int _inputDim;

    /// <summary>
    /// Gets the number of trees in the forest.
    /// </summary>
    public int NumTrees => _numTrees;

    /// <summary>
    /// Gets the number of samples used to train each tree.
    /// </summary>
    public int MaxSamples => _maxSamples;

    /// <summary>
    /// Creates a new Isolation Forest anomaly detector.
    /// </summary>
    /// <param name="numTrees">
    /// The number of isolation trees in the forest. Default is 100 (industry standard).
    /// More trees provide more stable results but increase computation time.
    /// The original paper suggests 100 trees are typically sufficient.
    /// </param>
    /// <param name="maxSamples">
    /// The number of samples to draw from X to train each tree. Default is 256 (industry standard).
    /// Smaller values make the algorithm faster and can improve detection of local anomalies.
    /// The original paper recommends 256 as a good default.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// Used to set the decision threshold after fitting.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The default values work well for most cases:
    /// - 100 trees provide stable results
    /// - 256 samples per tree balance speed and accuracy
    /// - 10% contamination is a reasonable starting assumption
    ///
    /// If you're unsure, start with defaults and adjust based on results.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when numTrees or maxSamples are less than 1.
    /// </exception>
    public IsolationForest(
        int numTrees = 100,
        int maxSamples = 256,
        double contamination = 0.1,
        int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        _averagePathLength = NumOps.Zero;
        if (numTrees < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numTrees),
                "Number of trees must be at least 1. Recommended value is 100.");
        }

        if (maxSamples < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSamples),
                "Max samples must be at least 1. Recommended value is 256.");
        }

        _numTrees = numTrees;
        _maxSamples = maxSamples;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _inputDim = X.Columns;
        int n = X.Rows;
        int effectiveMaxSamples = Math.Min(_maxSamples, n);

        if (effectiveMaxSamples < 2)
        {
            throw new ArgumentException(
                $"Isolation Forest requires at least 2 samples. Got {n} samples with maxSamples={_maxSamples}.",
                nameof(X));
        }

        int maxDepth = (int)Math.Ceiling(Math.Log(effectiveMaxSamples) / Math.Log(2));

        // Calculate the average path length for normalization
        _averagePathLength = NumOps.FromDouble(AveragePathLength(effectiveMaxSamples));

        // Build the forest
        _trees = new List<IsolationTree>(_numTrees);
        var columns = AnomalyTensorOps.Columns(X);

        for (int t = 0; t < _numTrees; t++)
        {
            // Sample indices for this tree
            var sampleIndices = SampleIndices(n, effectiveMaxSamples);
            var tree = BuildTree(columns, sampleIndices, 0, maxDepth);
            _trees.Add(tree);
        }

        // Calculate scores for training data to set the threshold
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

        if (X.Columns != _inputDim)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_inputDim} features.",
                nameof(X));
        }

        if (!NumOps.GreaterThan(_averagePathLength, NumOps.Zero))
        {
            throw new InvalidOperationException(
                "Average path length is invalid. Ensure the model was fitted with at least 2 samples.");
        }

        var trees = _trees ?? throw new InvalidOperationException("_trees is null where it is required.");
        var engine = Engine;
        T nTreesT = NumOps.FromDouble(_numTrees);

        // Path lengths of every point, summed over the trees in order
        var columns = AnomalyTensorOps.Columns(X);
        var allRows = AnomalyTensorOps.Range(X.Rows);
        var totalPathLength = Tensor<T>.CreateZeros(new[] { X.Rows });
        foreach (var tree in trees)
        {
            totalPathLength = engine.TensorAdd(totalPathLength, TreePathLengths(columns, allRows, tree));
        }

        var avgPathLength = engine.TensorDivideScalar(totalPathLength, nTreesT);

        // Anomaly score: s(x, n) = 2^(-E(h(x))/c(n))
        var exponent = engine.TensorNegate(engine.TensorDivideScalar(avgPathLength, _averagePathLength));
        return engine.TensorExp2(exponent).ToVector();
    }

    private int[] SampleIndices(int n, int sampleSize)
    {
        var indices = new int[sampleSize];
        var available = Enumerable.Range(0, n).ToList();

        for (int i = 0; i < sampleSize; i++)
        {
            int idx = _random.Next(available.Count);
            indices[i] = available[idx];
            available.RemoveAt(idx);
        }

        return indices;
    }

    private IsolationTree BuildTree(Tensor<T>[] columns, int[] indices, int currentDepth, int maxDepth)
    {
        int n = indices.Length;

        // Stopping conditions: external node
        if (currentDepth >= maxDepth || n <= 1)
        {
            return new IsolationTree
            {
                IsExternal = true,
                Size = n
            };
        }

        // Select a random feature
        int splitFeature = _random.Next(columns.Length);

        // Find min and max for the selected feature among the samples
        var engine = Engine;
        var rows = Vector<int>.Wrap(indices).AsTensor();
        var values = engine.TensorGather(columns[splitFeature], rows, 0);
        T minVal = engine.TensorMinValue(values);
        T maxVal = engine.TensorMaxValue(values);

        // If all values are the same, create external node
        if (NumOps.Equals(minVal, maxVal))
        {
            return new IsolationTree
            {
                IsExternal = true,
                Size = n
            };
        }

        // Select a random split value between min and max
        T fraction = NumOps.FromDouble(_random.NextDouble());
        T splitThreshold = NumOps.Add(minVal, NumOps.Multiply(fraction, NumOps.Subtract(maxVal, minVal)));

        // Partition the indices (order preserved): values below the threshold go left
        var goesLeft = engine.TensorLessThan(values, splitThreshold);
        var leftIndices = AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.One);
        var rightIndices = AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.Zero);

        // Recursively build children
        return new IsolationTree
        {
            IsExternal = false,
            SplitFeature = splitFeature,
            SplitValue = splitThreshold,
            Left = BuildTree(columns, leftIndices.ToArray(), currentDepth + 1, maxDepth),
            Right = BuildTree(columns, rightIndices.ToArray(), currentDepth + 1, maxDepth)
        };
    }

    // Path length h(x) of every row in one tree: rows are routed down the tree as index sets, and each
    // external node contributes its depth plus the c(size) adjustment for the rows that reach it.
    private Tensor<T> TreePathLengths(Tensor<T>[] columns, Tensor<int> rows, IsolationTree tree)
    {
        var leafRows = new List<Tensor<int>>();
        var leafValues = new List<Tensor<T>>();
        RoutePathLengths(columns, rows, tree, 0, leafRows, leafValues);
        return AnomalyTensorOps.AssembleByRow(leafRows, leafValues);
    }

    private void RoutePathLengths(Tensor<T>[] columns, Tensor<int> rows, IsolationTree node, int currentDepth,
        List<Tensor<int>> leafRows, List<Tensor<T>> leafValues)
    {
        if (rows.Length == 0)
        {
            return;
        }

        if (node.IsExternal)
        {
            leafRows.Add(rows);
            leafValues.Add(Engine.Fill(rows.Length, NumOps.FromDouble(currentDepth + AveragePathLength(node.Size))).AsTensor());
            return;
        }

        var engine = Engine;
        var goesLeft = engine.TensorLessThan(engine.TensorGather(columns[node.SplitFeature], rows, 0), node.SplitValue);
        RoutePathLengths(columns, AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.One),
            node.Left ?? throw new InvalidOperationException("node.Left is null where it is required."),
            currentDepth + 1, leafRows, leafValues);
        RoutePathLengths(columns, AnomalyTensorOps.SelectWhere(rows, goesLeft, NumOps.Zero),
            node.Right ?? throw new InvalidOperationException("node.Right is null where it is required."),
            currentDepth + 1, leafRows, leafValues);
    }

    /// <summary>
    /// Calculates the average path length of unsuccessful search in a Binary Search Tree.
    /// This is used for normalizing the path lengths.
    /// </summary>
    /// <param name="n">The number of data points.</param>
    /// <returns>The average path length c(n).</returns>
    private static double AveragePathLength(int n)
    {
        // Handle base cases with ternary
        if (n <= 2) return n <= 1 ? 0 : 1;

        // c(n) = 2H(n-1) - 2(n-1)/n
        // where H(i) is the harmonic number ≈ ln(i) + Euler's constant
        double harmonicNumber = Math.Log(n - 1) + 0.5772156649; // Euler-Mascheroni constant
        return 2.0 * harmonicNumber - 2.0 * (n - 1) / n;
    }

    /// <summary>
    /// Internal class representing a node in an Isolation Tree.
    /// </summary>
    private class IsolationTree
    {
        public bool IsExternal { get; set; }
        public int Size { get; set; }
        public int SplitFeature { get; set; }
        public T SplitValue { get; set; } = MathHelper.GetNumericOperations<T>().Zero;
        public IsolationTree? Left { get; set; }
        public IsolationTree? Right { get; set; }
    }
}
