using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TreeBased;

/// <summary>
/// Detects anomalies using SCiForest (Isolation Forest with Split-selection Criterion).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SCiForest improves on Isolation Forest for anomalies that come in small,
/// tight groups (clustered anomalies). Instead of a random cut on one feature, every node tries several
/// random oblique cuts and keeps the one that best separates the data into two tighter groups.
/// </para>
/// <para>
/// The algorithm (Liu, Ting and Zhou, 2010) works by:
/// 1. Each tree is built from a random sub-sample of psi rows (without replacement)
/// 2. At each node, tau random hyper-planes f(x) = sum_j c_j x_j / sigma_j - p over q random attributes
///    (c_j uniform in [-1, 1], sigma_j the attribute's standard deviation in the node) are tried, each with
///    the split point p that maximizes Sdgain = (sigma(Y) - (sigma(Y_l) + sigma(Y_r)) / 2) / sigma(Y) of the
///    projected values Y; the hyper-plane with the highest Sdgain splits the node (f &lt; 0 left, f &gt;= 0 right)
/// 3. Nodes are split until they hold at most two rows
/// 4. At evaluation, an edge only counts while f(x) stays inside the node's acceptable range
///    [-v, v], v = max f - min f over the node's training rows, which fences off unseen anomalies
/// 5. The anomaly score is s = 2^(-E[h(x)] / c(psi))
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Clustered anomalies close to normal data
/// - When anomalies are only visible in combinations of features
/// - When standard Isolation Forest doesn't perform well
/// </para>
/// <para>
/// <b>Paper Defaults:</b>
/// - Number of trees t: 100
/// - Sub-sample size psi: 256
/// - Attributes per hyper-plane q: 2
/// - Hyper-planes per node tau: 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Liu, F.T., Ting, K.M., Zhou, Z.H. (2010). "On Detecting Clustered Anomalies Using SCiForest."
/// ECML PKDD 2010, LNCS 6322, pp. 274-290.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("On Detecting Clustered Anomalies Using SCiForest", "https://doi.org/10.1007/978-3-642-15883-4_18", Year = 2010, Authors = "Fei Tony Liu, Kai Ming Ting, Zhi-Hua Zhou")]
public partial class SCiForest<T> : AnomalyDetectorBase<T>
{
    private readonly int _numTrees;
    private readonly int _maxSamples;
    private readonly int _attributesPerHyperplane;
    private readonly int _hyperplanesPerNode;
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
    /// Gets the number of attributes q combined in each random hyper-plane.
    /// </summary>
    public int AttributesPerHyperplane => _attributesPerHyperplane;

    /// <summary>
    /// Gets the number of random hyper-planes tau tried at each node.
    /// </summary>
    public int HyperplanesPerNode => _hyperplanesPerNode;

    /// <summary>
    /// Creates a new SCiForest anomaly detector.
    /// </summary>
    /// <param name="numTrees">Number of trees in the forest (t). Default is 100.</param>
    /// <param name="maxSamples">Sub-sample size per tree (psi). Default is 256.</param>
    /// <param name="attributesPerHyperplane">Attributes per hyper-plane (q). Default is 2.</param>
    /// <param name="hyperplanesPerNode">Hyper-planes tried per node (tau). Default is 10.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SCiForest(int numTrees = 100, int maxSamples = 256, int attributesPerHyperplane = 2,
        int hyperplanesPerNode = 10, double contamination = 0.1, int randomSeed = 42)
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

        if (attributesPerHyperplane < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(attributesPerHyperplane),
                "AttributesPerHyperplane must be at least 1. Recommended is 2.");
        }

        if (hyperplanesPerNode < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(hyperplanesPerNode),
                "HyperplanesPerNode must be at least 1. Recommended is 10.");
        }

        NumericGuard.RejectIntegerTypes<T>("SCiForest");

        _numTrees = numTrees;
        _maxSamples = maxSamples;
        _attributesPerHyperplane = attributesPerHyperplane;
        _hyperplanesPerNode = hyperplanesPerNode;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        int nSamples = Math.Min(_maxSamples, X.Rows);
        _trainingSampleSize = nSamples;
        int q = Math.Min(_attributesPerHyperplane, _nFeatures);
        var data = AnomalyTensorOps.ToTensor(X);

        _trees = new List<SCiTree>();
        for (int t = 0; t < _numTrees; t++)
        {
            var random = RandomHelper.CreateSeededRandom(_randomSeed + t);

            // Sub-sample without replacement
            var sampleIndices = SampleIndices(X.Rows, nSamples, random);
            var tree = new SCiTree(_nFeatures, q, _hyperplanesPerNode, random, NumOps);
            tree.Build(AnomalyTensorOps.SelectRows(data, sampleIndices));
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

        // c(psi), the average path length of an unsuccessful BST search over the training sub-sample size
        T c = AveragePathLength(_trainingSampleSize, NumOps);

        var trees = _trees;
        if (trees is null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Anomaly score: s = 2^(-avgPathLength/c), or 0.5 when c is not positive
        if (!NumOps.GreaterThan(c, NumOps.Zero))
        {
            return Engine.Fill(X.Rows, NumOps.FromDouble(0.5));
        }

        // Average path length across all trees (summed in tree order for every point)
        var data = AnomalyTensorOps.ToTensor(X);
        var allRows = AnomalyTensorOps.Range(X.Rows);
        var totalPathLength = Tensor<T>.CreateZeros(new[] { X.Rows });
        for (int t = 0; t < trees.Count; t++)
        {
            totalPathLength = Engine.TensorAdd(totalPathLength, trees[t].PathLengths(data, allRows));
        }

        var avgPathLength = Engine.TensorDivideScalar(totalPathLength, NumOps.FromDouble(trees.Count));
        var exponent = Engine.TensorNegate(Engine.TensorDivideScalar(avgPathLength, c));
        return Engine.TensorExp(Engine.TensorMultiplyScalar(exponent, NumOps.FromDouble(Math.Log(2)))).ToVector();
    }

    // c(m) = 2 H(m - 1) - 2 (m - 1) / m for m > 2, 1 for m = 2, 0 otherwise (Equation 1), H(i) ~ ln(i) + Euler's constant.
    private static T AveragePathLength(int m, INumericOperations<T> numOps)
    {
        if (m > 2)
            return numOps.FromDouble(2 * (Math.Log(m - 1) + 0.5772156649) - (2.0 * (m - 1) / m));
        return numOps.FromDouble(m == 2 ? 1 : 0);
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

    /// <summary>
    /// The split of projected values with the highest Sdgain (Equation 3): over the sorted values, every
    /// boundary between two distinct neighbours is a candidate; Sdgain = (sigma(Y) - (sigma(Y_l) +
    /// sigma(Y_r)) / 2) / sigma(Y) with population standard deviations. Returns the gain and the split point p
    /// (midway between the two neighbours, so y &lt; p goes left), the first best boundary on ties, or a
    /// gain of negative infinity when the values are all equal.
    /// </summary>
    internal static (T Gain, T SplitPoint) BestSdGainSplit(Tensor<T> projections)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int m = projections.Length;
        T noGain = numOps.Negate(numOps.MaxValue);
        if (m < 2)
        {
            return (noGain, numOps.Zero);
        }

        var (sorted, _) = engine.TensorSort(projections, 0, false);
        var centered = engine.TensorSubtractScalar(sorted, engine.TensorMean(sorted));
        T total = numOps.FromDouble(m);
        T sigma = numOps.Sqrt(numOps.Divide(engine.TensorSum(engine.TensorSquare(centered)), total));
        if (!numOps.GreaterThan(sigma, numOps.Zero))
        {
            return (noGain, numOps.Zero);
        }

        // Left part = the first i sorted values (i = 1..m-1), right part the rest; std from the sums of the
        // centered values and their squares
        var prefix = engine.TensorSlice(engine.TensorCumSum(centered, 0), new[] { 0 }, new[] { m - 1 });
        var prefixSquares = engine.TensorSlice(engine.TensorCumSum(engine.TensorSquare(centered), 0), new[] { 0 }, new[] { m - 1 });
        T sum = engine.TensorSum(centered);
        T sumSquares = engine.TensorSum(engine.TensorSquare(centered));
        var leftCount = engine.TensorAddScalar(AnomalyTensorOps.Range(m - 1).Cast<T>(), numOps.One);
        var rightCount = engine.ScalarMinusTensor(total, leftCount);
        var leftStd = PartStd(engine, numOps, prefix, prefixSquares, leftCount);
        var rightStd = PartStd(engine, numOps, engine.ScalarMinusTensor(sum, prefix),
            engine.ScalarMinusTensor(sumSquares, prefixSquares), rightCount);
        var gains = engine.TensorDivideScalar(engine.ScalarMinusTensor(sigma,
            engine.TensorMultiplyScalar(engine.TensorAdd(leftStd, rightStd), numOps.FromDouble(0.5))), sigma);

        // Only boundaries between distinct neighbours separate the values
        var lower = engine.TensorSlice(sorted, new[] { 0 }, new[] { m - 1 });
        var upper = engine.TensorSlice(sorted, new[] { 1 }, new[] { m - 1 });
        var distinct = engine.TensorLessThan(lower, upper);
        gains = engine.TensorWhere(distinct, gains, engine.Fill(m - 1, noGain).AsTensor());

        int best = AnomalyTensorOps.FirstArgMax(gains);
        T gain = engine.TensorSlice(gains, new[] { best }, new[] { 1 }).ToVector()[0];
        T low = engine.TensorSlice(lower, new[] { best }, new[] { 1 }).ToVector()[0];
        T high = engine.TensorSlice(upper, new[] { best }, new[] { 1 }).ToVector()[0];
        return (gain, numOps.Multiply(numOps.Add(low, high), numOps.FromDouble(0.5)));
    }

    // sqrt(max(sumSquares / count - (sum / count)^2, 0))
    private static Tensor<T> PartStd(IEngine engine, INumericOperations<T> numOps, Tensor<T> sum, Tensor<T> sumSquares, Tensor<T> count)
    {
        var mean = engine.TensorDivide(sum, count);
        var variance = engine.TensorSubtract(engine.TensorDivide(sumSquares, count), engine.TensorSquare(mean));
        return engine.TensorSqrt(engine.TensorMax(variance, numOps.Zero));
    }

    private partial class SCiTree
    {
        private readonly INumericOperations<T> _ops;
        private readonly int _nFeatures;
        private readonly int _attributesPerHyperplane;
        private readonly int _hyperplanesPerNode;
        private readonly Random _random;
        [AiDotNet.Attributes.TrainableParameter]
        private Vector<T>? _weights;
        private T _splitPoint;
        private T _upperLimit;
        private T _lowerLimit;
        private SCiTree? _left;
        private SCiTree? _right;
        private int _size;
        private bool _isLeaf;

        public SCiTree(int nFeatures, int attributesPerHyperplane, int hyperplanesPerNode, Random random,
            INumericOperations<T> numOps)
        {
            _ops = numOps;
            _nFeatures = nFeatures;
            _attributesPerHyperplane = attributesPerHyperplane;
            _hyperplanesPerNode = hyperplanesPerNode;
            _random = random;
            _isLeaf = true;
            _splitPoint = _ops.Zero;
            _upperLimit = _ops.Zero;
            _lowerLimit = _ops.Zero;
        }

        // Algorithm 1: split while the node holds more than two rows.
        public void Build(Tensor<T> data)
        {
            var engine = AiDotNetEngine.Current;
            _size = data.Shape[0];
            if (_size <= 2)
            {
                _isLeaf = true;
                return;
            }

            // Attribute standard deviations in this node; a constant attribute contributes nothing
            var std = engine.TensorStdMean(data, new[] { 0 }, 0, false).Item1;
            var positiveStd = engine.TensorGreaterThan(std, _ops.Zero);
            var inverseStd = engine.TensorWhere(positiveStd,
                engine.TensorReciprocal(engine.TensorWhere(positiveStd, std, Tensor<T>.CreateOnes(new[] { _nFeatures }))),
                Tensor<T>.CreateZeros(new[] { _nFeatures }));

            // tau random hyper-planes over q attributes without replacement, coefficients uniform in [-1, 1];
            // keep the one whose best split has the highest Sdgain (the first on ties)
            var trialWeights = new List<Vector<T>>();
            var trialProjections = new List<Tensor<T>>();
            var trialGains = new List<Tensor<T>>();
            var trialSplits = new List<Tensor<T>>();
            for (int trial = 0; trial < _hyperplanesPerNode; trial++)
            {
                var pool = Enumerable.Range(0, _nFeatures).ToArray();
                var coefficients = new double[_nFeatures];
                for (int i = 0; i < _attributesPerHyperplane; i++)
                {
                    int j = i + _random.Next(_nFeatures - i);
                    (pool[i], pool[j]) = (pool[j], pool[i]);
                    coefficients[pool[i]] = _random.NextDouble() * 2.0 - 1.0;
                }

                var weights = engine.TensorMultiply(
                    Tensor<double>.FromVector(Vector<double>.Wrap(coefficients), new[] { _nFeatures }).Cast<T>(), inverseStd).ToVector();
                var projections = AnomalyTensorOps.Project(data, weights);
                var (gain, split) = BestSdGainSplit(projections);
                trialWeights.Add(weights);
                trialProjections.Add(projections);
                trialGains.Add(Tensor<T>.CreateDefault(new[] { 1 }, gain));
                trialSplits.Add(Tensor<T>.CreateDefault(new[] { 1 }, split));
            }

            var gains = engine.Concat(trialGains, 0);
            int best = AnomalyTensorOps.FirstArgMax(gains);
            T bestGain = engine.TensorSlice(gains, new[] { best }, new[] { 1 }).ToVector()[0];
            if (!_ops.GreaterThan(bestGain, _ops.Negate(_ops.MaxValue)))
            {
                // Every hyper-plane projects the node's rows onto a single value: nothing separates them
                _isLeaf = true;
                return;
            }

            var bestWeights = trialWeights[best];
            var bestProjections = trialProjections[best];
            T bestSplit = engine.TensorSlice(engine.Concat(trialSplits, 0), new[] { best }, new[] { 1 }).ToVector()[0];

            // f(x) = w . x - p: f < 0 goes left; acceptable range [-v, v] with v = max f - min f
            var f = engine.TensorSubtractScalar(bestProjections, bestSplit);
            T v = _ops.Subtract(engine.TensorMaxValue(f), engine.TensorMinValue(f));
            var goesLeft = engine.TensorLessThan(f, _ops.Zero);
            var leftIndices = AnomalyTensorOps.NonzeroIndices(goesLeft);
            var rightIndices = AnomalyTensorOps.NonzeroIndices(engine.ScalarMinusTensor(_ops.One, goesLeft));

            _isLeaf = false;
            _weights = bestWeights;
            _splitPoint = bestSplit;
            _upperLimit = v;
            _lowerLimit = _ops.Negate(v);
            _left = new SCiTree(_nFeatures, _attributesPerHyperplane, _hyperplanesPerNode, _random, _ops);
            _right = new SCiTree(_nFeatures, _attributesPerHyperplane, _hyperplanesPerNode, _random, _ops);
            _left.Build(AnomalyTensorOps.SelectRows(data, leftIndices));
            _right.Build(AnomalyTensorOps.SelectRows(data, rightIndices));
        }

        /// <summary>
        /// Path lengths (Algorithm 2) of the given rows of <paramref name="data"/>, in
        /// the order of <paramref name="rows"/>.
        /// </summary>
        public Tensor<T> PathLengths(Tensor<T> data, Tensor<int> rows)
        {
            var leafRows = new List<Tensor<int>>();
            var leafValues = new List<Tensor<T>>();
            Route(data, rows, AnomalyTensorOps.Range(rows.Length), Tensor<T>.CreateZeros(new[] { rows.Length }),
                leafRows, leafValues);
            return AnomalyTensorOps.AssembleByRow(leafRows, leafValues);
        }

        // Routes rows (identified by their positions in the caller's row list) down the tree with their edge
        // counts e; f(x) >= 0 goes right, counting the edge only when f(x) < UpperLimit, f(x) < 0 goes left,
        // counting it only when LowerLimit <= f(x). Each leaf contributes e + c(size).
        private void Route(Tensor<T> data, Tensor<int> rows, Tensor<int> positions, Tensor<T> edges,
            List<Tensor<int>> leafRows, List<Tensor<T>> leafValues)
        {
            if (rows.Length == 0)
            {
                return;
            }

            var engine = AiDotNetEngine.Current;
            if (_isLeaf)
            {
                leafRows.Add(positions);
                leafValues.Add(engine.TensorAddScalar(edges, AveragePathLength(_size, _ops)));
                return;
            }

            var weights = _weights;
            var left = _left;
            var right = _right;
            if (weights is null || left is null || right is null)
            {
                throw new InvalidOperationException(
                    "Non-leaf SCiTree node has uninitialized children or weights. " +
                    "This indicates a corrupted tree structure.");
            }

            var f = engine.TensorSubtractScalar(AnomalyTensorOps.Project(engine.TensorIndexSelect(data, rows, 0), weights), _splitPoint);
            var goesLeft = engine.TensorLessThan(f, _ops.Zero);
            var countRight = engine.TensorLessThan(f, _upperLimit);
            var countLeft = engine.ScalarMinusTensor(_ops.One, engine.TensorLessThan(f, _lowerLimit));
            var newEdges = engine.TensorAdd(edges, engine.TensorWhere(goesLeft, countLeft, countRight));
            var leftMask = engine.TensorEqScalar(goesLeft, _ops.One);
            var rightMask = engine.TensorEqScalar(goesLeft, _ops.Zero);
            left.Route(data, AnomalyTensorOps.SelectWhere(rows, goesLeft, _ops.One),
                AnomalyTensorOps.SelectWhere(positions, goesLeft, _ops.One),
                engine.TensorMaskedSelect(newEdges, leftMask), leafRows, leafValues);
            right.Route(data, AnomalyTensorOps.SelectWhere(rows, goesLeft, _ops.Zero),
                AnomalyTensorOps.SelectWhere(positions, goesLeft, _ops.Zero),
                engine.TensorMaskedSelect(newEdges, rightMask), leafRows, leafValues);
        }
    }
}
