using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Distance;

/// <summary>
/// ReliefF feature selection algorithm.
/// </summary>
/// <remarks>
/// <para>
/// ReliefF is a feature weighting algorithm that estimates feature quality based on
/// how well features distinguish between instances that are near each other.
/// </para>
/// <para>
/// For each randomly sampled instance, ReliefF:
/// 1. Finds k nearest neighbors from the same class (hits)
/// 2. Finds k nearest neighbors from each different class (misses)
/// 3. Updates feature weights based on distance differences
/// </para>
/// <para><b>For Beginners:</b> ReliefF scores each feature by asking:
/// "Does this feature help distinguish different classes?"
///
/// If instances from the same class have similar values but instances from
/// different classes have different values, the feature gets a high score.
///
/// Unlike correlation-based methods, ReliefF can detect feature interactions
/// and works well with non-linear relationships.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    // Fitted parameters
    private Vector<T>? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the number of nearest neighbors used.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the computed feature weights.
    /// </summary>
    public Vector<T>? FeatureWeights => _featureWeights;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="ReliefF{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nNeighbors">Number of nearest neighbors (k). Defaults to 10.</param>
    /// <param name="nIterations">Number of sampling iterations. Defaults to all samples.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public ReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int nIterations = -1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefF requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits ReliefF by computing feature weights.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values (class labels).</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var featureShape = new[] { p };

        // Normalize features to [0, 1] for distance computation (0 for a column whose range is not above 1e-10)
        var minVals = PreprocessingTensorOps.ColumnMin(X);
        var range = engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(X), minVals);
        var varying = engine.TensorGreaterThan(range, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var safeRange = engine.TensorWhere(varying, range, PreprocessingTensorOps.Filled(featureShape, NumOps.One));
        var Xnorm = engine.TensorMultiply(
            engine.TensorDivide(engine.TensorSubtract(X, engine.TensorBroadcastTo(engine.Reshape(minVals, new[] { 1, p }), new[] { n, p })),
                engine.TensorBroadcastTo(engine.Reshape(safeRange, new[] { 1, p }), new[] { n, p })),
            engine.TensorBroadcastTo(engine.Reshape(varying, new[] { 1, p }), new[] { n, p }));

        // Get class information (distinct target values) and the miss coefficients
        // prior_c / (1 - prior_class) for every other class c (0 on the diagonal)
        var classLabels = PreprocessingTensorOps.SortedUnique(y);
        int numClasses = classLabels.Length;
        var classIndexTensor = PreprocessingTensorOps.LookupSorted(classLabels, y, out _);
        var classIndex = classIndexTensor.AsSpan().ToArray();
        var classPriors = engine.TensorDivideScalar(
            PreprocessingTensorOps.CategoryCounts<T>(classIndexTensor, numClasses), NumOps.FromDouble(n));
        var eye = engine.TensorEye<T>(numClasses);
        var missCoefficients = engine.TensorWhere(eye, new Tensor<T>(new[] { numClasses, numClasses }),
            engine.TensorOuter(engine.TensorReciprocal(PreprocessingTensorOps.OneMinus(classPriors)), classPriors));

        // Number of iterations
        int m = _nIterations > 0 ? _nIterations : n;
        int k = Math.Min(_nNeighbors, n - 1);
        T scale = NumOps.FromDouble(1.0 / ((double)m * k));
        var zeroRow = new Tensor<T>(new[] { 1, p });

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // ReliefF algorithm
        var weights = new Tensor<T>(featureShape);
        for (int iter = 0; iter < m; iter++)
        {
            int idx = random.Next(n);
            int instanceClass = classIndex[idx];
            var instance = PreprocessingTensorOps.Row(Xnorm, idx);
            var absDiffs = engine.TensorAbs(engine.TensorSubtract(Xnorm, engine.TensorBroadcastTo(instance, new[] { n, p })));

            // k nearest neighbours of every class (Euclidean, ties in index order), excluding the instance
            var order = PreprocessingTensorOps.StableOrder(
                engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(
                    engine.TensorSubtract(Xnorm, engine.TensorBroadcastTo(instance, new[] { n, p }))), new[] { 1 }, false)),
                descending: false);
            var neighbours = new List<int>[numClasses];
            for (int c = 0; c < numClasses; c++)
                neighbours[c] = new List<int>();
            foreach (int candidate in order)
            {
                var bucket = neighbours[classIndex[candidate]];
                if (candidate != idx && bucket.Count < k)
                    bucket.Add(candidate);
            }

            // Per-class sums of |x_idx - x_neighbour| ([numClasses, p])
            var classSums = new Tensor<T>[numClasses];
            for (int c = 0; c < numClasses; c++)
            {
                classSums[c] = neighbours[c].Count == 0
                    ? zeroRow
                    : engine.ReduceSum(engine.TensorIndexSelect(absDiffs,
                        new Tensor<int>(neighbours[c].ToArray(), new[] { neighbours[c].Count }), 0), new[] { 0 }, true);
            }

            // Hits (same class) decrease the weight; misses (other classes, prior-weighted) increase it
            var hitSum = engine.Reshape(classSums[instanceClass], featureShape);
            var missSum = engine.Reshape(engine.TensorMatMul(PreprocessingTensorOps.Row(missCoefficients, instanceClass),
                engine.TensorConcatenate(classSums, 0)), featureShape);
            weights = engine.TensorAdd(weights, engine.TensorSubtract(
                engine.TensorMultiplyScalar(missSum, scale), engine.TensorMultiplyScalar(hitSum, scale)));
        }

        _featureWeights = weights.ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(weights, nToSelect);

        IsFitted = true;
    }
    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting ReliefF-ranked features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("ReliefF has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ReliefF does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("ReliefF has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
