using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Permutation Importance feature selection by measuring score decrease when shuffling features.
/// </summary>
/// <remarks>
/// <para>
/// Permutation Importance measures the decrease in model performance when a single feature's
/// values are randomly shuffled, breaking the relationship between the feature and the target.
/// Features whose shuffling causes large performance drops are considered important.
/// </para>
/// <para><b>For Beginners:</b> Imagine you have a trained model and want to know which features
/// it relies on. You shuffle one feature's values randomly (like shuffling a deck of cards)
/// and see how much worse the model performs. If performance drops a lot, that feature was
/// important. If shuffling makes no difference, the feature wasn't useful.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PermutationImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nRepeats;
    private readonly Func<Matrix<T>, Vector<T>, T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _importances;
    private Vector<T>? _importanceStds;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private T _baselineScore = MathHelper.GetNumericOperations<T>().Zero;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NRepeats => _nRepeats;
    public Vector<T>? Importances => _importances;
    public Vector<T>? ImportanceStds => _importanceStds;
    public T BaselineScore => _baselineScore;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PermutationImportance(
        int nFeaturesToSelect = 10,
        int nRepeats = 5,
        Func<Matrix<T>, Vector<T>, T>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nRepeats < 1)
            throw new ArgumentException("Number of repeats must be at least 1.", nameof(nRepeats));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nRepeats = nRepeats;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PermutationImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scorer = _scorer ?? DefaultScorer;

        // Compute baseline score
        _baselineScore = scorer(data, target);

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var shape = new[] { n, p };
        var identity = engine.TensorEye<T>(p);

        // For each feature, permute it (one permutation per repeat, in order) and measure score decrease
        var permutedScores = new T[p * _nRepeats];
        for (int j = 0; j < p; j++)
        {
            var column = PreprocessingTensorOps.ColumnVector(X, j);
            var columnMask = engine.TensorBroadcastTo(PreprocessingTensorOps.Row(identity, j), shape);

            for (int r = 0; r < _nRepeats; r++)
            {
                // Create permuted data
                var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
                var shuffled = engine.TensorIndexSelect(column, new Tensor<int>(indices, new[] { n }), 0);
                var permutedData = engine.TensorWhere(columnMask,
                    engine.TensorBroadcastTo(engine.Reshape(shuffled, new[] { n, 1 }), shape), X).ToMatrix();
                permutedScores[j * _nRepeats + r] = scorer(permutedData, target);
            }
        }

        // Mean and sample standard deviation of the score decreases per feature
        var decreases = engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p, _nRepeats }, _baselineScore),
            new Tensor<T>(permutedScores, new[] { p, _nRepeats }));
        var importances = engine.ReduceMean(decreases, new[] { 1 }, false);
        _importances = importances.ToVector();
        _importanceStds = _nRepeats > 1
            ? engine.TensorSqrt(engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(
                decreases, engine.TensorBroadcastTo(engine.Reshape(importances, new[] { p, 1 }), new[] { p, _nRepeats }))),
                new[] { 1 }, false), NumOps.FromDouble(_nRepeats - 1))).ToVector()
            : new Vector<T>(p);

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }

    private T DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        // Simple R² score of the averaged correlation predictor yMean + sum_j r_j * z_j * yStd / p
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        var shape = new[] { n, p };
        var featureShape = new[] { p };
        T floor = NumOps.FromDouble(1e-10);
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Means, standard deviations and Pearson correlations (0 for a constant feature or target)
        var xMeans = engine.ReduceMean(X, new[] { 0 }, true);
        var xDiff = engine.TensorSubtract(X, engine.TensorBroadcastTo(xMeans, shape));
        var yMean = engine.ReduceMean(y, new[] { 0 }, true);
        var yDiff = engine.TensorSubtract(y, engine.TensorBroadcastTo(yMean, new[] { n }));
        var sxy = engine.ReduceSum(engine.TensorMultiply(xDiff,
            engine.TensorBroadcastTo(engine.Reshape(yDiff, new[] { n, 1 }), shape)), new[] { 0 }, false);
        var sxx = engine.ReduceSum(engine.TensorSquare(xDiff), new[] { 0 }, false);
        var syy = engine.ReduceSum(engine.TensorSquare(yDiff), new[] { 0 }, true);
        var xStds = engine.TensorSqrt(engine.TensorDivideScalar(sxx, NumOps.FromDouble(n)));
        var yStd = engine.TensorBroadcastTo(engine.TensorSqrt(engine.TensorDivideScalar(syy, NumOps.FromDouble(n))), featureShape);
        var ones = PreprocessingTensorOps.Filled(featureShape, NumOps.One);
        var zeros = new Tensor<T>(featureShape);
        var usable = engine.TensorMultiply(engine.TensorGreaterThan(xStds, floor), engine.TensorGreaterThan(yStd, floor));
        var denominators = engine.TensorWhere(usable,
            engine.TensorSqrt(engine.TensorMultiply(sxx, engine.TensorBroadcastTo(syy, featureShape))), ones);
        var weights = engine.TensorWhere(usable, engine.TensorDivideScalar(
            engine.TensorMultiply(engine.TensorDivide(sxy, denominators), yStd), NumOps.FromDouble(p)), zeros);

        // Predict and compute R²
        var safeStds = engine.TensorWhere(usable, xStds, ones);
        var zScores = engine.TensorDivide(xDiff, engine.TensorBroadcastTo(engine.Reshape(safeStds, new[] { 1, p }), shape));
        var predictions = engine.TensorAdd(engine.TensorBroadcastTo(yMean, new[] { n }), engine.ReduceSum(
            engine.TensorMultiply(zScores, engine.TensorBroadcastTo(engine.Reshape(weights, new[] { 1, p }), shape)),
            new[] { 1 }, false));
        T ssRes = engine.TensorSumOfSquares(engine.TensorSubtract(y, predictions));
        T ssTot = engine.TensorSumOfSquares(yDiff);

        return PreprocessingTensorOps.IsAbove(ssTot, floor) ? NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot)) : NumOps.Zero;
    }
    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PermutationImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PermutationImportance has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
