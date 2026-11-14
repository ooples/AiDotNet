using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.CrossValidation;

/// <summary>
/// Cross-Validated Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their predictive performance evaluated using
/// k-fold cross-validation with a simple linear model.
/// </para>
/// <para><b>For Beginners:</b> Cross-validation tests how well features predict
/// targets on data not used for training. Features that consistently help
/// predictions across multiple data splits are selected.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CrossValidatedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nFolds;
    private readonly int? _randomState;

    private Vector<T>? _cvScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NFolds => _nFolds;
    public Vector<T>? CVScores => _cvScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CrossValidatedSelector(
        int nFeaturesToSelect = 10,
        int nFolds = 5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nFolds = nFolds;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CrossValidatedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        // Create random fold assignments
        var indices = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rand.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var folds = new int[n];
        for (int i = 0; i < n; i++)
            folds[indices[i]] = i % _nFolds;

        // Evaluate each feature individually: a simple linear regression per feature and fold, all features at once
        var shape = new[] { n, p };
        var yWide = engine.TensorBroadcastTo(y, shape);
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        T zero = NumOps.Zero;
        var zeros = new Tensor<T>(new[] { p });
        var totalR2 = new Tensor<T>(new[] { p });

        for (int fold = 0; fold < _nFolds; fold++)
        {
            // Split data
            int foldIndex = fold;
            var testFlags = Array.ConvertAll(folds, f => f == foldIndex ? 1 : 0);
            int testCount = testFlags.Sum();
            int trainCount = n - testCount;
            if (trainCount < 2 || testCount == 0) continue;

            var testRows = engine.TensorBroadcastTo(
                new Tensor<int>(testFlags, new[] { n, 1 }).Cast<T>(), shape);
            var trainRows = PreprocessingTensorOps.OneMinus(testRows);
            T trainSize = PreprocessingTensorOps.CountAsScalar<T>(trainCount);
            T testSize = PreprocessingTensorOps.CountAsScalar<T>(testCount);

            // Simple linear regression on the training rows
            var xMean = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(X, trainRows), new[] { 0 }, true), trainSize);
            var yMean = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(yWide, trainRows), new[] { 0 }, true), trainSize);
            var dx = engine.TensorMultiply(engine.TensorSubtract(X, engine.TensorBroadcastTo(xMean, shape)), trainRows);
            var dy = engine.TensorSubtract(yWide, engine.TensorBroadcastTo(yMean, shape));
            var numerator = engine.ReduceSum(engine.TensorMultiply(dx, dy), new[] { 0 }, true);
            var denominator = engine.ReduceSum(engine.TensorMultiply(dx, dx), new[] { 0 }, true);
            var slope = engine.TensorWhere(engine.TensorGreaterThan(denominator, floor),
                engine.TensorDivide(numerator, denominator), new Tensor<T>(new[] { 1, p }));
            var intercept = engine.TensorSubtract(yMean, engine.TensorMultiply(slope, xMean));

            // Predict on test set
            var prediction = engine.TensorAdd(
                engine.TensorMultiply(X, engine.TensorBroadcastTo(slope, shape)), engine.TensorBroadcastTo(intercept, shape));
            var residual = engine.TensorMultiply(engine.TensorSubtract(yWide, prediction), testRows);
            var ssRes = engine.ReduceSum(engine.TensorSquare(residual), new[] { 0 }, false);
            var testYMean = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(yWide, testRows), new[] { 0 }, true), testSize);
            var testDeviation = engine.TensorMultiply(engine.TensorSubtract(yWide, engine.TensorBroadcastTo(testYMean, shape)), testRows);
            var ssTot = engine.ReduceSum(engine.TensorSquare(testDeviation), new[] { 0 }, false);

            var r2 = engine.TensorWhere(engine.TensorGreaterThan(ssTot, floor),
                PreprocessingTensorOps.OneMinus(engine.TensorDivide(ssRes, ssTot)), zeros);
            totalR2 = engine.TensorAdd(totalR2, engine.TensorMax(r2, zero)); // Avoid negative R2
        }

        var cvScores = engine.TensorDivideScalar(totalR2, NumOps.FromDouble(_nFolds));
        _cvScores = cvScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(cvScores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("CrossValidatedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CrossValidatedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CrossValidatedSelector has not been fitted.");

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
