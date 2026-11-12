using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Boosting-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a boosting-inspired approach where samples are reweighted based on
/// prediction errors, focusing feature selection on harder-to-predict samples.
/// </para>
/// <para><b>For Beginners:</b> Like gradient boosting, this method focuses more
/// attention on the samples that are hardest to predict. Features that help
/// predict these difficult samples get higher importance scores, leading to
/// more discriminative feature selection.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BoostingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nRounds;
    private readonly double _learningRate;

    private Vector<T>? _boostingScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NRounds => _nRounds;
    public Vector<T>? BoostingScores => _boostingScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BoostingFeatureSelector(
        int nFeaturesToSelect = 10,
        int nRounds = 10,
        double learningRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nRounds = nRounds;
        _learningRate = learningRate;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BoostingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var shape = new[] { n, p };
        var yColumn = engine.TensorBroadcastTo(engine.Reshape(y, new[] { n, 1 }), shape);
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        T learningRate = NumOps.FromDouble(_learningRate);
        T offset = NumOps.FromDouble(0.01);
        T zero = NumOps.Zero;
        var ones = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);

        // Initialize sample weights uniformly
        var weights = PreprocessingTensorOps.Filled(new[] { n }, NumOps.FromDouble(1.0 / n));

        // Round r adds 1 / (nRounds - r) to its best feature
        var increments = engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { _nRounds }, NumOps.One),
            engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { _nRounds }, NumOps.FromDouble(_nRounds)),
                PreprocessingTensorOps.Range<T>(_nRounds)));
        var bestFeatures = new List<int>();
        var bestIncrements = new List<Tensor<T>>();
        var predictions = new Tensor<T>(new[] { n });

        for (int round = 0; round < _nRounds && p > 0; round++)
        {
            // Weighted single-feature regression of y on every feature
            var w = engine.TensorBroadcastTo(engine.Reshape(weights, new[] { n, 1 }), shape);
            T wSum = engine.TensorSum(weights);
            var xMean = engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(w, X), new[] { 0 }, true), wSum);
            var yMean = engine.TensorBroadcastTo(engine.Reshape(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorMultiply(weights, y), new[] { 0 }, true), wSum), new[] { 1, 1 }), shape);
            var xd = engine.TensorSubtract(X, engine.TensorBroadcastTo(xMean, shape));
            var yd = engine.TensorSubtract(yColumn, yMean);
            var sxy = engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiply(w, xd), yd), new[] { 0 }, false);
            var sxx = engine.ReduceSum(engine.TensorMultiply(engine.TensorMultiply(w, xd), xd), new[] { 0 }, false);
            var usable = engine.TensorGreaterThan(sxx, floor);
            var beta = engine.TensorWhere(usable,
                engine.TensorDivide(sxy, engine.TensorWhere(usable, sxx, ones)),
                new Tensor<T>(new[] { p }));

            // Predictions and weighted squared error per feature; the smallest error (first on ties) wins
            var pred = engine.TensorAdd(engine.TensorMultiply(engine.TensorBroadcastTo(
                engine.Reshape(beta, new[] { 1, p }), shape), xd), yMean);
            var weightedError = engine.ReduceSum(engine.TensorMultiply(w, engine.TensorSquare(engine.TensorSubtract(yColumn, pred))),
                new[] { 0 }, false);
            int bestFeature = PreprocessingTensorOps.FirstArgMax(engine.TensorNegate(weightedError));

            bestFeatures.Add(bestFeature);
            bestIncrements.Add(engine.TensorSlice(increments, new[] { round }, new[] { 1 }));

            // Update predictions
            var bestPred = PreprocessingTensorOps.ColumnVector(pred, bestFeature);
            predictions = engine.TensorAdd(predictions,
                engine.TensorMultiplyScalar(engine.TensorSubtract(bestPred, predictions), learningRate));

            // Update weights based on residuals
            var residuals = engine.TensorAbs(engine.TensorSubtract(y, predictions));
            T maxResidual = PreprocessingTensorOps.ScalarMax(zero, engine.TensorMaxValue(residuals));

            if (PreprocessingTensorOps.IsAbove(maxResidual, floor))
            {
                weights = engine.TensorAddScalar(engine.TensorDivideScalar(residuals, maxResidual), offset);
                weights = engine.TensorDivideScalar(weights, engine.TensorSum(weights));
            }
        }

        var boostingScores = FeatureSelectionTensorOps.ScatterAdd(bestFeatures, bestIncrements, p);
        _boostingScores = boostingScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(boostingScores, numToSelect);

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
            throw new InvalidOperationException("BoostingFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BoostingFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BoostingFeatureSelector has not been fitted.");

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
