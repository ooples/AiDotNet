using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Drop Column Importance feature selection by measuring score decrease when dropping features.
/// </summary>
/// <remarks>
/// <para>
/// Drop Column Importance measures the decrease in model performance when each feature is
/// completely removed from the dataset and the model is retrained. Unlike permutation
/// importance, this captures the feature's contribution to model learning, not just prediction.
/// </para>
/// <para><b>For Beginners:</b> Instead of shuffling a feature (like permutation importance),
/// you completely remove it and retrain the model from scratch. If the model performs much
/// worse without a feature, that feature is important for learning. This is more thorough
/// but also more computationally expensive since you retrain for each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DropColumnImportance<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, T>? _scorer;
    private readonly int _nFolds;

    private Vector<T>? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private T _baselineScore = MathHelper.GetNumericOperations<T>().Zero;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Importances => _importances;
    public T BaselineScore => _baselineScore;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DropColumnImportance(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, T>? scorer = null,
        int nFolds = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scorer = scorer;
        _nFolds = nFolds;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DropColumnImportance requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var scorer = _scorer ?? DefaultCVScorer;

        // Compute baseline score with all features
        _baselineScore = scorer(data, target);

        // For each feature, drop it and measure score decrease
        var droppedScores = new T[p];
        var allColumns = Enumerable.Range(0, p).ToArray();
        for (int j = 0; j < p; j++)
        {
            int dropped = j;
            var droppedData = PreprocessingTensorOps.SelectColumns(data, allColumns.Where(c => c != dropped).ToArray());
            droppedScores[j] = scorer(droppedData, target);
        }

        var importances = Engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p }, _baselineScore),
            new Tensor<T>(droppedScores, new[] { p }));
        _importances = importances.ToVector();

        // Select top features by importance
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);

        IsFitted = true;
    }

    private T DefaultCVScorer(Matrix<T> data, Vector<T> target)
    {
        // Cross-validated R² score over contiguous folds (the last fold takes the remainder)
        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        int foldSize = n / _nFolds;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var totalScore = new Tensor<T>(new[] { 1 });

        for (int fold = 0; fold < _nFolds; fold++)
        {
            int testStart = fold * foldSize;
            int testEnd = fold == _nFolds - 1 ? n : (fold + 1) * foldSize;
            int testCount = testEnd - testStart;
            int trainCount = n - testCount;

            if (trainCount > 0 && testCount > 0 && p > 0)
            {
                var trainRows = Enumerable.Range(0, n).Where(i => i < testStart || i >= testEnd).ToArray();
                var trainIndex = new Tensor<int>(trainRows, new[] { trainCount });
                var score = ComputeR2Score(
                    engine.TensorIndexSelect(X, trainIndex, 0), engine.TensorIndexSelect(y, trainIndex, 0),
                    engine.TensorSlice(X, new[] { testStart, 0 }, new[] { testCount, p }),
                    engine.TensorSlice(y, new[] { testStart }, new[] { testCount }));
                totalScore = engine.TensorAdd(totalScore, score);
            }
        }

        return NumOps.Divide(engine.TensorSum(totalScore), NumOps.FromDouble(_nFolds));
    }

    /// <summary>
    /// Test-set R² of the per-feature correlation predictor <c>yMean + sum_j r_j * z_j * yStd</c> fitted on the
    /// training rows (population statistics; features with std at most 1e-10 contribute nothing). Returns [1].
    /// </summary>
    private Tensor<T> ComputeR2Score(Tensor<T> trainX, Tensor<T> trainY, Tensor<T> testX, Tensor<T> testY)
    {
        var engine = Engine;
        int nTrain = trainX.Shape[0];
        int nTest = testX.Shape[0];
        int nFeatures = trainX.Shape[1];
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var featureShape = new[] { nFeatures };
        var trainShape = new[] { nTrain, nFeatures };
        var testShape = new[] { nTest, nFeatures };

        // Means, correlations and standard deviations
        var xMeans = engine.ReduceMean(trainX, new[] { 0 }, true);
        var xDiff = engine.TensorSubtract(trainX, engine.TensorBroadcastTo(xMeans, trainShape));
        var yMean = engine.ReduceMean(trainY, new[] { 0 }, true);
        var yDiff = engine.TensorSubtract(trainY, engine.TensorBroadcastTo(yMean, new[] { nTrain }));
        var yVar = engine.ReduceMean(engine.TensorSquare(yDiff), new[] { 0 }, true);
        var yStd = engine.TensorSqrt(yVar);

        var sxy = engine.ReduceSum(engine.TensorMultiply(xDiff,
            engine.TensorBroadcastTo(engine.Reshape(yDiff, new[] { nTrain, 1 }), trainShape)), new[] { 0 }, false);
        var sxx = engine.ReduceSum(engine.TensorSquare(xDiff), new[] { 0 }, false);
        var xStds = engine.TensorSqrt(engine.TensorDivideScalar(sxx, NumOps.FromDouble(nTrain)));
        var yVarFeatures = engine.TensorBroadcastTo(yVar, featureShape);
        var zeros = new Tensor<T>(featureShape);
        var correlations = engine.TensorWhere(
            engine.TensorMultiply(engine.TensorGreaterThan(sxx, floor), engine.TensorGreaterThan(yVarFeatures, floor)),
            engine.TensorDivide(sxy, engine.TensorSqrt(engine.TensorMultiplyScalar(
                engine.TensorMultiply(sxx, yVarFeatures), NumOps.FromDouble(nTrain)))),
            zeros);

        // Predict on test set
        var usable = engine.TensorGreaterThan(xStds, floor);
        var safeStds = engine.TensorWhere(usable, xStds, PreprocessingTensorOps.Filled(featureShape, NumOps.One));
        var zScores = engine.TensorDivide(
            engine.TensorSubtract(testX, engine.TensorBroadcastTo(xMeans, testShape)),
            engine.TensorBroadcastTo(engine.Reshape(safeStds, new[] { 1, nFeatures }), testShape));
        var weights = engine.TensorWhere(usable,
            engine.TensorMultiply(correlations, engine.TensorBroadcastTo(yStd, featureShape)), zeros);
        var predictions = engine.TensorAdd(engine.TensorBroadcastTo(yMean, new[] { nTest }), engine.ReduceSum(
            engine.TensorMultiply(zScores, engine.TensorBroadcastTo(engine.Reshape(weights, new[] { 1, nFeatures }), testShape)),
            new[] { 1 }, false));

        var ssRes = engine.ReduceSum(engine.TensorSquare(engine.TensorSubtract(testY, predictions)), new[] { 0 }, true);
        var testDiff = engine.TensorSubtract(testY, engine.TensorBroadcastTo(engine.ReduceMean(testY, new[] { 0 }, true), new[] { nTest }));
        var ssTot = engine.ReduceSum(engine.TensorSquare(testDiff), new[] { 0 }, true);
        var one = PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.One);
        return engine.TensorWhere(engine.TensorGreaterThan(ssTot, floor),
            engine.TensorSubtract(one, engine.TensorDivide(ssRes, ssTot)), new Tensor<T>(new[] { 1 }));
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
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DropColumnImportance does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DropColumnImportance has not been fitted.");

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
