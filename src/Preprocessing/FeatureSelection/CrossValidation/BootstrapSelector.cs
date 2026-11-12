using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.CrossValidation;

/// <summary>
/// Bootstrap based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance stability across multiple bootstrap
/// samples, ensuring robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap sampling creates many random subsets of
/// data (with replacement). Features that are consistently important across all
/// these subsets are more reliable than those that only work on specific subsets.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BootstrapSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _sampleRatio;
    private readonly int? _randomState;

    private Vector<T>? _stabilityScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstraps => _nBootstraps;
    public double SampleRatio => _sampleRatio;
    public Vector<T>? StabilityScores => _stabilityScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BootstrapSelector(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double sampleRatio = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _sampleRatio = sampleRatio;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = Tensor<T>.FromVector(target);

        int sampleSize = (int)(n * _sampleRatio);
        var selectionCounts = new int[p];
        T count = NumOps.FromDouble(sampleSize);
        T varianceFloor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var zeros = new Tensor<T>(new[] { p });

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Create bootstrap sample
            var sampleIndices = new int[sampleSize];
            for (int i = 0; i < sampleSize; i++)
                sampleIndices[i] = rand.Next(n);
            var rows = new Tensor<int>(sampleIndices, new[] { sampleSize });
            var xs = engine.TensorIndexSelect(X, rows, 0);
            var ys = engine.TensorIndexSelect(y, rows, 0);

            // Compute |correlation| with target for each feature on bootstrap sample (raw-moment form)
            var yMean = engine.TensorBroadcastTo(engine.ReduceMean(ys, new[] { 0 }, true), new[] { p });
            var yVar = engine.TensorSubtract(
                engine.TensorBroadcastTo(engine.ReduceMean(engine.TensorSquare(ys), new[] { 0 }, true), new[] { p }),
                engine.TensorSquare(yMean));
            var xMean = engine.TensorDivideScalar(engine.ReduceSum(xs, new[] { 0 }, false), count);
            var xVar = engine.TensorSubtract(
                engine.TensorDivideScalar(engine.ReduceSum(engine.TensorSquare(xs), new[] { 0 }, false), count),
                engine.TensorSquare(xMean));
            var xyMean = engine.TensorDivideScalar(engine.Reshape(
                engine.TensorMatMul(engine.TensorTranspose(xs), engine.Reshape(ys, new[] { sampleSize, 1 })), new[] { p }), count);
            var covar = engine.TensorSubtract(xyMean, engine.TensorMultiply(xMean, yMean));
            var usable = engine.TensorMultiply(
                engine.TensorGreaterThan(xVar, varianceFloor), engine.TensorGreaterThan(yVar, varianceFloor));
            var correlations = engine.TensorWhere(usable,
                engine.TensorAbs(engine.TensorDivide(covar, engine.TensorSqrt(engine.TensorMultiply(xVar, yVar)))), zeros);

            // Select top features for this bootstrap
            foreach (int j in PreprocessingTensorOps.TopKIndicesSorted(correlations, _nFeaturesToSelect))
                selectionCounts[j]++;
        }

        // Stability score = proportion of bootstraps where feature was selected
        var stability = engine.TensorDivideScalar(
            new Tensor<int>(selectionCounts, new[] { p }).Cast<T>(), NumOps.FromDouble(_nBootstraps));
        _stabilityScores = stability.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(stability, numToSelect);

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
            throw new InvalidOperationException("BootstrapSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BootstrapSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapSelector has not been fitted.");

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
