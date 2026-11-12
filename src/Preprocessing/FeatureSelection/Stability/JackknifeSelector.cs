using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Jackknife based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their stability under jackknife resampling,
/// which systematically leaves out one sample at a time.
/// </para>
/// <para><b>For Beginners:</b> Jackknife is a resampling method that removes one
/// data point at a time and recalculates statistics. Features with consistent
/// importance scores across all jackknife samples are more reliable.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class JackknifeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxJackknifeIterations;

    private Vector<T>? _meanImportance;
    private Vector<T>? _importanceVariance;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MeanImportance => _meanImportance;
    public Vector<T>? ImportanceVariance => _importanceVariance;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public JackknifeSelector(
        int nFeaturesToSelect = 10,
        int maxJackknifeIterations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxJackknifeIterations = maxJackknifeIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "JackknifeSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        int nIterations = Math.Min(n, _maxJackknifeIterations);
        var allImportance = new Tensor<T>[nIterations];

        // Jackknife: leave one out at a time
        for (int leave = 0; leave < nIterations; leave++)
        {
            // Compute feature importance (|correlation| with target) on the leave-one-out rows
            int left = leave;
            var kept = Enumerable.Range(0, n).Where(i => i != left).ToArray();
            var rows = new Tensor<int>(kept, new[] { n - 1 });
            allImportance[leave] = n - 1 < 2
                ? PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero)
                : Engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(
                    Engine.TensorIndexSelect(x, rows, 0), Engine.TensorIndexSelect(y, rows, 0)));
        }

        // Mean and jackknife variance (Efron and Stein, 1981: (m - 1) / m sum_i (theta_i - mean)^2) of the importance
        var stacked = Engine.TensorStack(allImportance, 0);
        var mean = Engine.ReduceMean(stacked, new[] { 0 }, false);
        var variance = Engine.TensorMultiplyScalar(
            SelectorTensorOps.ColumnSumOfSquares(FilterStatisticTensorOps.CenterColumns(stacked)),
            NumOps.FromDouble((nIterations - 1) / (double)nIterations));
        _meanImportance = mean.ToVector();
        _importanceVariance = variance.ToVector();

        // Select features with high mean importance and low variance (stable): mean / (std + 1e-6)
        var stabilityScores = Engine.TensorDivide(mean,
            Engine.TensorAddScalar(Engine.TensorSqrt(variance), NumOps.FromDouble(1e-6)));

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(stabilityScores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("JackknifeSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("JackknifeSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JackknifeSelector has not been fitted.");

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
