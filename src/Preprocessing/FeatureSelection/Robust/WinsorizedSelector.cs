using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Winsorized Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Winsorization to replace extreme values with less extreme values before
/// computing feature importance, providing robust feature selection.
/// </para>
/// <para><b>For Beginners:</b> Winsorizing means replacing the most extreme values
/// in your data with less extreme ones (typically at the 5th and 95th percentile).
/// This way, outliers don't disappear but are "capped" at reasonable values,
/// giving more robust feature importance measurements.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WinsorizedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    private Vector<T>? _winsorizedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowerPercentile => _lowerPercentile;
    public double UpperPercentile => _upperPercentile;
    public Vector<T>? WinsorizedScores => _winsorizedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WinsorizedSelector(
        int nFeaturesToSelect = 10,
        double lowerPercentile = 0.05,
        double upperPercentile = 0.95,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lowerPercentile < 0 || lowerPercentile >= upperPercentile || upperPercentile > 1)
            throw new ArgumentException("Percentiles must satisfy 0 <= lower < upper <= 1.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WinsorizedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Winsorize target and features, then correlate
        var yWinsorized = FeatureSelectionTensorOps.WinsorizeColumns(
            engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _lowerPercentile, _upperPercentile);
        var xWinsorized = FeatureSelectionTensorOps.WinsorizeColumns(Tensor<T>.FromMatrix(data), _lowerPercentile, _upperPercentile);
        var scores = engine.TensorAbs(FeatureSelectionTensorOps.PearsonWithColumn(xWinsorized, yWinsorized));

        _winsorizedScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("WinsorizedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WinsorizedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WinsorizedSelector has not been fitted.");

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
