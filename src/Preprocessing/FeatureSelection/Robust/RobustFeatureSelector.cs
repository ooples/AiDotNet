using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Robust Feature Selection resistant to outliers and noise.
/// </summary>
/// <remarks>
/// <para>
/// Uses robust statistics (median, MAD, rank-based measures) to identify
/// important features while being resistant to outliers and noise in the data.
/// </para>
/// <para><b>For Beginners:</b> Regular feature selection can be fooled by extreme
/// values (outliers) in the data. This method uses statistics that are "robust"
/// - they don't change much when a few data points are extreme. This gives
/// more reliable feature selection for messy real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RobustFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _outlierThreshold;

    private Vector<T>? _robustScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? RobustScores => _robustScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RobustFeatureSelector(
        int nFeaturesToSelect = 10,
        double outlierThreshold = 3.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _outlierThreshold = outlierThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RobustFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var yColumn = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Non-outlier rows by the target: |y - median| / (MAD + 1e-10) <= threshold
        var deviation = engine.TensorAbs(engine.TensorSubtract(yColumn, FeatureSelectionTensorOps.ColumnMedian(yColumn, keepDims: true)));
        var mad = engine.TensorAddScalar(FeatureSelectionTensorOps.ColumnMedian(deviation, keepDims: true), NumOps.FromDouble(1e-10));
        var scaled = engine.Reshape(engine.TensorDivide(deviation, mad), new[] { n });
        var nonOutlierIndices = PreprocessingTensorOps.MaskPositions(
            engine.TensorLessEqual(scaled, PreprocessingTensorOps.Filled(new[] { n }, NumOps.FromDouble(_outlierThreshold))));

        // Spearman correlation (average ranks) on the non-outlier rows
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (nonOutlierIndices.Length >= 3)
        {
            var rows = new Tensor<int>(nonOutlierIndices, new[] { nonOutlierIndices.Length });
            var xRanks = FeatureSelectionTensorOps.AverageRanks(engine.TensorIndexSelect(Tensor<T>.FromMatrix(data), rows, 0));
            var yRanks = FeatureSelectionTensorOps.AverageRanks(engine.TensorIndexSelect(yColumn, rows, 0));
            scores = FeatureSelectionTensorOps.PearsonWithColumn(xRanks, yRanks);
        }

        _robustScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(scores), Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("RobustFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RobustFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RobustFeatureSelector has not been fitted.");

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
