using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robustness;

/// <summary>
/// Outlier Robust Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maintain their predictive relationship with the target
/// even when outliers are present, using robust statistics.
/// </para>
/// <para><b>For Beginners:</b> Some features are heavily influenced by extreme values
/// (outliers). This selector finds features whose relationship with the target is
/// consistent whether outliers are present or removed.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OutlierRobustSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _outlierThreshold;

    private Vector<T>? _robustnessScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double OutlierThreshold => _outlierThreshold;
    public Vector<T>? RobustnessScores => _robustnessScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public OutlierRobustSelector(
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
            "OutlierRobustSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var x = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Outliers: |x - median| / (1.4826 * MAD) above the threshold (only when that MAD is above 1e-10)
        var deviation = engine.TensorAbs(engine.TensorSubtract(x, SelectorTensorOps.ColumnMedian(x, keepDims: true)));
        var mad = engine.TensorMultiplyScalar(SelectorTensorOps.ColumnMedian(deviation, keepDims: true), NumOps.FromDouble(1.4826));
        var spread = engine.TensorBroadcastTo(engine.TensorGreaterThan(mad, NumOps.FromDouble(1e-10)), new[] { n, p });
        var isOutlier = engine.TensorMultiply(spread,
            engine.TensorGreaterThan(engine.TensorDivide(deviation, mad), NumOps.FromDouble(_outlierThreshold)));

        // Correlation with all data and without each column's outliers (needs more than 2 rows)
        var all = PreprocessingTensorOps.Filled(new[] { n, p }, NumOps.One);
        var corrAll = SelectorTensorOps.MaskedPearson(x, y, all, 2);
        var corrFiltered = SelectorTensorOps.MaskedPearson(x, y, PreprocessingTensorOps.Not(isOutlier), 3);

        // Robustness score: high correlation that is stable with and without outliers
        var stability = PreprocessingTensorOps.OneMinus(engine.TensorAbs(engine.TensorSubtract(corrAll, corrFiltered)));
        var scores = engine.TensorMultiply(engine.TensorAbs(corrAll), stability);

        _robustnessScores = scores.ToVector();
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
            throw new InvalidOperationException("OutlierRobustSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("OutlierRobustSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("OutlierRobustSelector has not been fitted.");

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
