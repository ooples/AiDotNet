using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Pearson Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Pearson Correlation measures the linear relationship between each feature
/// and the target variable. Features with higher absolute correlation are
/// considered more predictive.
/// </para>
/// <para><b>For Beginners:</b> Pearson correlation measures how well a straight line
/// fits the relationship between a feature and the target. A value of 1 means perfect
/// positive linear relationship, -1 means perfect negative, and 0 means no linear
/// relationship. Good for finding features with linear dependencies.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PearsonCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _useAbsoluteValue;

    private Vector<T>? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool UseAbsoluteValue => _useAbsoluteValue;
    public Vector<T>? CorrelationScores => _correlationScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PearsonCorrelation(
        int nFeaturesToSelect = 10,
        bool useAbsoluteValue = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _useAbsoluteValue = useAbsoluteValue;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PearsonCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Pearson correlation of every feature with the target (0 when either sum of squares is not above 1e-10)
        var correlations = PreprocessingTensorOps.TargetCorrelations(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target));
        var absolute = Engine.TensorAbs(correlations);
        _correlationScores = (_useAbsoluteValue ? absolute : correlations).ToVector();

        // Select top features by absolute correlation
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(absolute, numToSelect);

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
            throw new InvalidOperationException("PearsonCorrelation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PearsonCorrelation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PearsonCorrelation has not been fitted.");

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
