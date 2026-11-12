using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Chi-Squared Feature Selection for categorical features.
/// </summary>
/// <remarks>
/// <para>
/// Uses the chi-squared statistic to measure the dependency between each
/// feature and the target variable. Features with higher chi-squared values
/// have stronger associations with the target.
/// </para>
/// <para><b>For Beginners:</b> The chi-squared test checks if two things
/// are related. If knowing the value of a feature helps predict the target,
/// they're related (high chi-squared). If the feature's value tells you
/// nothing about the target, chi-squared is low. This method keeps features
/// that are most related to what you're trying to predict.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ChiSquaredSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _chiSquaredScores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ChiSquaredScores => _chiSquaredScores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ChiSquaredSelection(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ChiSquaredSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize target (rounded class labels) and features (range-scaled bins, 0 for a constant feature)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int nClasses = classes.Length;
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, nClasses);
        var featureBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);

        // Chi-squared over the contingency table; df = (non-empty bins - 1) * (classes - 1)
        var chiSquared = FeatureSelectionTensorOps.ContingencyChiSquared(featureBins, classOneHot, _nBins, out var occupiedBins);
        var df = engine.TensorMultiplyScalar(engine.TensorAddScalar(occupiedBins, NumOps.FromDouble(-1)),
            NumOps.FromDouble(nClasses - 1));
        _chiSquaredScores = chiSquared.ToVector();
        _pValues = FeatureSelectionTensorOps.ChiSquaredSurvival(chiSquared, df).ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(chiSquared, numToSelect);

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
            throw new InvalidOperationException("ChiSquaredSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ChiSquaredSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquaredSelection has not been fitted.");

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
