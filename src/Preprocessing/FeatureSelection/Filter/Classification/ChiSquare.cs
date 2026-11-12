using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// Chi-Square (χ²) test for feature selection in classification.
/// </summary>
/// <remarks>
/// <para>
/// Chi-Square feature selection measures the dependence between features and
/// class labels using contingency tables. Features with high chi-square scores
/// have a significant statistical relationship with the target.
/// </para>
/// <para><b>For Beginners:</b> The Chi-Square test checks if the distribution of
/// a feature is different for different classes. If the values of a feature look
/// very different between classes, it gets a high score because that difference
/// can help predict class membership.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ChiSquare<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _chiSquareScores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? ChiSquareScores => _chiSquareScores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ChiSquare(
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
            "ChiSquare requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class labels (rounded target values) and range-scaled feature bins (0 for a constant feature)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int nClasses = classes.Length;
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, nClasses);
        var featureBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);

        // Chi-square statistic over the contingency table and its chi2((bins - 1)(classes - 1)) p-value
        var chiSquare = FeatureSelectionTensorOps.ContingencyChiSquared(featureBins, classOneHot, _nBins, out _);
        var df = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble((_nBins - 1) * (nClasses - 1)));
        _chiSquareScores = chiSquare.ToVector();
        _pValues = FeatureSelectionTensorOps.ChiSquaredSurvival(chiSquare, df).ToVector();

        // Select top features by chi-square score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(chiSquare, numToSelect);

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
            throw new InvalidOperationException("ChiSquare has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ChiSquare does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ChiSquare has not been fitted.");

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
