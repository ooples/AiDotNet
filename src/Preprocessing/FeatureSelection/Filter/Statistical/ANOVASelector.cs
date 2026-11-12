using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// ANOVA-based feature selection for multi-class problems.
/// </summary>
/// <remarks>
/// <para>
/// Analysis of Variance (ANOVA) extends the t-test to multiple classes. It measures
/// how much the mean of each feature varies across classes compared to within-class
/// variation.
/// </para>
/// <para><b>For Beginners:</b> ANOVA checks if the average value of a feature is
/// significantly different across multiple groups. If a feature has very different
/// averages for each class, it's good at distinguishing between them. Unlike the
/// t-test, ANOVA works with any number of classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _fStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FStatistics => _fStatistics;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ANOVASelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Group samples by class
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int nClasses);

        // Between-group (SSB) and within-group (SSW) sums of squares of every feature
        var (ssb, ssw) = FilterStatisticTensorOps.AnovaSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ClassOneHot<T>(classIndex, nClasses));

        // F = (SSB / dfb) / (SSW / dfw) = (SSB dfw / dfb) / SSW; a feature with no within-class scatter but
        // distinct class means separates the classes perfectly (+inf), a constant one scores 0
        int dfb = nClasses - 1;
        int dfw = n - nClasses;
        var fStatistics = dfw > 0
            ? FilterStatisticTensorOps.SeparationAwareRatio(
                engine.TensorMultiplyScalar(ssb, NumOps.Divide(NumOps.FromDouble(dfw), NumOps.FromDouble(dfb))), ssw, 1e-10)
            : FilterStatisticTensorOps.FilledLike(ssb, NumOps.Zero);
        _fStatistics = fStatistics.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(fStatistics, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ANOVASelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

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
