using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// ANOVA (Analysis of Variance) F-test for feature selection in classification problems.
/// </summary>
/// <remarks>
/// <para>
/// ANOVA F-test compares the variance between groups (classes) to the variance within groups.
/// A high F-statistic indicates that the feature discriminates well between classes.
/// This is equivalent to f_classif in scikit-learn.
/// </para>
/// <para><b>For Beginners:</b> ANOVA asks: "Is the difference between class means large
/// compared to the scatter within each class?" If class means are very different but
/// values within each class are similar, the feature is good at separating classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ANOVA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _fStatistics;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FStatistics => _fStatistics;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ANOVA(
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
            "ANOVA requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var classIndex = FeatureSelectionTensorOps.ClassIndex(target, out int k);
        int dfBetween = k - 1;
        int dfWithin = n - k;

        // SS_between = sum n_c (class_mean - overall_mean)^2, SS_within = sum (x - class_mean)^2, per feature
        var (ssBetween, ssWithin) = FeatureSelectionTensorOps.AnovaSums(
            FeatureSelectionTensorOps.AsTensor(data), FeatureSelectionTensorOps.ClassOneHot<T>(classIndex, k));
        var msBetween = engine.TensorDivideScalar(ssBetween, NumOps.FromDouble(Math.Max(1, dfBetween)));
        var msWithin = engine.TensorDivideScalar(ssWithin, NumOps.FromDouble(Math.Max(1, dfWithin)));

        // F = MS_between / MS_within; with no within-class scatter a feature whose class means differ
        // separates the classes perfectly (F = +inf, as scikit-learn's f_classif reports), a constant one scores 0
        var fStatistics = FeatureSelectionTensorOps.SeparationAwareRatio(msBetween, msWithin, 1e-10);
        _fStatistics = fStatistics.ToVector();

        // p = P(F(k - 1, n - k) > F), the F-distribution survival function (scipy.stats.f.sf)
        _pValues = dfBetween > 0 && dfWithin > 0
            ? FeatureSelectionTensorOps.FSurvival(fStatistics, dfBetween, dfWithin).ToVector()
            : FeatureSelectionTensorOps.FilledLike(fStatistics, NumOps.One).ToVector();

        // Select top features by F-statistic
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(fStatistics, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("ANOVA has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ANOVA does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVA has not been fitted.");

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
