using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// ANOVA F-test for feature selection in multi-class classification.
/// </summary>
/// <remarks>
/// <para>
/// Uses one-way Analysis of Variance to test if feature means differ significantly
/// across multiple classes. Features with high F-statistics indicate strong
/// discriminative power between classes.
/// </para>
/// <para><b>For Beginners:</b> ANOVA asks: "Does this feature have different average
/// values across the different classes?" If a feature has very similar averages
/// in all classes, it can't help tell them apart. Features with high F-scores
/// have means that vary significantly between classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ANOVAFTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

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

    public ANOVAFTest(
        int nFeaturesToSelect = 10,
        double pValueThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _pValueThreshold = pValueThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANOVAFTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        var engine = Engine;

        // Group samples by class
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int k);
        if (k < 2)
            throw new ArgumentException("At least 2 classes are required for ANOVA.");

        // Between- and within-group sums of squares of every feature
        var (ssBetween, ssWithin) = FilterStatisticTensorOps.AnovaSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ClassOneHot<T>(classIndex, k));

        // Degrees of freedom
        int dfBetween = k - 1;
        int dfWithin = n - k;

        // Mean squares and F-statistic
        T epsilon = NumOps.FromDouble(1e-10);
        var msBetween = engine.TensorDivideScalar(ssBetween, NumOps.FromDouble(dfBetween));
        var msWithin = engine.TensorDivideScalar(ssWithin, NumOps.FromDouble(dfWithin + 1e-10));
        var fStatistics = engine.TensorDivide(msBetween, engine.TensorAddScalar(msWithin, epsilon));
        _fStatistics = fStatistics.ToVector();

        // p = P(F(k - 1, n - k) > F) from the F distribution (scipy.stats.f.sf)
        var pValues = dfWithin > 0
            ? FilterStatisticTensorOps.FSurvival(fStatistics, dfBetween, dfWithin)
            : FilterStatisticTensorOps.FilledLike(fStatistics, NumOps.One);
        _pValues = pValues.ToVector();

        // Select significant features, or the top features by F-statistic
        _selectedIndices = FilterStatisticTensorOps.SignificantOrTopK(
            pValues, NumOps.FromDouble(_pValueThreshold), fStatistics, _nFeaturesToSelect);

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
            throw new InvalidOperationException("ANOVAFTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ANOVAFTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVAFTest has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
