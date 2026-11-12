using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// F-Test for feature selection in regression problems.
/// </summary>
/// <remarks>
/// <para>
/// The F-test computes the ratio of the variance explained by the relationship between
/// a feature and the target to the unexplained variance. Higher F-statistics indicate
/// features with stronger linear relationships to the target.
/// </para>
/// <para><b>For Beginners:</b> The F-test asks: "How much of the target's variation
/// can be explained by this feature compared to random noise?" A high F-value means
/// the feature explains a significant portion of the target variability. It's used
/// for continuous targets (regression), not classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _fScores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FScores => _fScores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FTest(
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
            "FTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Centred sums of every feature with the target
        var (sxx, sxy, ssTotal) = FilterStatisticTensorOps.CenteredCrossSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));

        // Simple linear regression y = a + b x: slope b = Sxy / Sxx (0 for a constant feature)
        var slope = FilterStatisticTensorOps.GuardedDivide(sxy, sxx, 1e-10, NumOps.Zero);

        // SS regression = b^2 Sxx, SS residual = Syy - 2 b Sxy + b^2 Sxx
        var ssReg = engine.TensorMultiply(engine.TensorSquare(slope), sxx);
        var ssRes = engine.TensorMax(
            engine.TensorAdd(
                engine.TensorSubtract(
                    FilterStatisticTensorOps.FilledLike(sxx, ssTotal),
                    engine.TensorMultiplyScalar(engine.TensorMultiply(slope, sxy), NumOps.FromDouble(2))),
                ssReg),
            NumOps.Zero);

        // F-statistic = (SS_reg / df_reg) / (SS_res / df_res), df_reg = 1, df_res = n - 2; a perfect linear
        // fit leaves no residual and gets F = +inf (constant features score 0)
        int dfReg = 1;
        int dfRes = n - 2;
        var msRes = dfRes > 0
            ? engine.TensorDivideScalar(ssRes, NumOps.FromDouble(dfRes))
            : FilterStatisticTensorOps.FilledLike(ssRes, NumOps.One);
        var fScores = FilterStatisticTensorOps.SeparationAwareRatio(ssReg, msRes, 1e-10);
        _fScores = fScores.ToVector();

        // p = P(F(1, n - 2) > F) from the F distribution (scipy.stats.f.sf)
        _pValues = dfRes > 0
            ? FilterStatisticTensorOps.FSurvival(fScores, dfReg, dfRes).ToVector()
            : FilterStatisticTensorOps.FilledLike(fScores, NumOps.One).ToVector();

        // Select top features by F-score
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(fScores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("FTest has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FTest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FTest has not been fitted.");

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
