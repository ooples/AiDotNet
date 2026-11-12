using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// F-Regression for feature selection based on linear relationship with continuous target.
/// </summary>
/// <remarks>
/// <para>
/// Computes the F-statistic for a univariate linear regression between each feature and
/// the continuous target. Features with higher F-statistics have stronger linear
/// relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> For regression problems (predicting continuous values),
/// we need to know which features have a linear relationship with what we're predicting.
/// F-regression fits a simple line between each feature and the target, then measures
/// how well that line explains the variation in the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private Vector<T>? _fStatistics;
    private Vector<T>? _pValues;
    private Vector<T>? _rSquaredValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FStatistics => _fStatistics;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? RSquaredValues => _rSquaredValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FRegression(
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
            "FRegression requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        var engine = Engine;

        // Centred sums of every feature with the target (ssTotal = total sum of squares of the target)
        var (sxx, sxy, ssTotal) = FilterStatisticTensorOps.CenteredCrossSums(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));

        // Slope beta1 = Sxy / Sxx (0 for a constant feature)
        var beta1 = FilterStatisticTensorOps.GuardedDivide(sxy, sxx, 1e-10, NumOps.Zero);

        // Residual sum of squares sum (y - ybar - beta1 (x - xbar))^2 = Syy - 2 beta1 Sxy + beta1^2 Sxx
        var ssTotalTensor = FilterStatisticTensorOps.FilledLike(sxx, ssTotal);
        var ssResidual = engine.TensorMax(
            engine.TensorAdd(
                engine.TensorSubtract(ssTotalTensor, engine.TensorMultiplyScalar(engine.TensorMultiply(beta1, sxy), NumOps.FromDouble(2))),
                engine.TensorMultiply(engine.TensorSquare(beta1), sxx)),
            NumOps.Zero);

        // Regression sum of squares
        var ssRegression = engine.TensorSubtract(ssTotalTensor, ssResidual);

        // R-squared
        _rSquaredValues = NumOps.GreaterThan(ssTotal, NumOps.FromDouble(1e-10))
            ? engine.TensorMax(engine.TensorDivideScalar(ssRegression, ssTotal), NumOps.Zero).ToVector()
            : FilterStatisticTensorOps.FilledLike(sxx, NumOps.Zero).ToVector();

        // F-statistic: F = (SSR/1) / (SSE/(n-2)); a perfect linear fit leaves no residual and gets F = +inf
        int dfResidual = Math.Max(1, n - 2);
        var msResidual = engine.TensorDivideScalar(ssResidual, NumOps.FromDouble(dfResidual));
        var fStatistics = FilterStatisticTensorOps.SeparationAwareRatio(ssRegression, msResidual, 1e-10);
        _fStatistics = fStatistics.ToVector();

        // P-value from the F-distribution (1, n-2 degrees of freedom), scipy.stats.f.sf
        var pValues = FilterStatisticTensorOps.FSurvival(fStatistics, 1, dfResidual);
        _pValues = pValues.ToVector();

        // Select significant features or top by F-statistic
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
            throw new InvalidOperationException("FRegression has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FRegression does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FRegression has not been fitted.");

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
