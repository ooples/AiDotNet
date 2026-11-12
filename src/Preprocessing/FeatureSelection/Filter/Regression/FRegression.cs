using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Regression;

/// <summary>
/// F-statistic based feature selection for regression.
/// </summary>
/// <remarks>
/// <para>
/// F-Regression computes the F-statistic between each feature and the target,
/// which measures the linear dependency between them. Features with higher
/// F-scores have stronger linear relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> The F-test checks how well each feature can predict
/// the target using a simple line (linear relationship). Features that follow
/// a straighter line with the target get higher scores. Good for finding features
/// that have clear linear patterns with your outcome.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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

    public FRegression(
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
            "FRegression requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Centred sums of every feature with the target (sst = total sum of squares of the target)
        var (sxx, sxy, sst) = FeatureSelectionTensorOps.CenteredCrossSums(
            FeatureSelectionTensorOps.AsTensor(data), FeatureSelectionTensorOps.AsTensor(target));
        T epsilon = NumOps.FromDouble(1e-10);
        var zeros = FeatureSelectionTensorOps.FilledLike(sxx, NumOps.Zero);

        // A constant feature or target carries no signal: F = 0, p = 1
        var informative = NumOps.LessThan(sst, epsilon)
            ? zeros
            : FeatureSelectionTensorOps.OneMinus(engine.TensorLessThan(sxx, epsilon));

        // Regression sum of squares SSR = beta^2 Sxx = Sxy^2 / Sxx, error sum of squares SSE = SST - SSR
        var ssr = FeatureSelectionTensorOps.GuardedDivide(engine.TensorSquare(sxy), sxx, 0.0, NumOps.Zero);
        var sse = engine.TensorSubtract(FeatureSelectionTensorOps.FilledLike(sxx, sst), ssr);

        // F-statistic: (SSR/1) / (SSE/(n-2)); a perfect linear fit leaves no error and gets F = +inf
        var fScores = n > 2
            ? engine.TensorWhere(
                informative,
                FeatureSelectionTensorOps.SeparationAwareRatio(ssr, engine.TensorDivideScalar(sse, NumOps.FromDouble(n - 2)), 1e-10),
                zeros)
            : zeros;
        _fScores = fScores.ToVector();

        // p = P(F(1, n - 2) > F) from the F distribution (scipy.stats.f.sf)
        _pValues = n > 2
            ? FeatureSelectionTensorOps.FSurvival(fScores, 1, n - 2).ToVector()
            : FeatureSelectionTensorOps.FilledLike(fScores, NumOps.One).ToVector();

        // Select top features by F-score
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(fScores, Math.Min(_nFeaturesToSelect, p));

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

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
