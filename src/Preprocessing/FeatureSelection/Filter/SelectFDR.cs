using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Select features using Benjamini-Hochberg False Discovery Rate control.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Benjamini-Hochberg procedure to control the expected proportion of
/// false discoveries among selected features. This provides more power than
/// FWER control while still limiting false positives.
/// </para>
/// <para><b>For Beginners:</b> When testing many features, some will look significant
/// by chance. FDR control says "among all the features we select, we want at most
/// 5% (or your alpha) to be false positives." This is less strict than FWER but
/// more practical for high-dimensional data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectFDR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _alpha;
    private readonly SelectKBestScoreFunc _scoreFunction;
    private readonly Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)>? _customScoreFunction;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private Vector<T>? _adjustedPValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Alpha => _alpha;
    /// <summary>
    /// Gets the built-in scoring function, used when no custom scorer was supplied.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunction => _scoreFunction;

    /// <summary>Gets the score of every feature after fitting.</summary>
    public Vector<T>? Scores => _scores;

    /// <summary>Gets the p-value of every feature after fitting.</summary>
    public Vector<T>? PValues => _pValues;
    /// <summary>Gets the Benjamini-Hochberg adjusted p-values after fitting.</summary>
    public Vector<T>? AdjustedPValues => _adjustedPValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a selector that scores features with a built-in scikit-learn scoring function.
    /// </summary>
    /// <param name="alpha">The false discovery rate to control.</param>
    /// <param name="scoreFunction">
    /// The scoring function; defaults to <see cref="SelectKBestScoreFunc.FClassif"/> (scikit-learn's default); it must produce p-values.
    /// </param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectFDR(
        double alpha = 0.05,
        SelectKBestScoreFunc scoreFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        _alpha = alpha;
        _scoreFunction = scoreFunction;
    }

    /// <summary>
    /// Creates a selector with a custom scorer, like scikit-learn's <c>score_func</c>: it receives the data and the
    /// target and returns one score per feature (higher is better) and one p-value per feature.
    /// </summary>
    /// <param name="alpha">The false discovery rate to control.</param>
    /// <param name="scoreFunction">The scorer.</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectFDR(
        double alpha,
        Func<Matrix<T>, Vector<T>, (Vector<T> Scores, Vector<T>? PValues)> scoreFunction,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        if (scoreFunction is null)
            throw new ArgumentNullException(nameof(scoreFunction));

        _alpha = alpha;
        _scoreFunction = SelectKBestScoreFunc.FClassif;
        _customScoreFunction = scoreFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectFDR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        Vector<T> scores;
        Vector<T>? pValues;
        if (_customScoreFunction is not null)
        {
            (scores, pValues) = _customScoreFunction(data, target);
            if (scores is null || scores.Length != p)
                throw new InvalidOperationException("The scoring function must return one score per feature.");
            if (pValues is not null && pValues.Length != p)
                throw new InvalidOperationException("The scoring function must return one p-value per feature.");
        }
        else
        {
            (scores, pValues) = StatisticalTestHelper.ComputeScores(data, target, _scoreFunction);
        }

        _scores = scores;
        _pValues = pValues;

        if (pValues is null)
            throw new InvalidOperationException("SelectFDR needs a scoring function that returns p-values.");

        // Benjamini-Hochberg: keep every feature whose adjusted p-value is at most alpha (scikit-learn's SelectFdr);
        // as in scikit-learn, no feature passing selects none
        _adjustedPValues = StatisticalTestHelper.BenjaminiHochbergCorrection(pValues);
        _selectedIndices = FilterStatisticTensorOps.MaskPositions(FeatureSelectionTensorOps.AtMost(
            FilterStatisticTensorOps.AsTensor(_adjustedPValues), NumOps.FromDouble(_alpha)));

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
            throw new InvalidOperationException("SelectFDR has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectFDR does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectFDR has not been fitted.");

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
