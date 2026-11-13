using AiDotNet.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Selects features based on a false discovery rate test.
/// </summary>
/// <remarks>
/// <para>
/// SelectFdr applies the Benjamini-Hochberg procedure to control the expected
/// proportion of false discoveries (incorrectly selected features) among the
/// selected features.
/// </para>
/// <para>
/// Unlike SelectFpr which controls false positives among ALL features,
/// SelectFdr controls false discoveries among SELECTED features only.
/// </para>
/// <para><b>For Beginners:</b> FDR correction is more powerful than FPR:
/// - FPR: "At most 5% of all 1000 features are false positives"
/// - FDR: "At most 5% of the 50 features I selected are false positives"
/// - FDR allows more features to be selected while controlling errors
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectFdr<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _alpha;
    private readonly SelectKBestScoreFunc _scoringFunction;

    // Fitted parameters
    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private Vector<T>? _adjustedPValues;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the significance level (alpha) for FDR control.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoringFunction => _scoringFunction;

    /// <summary>
    /// Gets the scores for each feature.
    /// </summary>
    public Vector<T>? Scores => _scores;

    /// <summary>
    /// Gets the original p-values for each feature.
    /// </summary>
    public Vector<T>? PValues => _pValues;

    /// <summary>
    /// Gets the FDR-adjusted p-values.
    /// </summary>
    public Vector<T>? AdjustedPValues => _adjustedPValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectFdr{T}"/>.
    /// </summary>
    /// <param name="alpha">Target FDR level. Defaults to 0.05.</param>
    /// <param name="scoringFunction">The scoring function to use. Defaults to FClassif, scikit-learn's default (f_classif); it must produce p-values.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SelectFdr(
        double alpha = 0.05,
        SelectKBestScoreFunc scoringFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentException("Alpha must be between 0 and 1 (exclusive).", nameof(alpha));
        }

        _alpha = alpha;
        _scoringFunction = scoringFunction;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectFdr requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing FDR-adjusted p-values.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // scores and p-values of every feature (scikit-learn's univariate scoring functions)
        var (scores, pValues) = StatisticalTestHelper.ComputeScoresWithPValues(data, target, _scoringFunction);
        _scores = scores;
        _pValues = pValues;

        // Benjamini-Hochberg: keep every feature whose adjusted p-value is at most alpha, which is
        // scikit-learn's SelectFdr rule (p_(r) <= alpha r / n for some rank r at or above the feature's)
        _adjustedPValues = StatisticalTestHelper.BenjaminiHochbergCorrection(pValues);
        var keep = FeatureScoringTensorOps.AtMost(FilterStatisticTensorOps.AsTensor(_adjustedPValues), NumOps.FromDouble(_alpha));

        // As in scikit-learn, no feature passing the test selects none
        _selectedIndices = FilterStatisticTensorOps.MaskPositions(keep);
        _supportMask = FeatureScoringTensorOps.SupportMask(_selectedIndices, p);

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting features passing FDR threshold.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SelectFdr has not been fitted.");
        }

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectFdr does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SelectFdr has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
