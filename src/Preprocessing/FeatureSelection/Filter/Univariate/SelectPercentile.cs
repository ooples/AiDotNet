using AiDotNet.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Selects features according to a percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// SelectPercentile selects the top percentile of features based on a scoring
/// function. For example, selecting the top 10% of features ranked by F-score.
/// </para>
/// <para>
/// This is similar to SelectKBest but uses a relative threshold (percentile)
/// instead of an absolute number of features.
/// </para>
/// <para><b>For Beginners:</b> Instead of specifying an exact number of features:
/// - SelectPercentile(50) keeps the top 50% of features
/// - The actual number depends on your original feature count
/// - Useful when you want a proportion, not an absolute count
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _percentile;
    private readonly SelectKBestScoreFunc _scoringFunction;

    // Fitted parameters
    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the percentile of features to select (0-100).
    /// </summary>
    public double Percentile => _percentile;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoringFunction => _scoringFunction;

    /// <summary>
    /// Gets the scores for each feature.
    /// </summary>
    public Vector<T>? Scores => _scores;

    /// <summary>
    /// Gets the p-values for each feature, or null when the scoring function has none (mutual information).
    /// </summary>
    public Vector<T>? PValues => _pValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectPercentile{T}"/>.
    /// </summary>
    /// <param name="percentile">Percentile of features to select (0-100). Defaults to 10.</param>
    /// <param name="scoringFunction">The scoring function to use. Defaults to FClassif, scikit-learn's default (f_classif).</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SelectPercentile(
        double percentile = 10.0,
        SelectKBestScoreFunc scoringFunction = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile < 0 || percentile > 100)
        {
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));
        }

        _percentile = percentile;
        _scoringFunction = scoringFunction;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectPercentile requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing feature scores.
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
        var (scores, pValues) = StatisticalTestHelper.ComputeScores(data, target, _scoringFunction);
        _scores = scores;
        _pValues = pValues;

        // scikit-learn's percentile rule: scores above the (100 - percentile)-th percentile, then ties at it
        _selectedIndices = FeatureScoringTensorOps.PercentileSupport(FilterStatisticTensorOps.AsTensor(scores), _percentile);
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
    /// Transforms the data by selecting top percentile features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SelectPercentile has not been fitted.");
        }

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectPercentile does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SelectPercentile has not been fitted.");
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
