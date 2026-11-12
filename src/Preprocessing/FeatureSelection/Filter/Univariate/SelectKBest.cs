using AiDotNet.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Selects the K highest-scoring features according to a scoring function.
/// </summary>
/// <remarks>
/// <para>
/// SelectKBest computes a score for each feature based on the relationship between
/// the feature and the target variable, then selects the top K features with the
/// highest scores.
/// </para>
/// <para>
/// Built-in scoring functions include:
/// - F-score for regression (linear relationship)
/// - Mutual information (any relationship type)
/// </para>
/// <para><b>For Beginners:</b> Not all features are equally useful for prediction.
/// SelectKBest helps you:
/// - Reduce the number of features to improve model speed
/// - Remove noisy features that might hurt model accuracy
/// - Find the most informative features for understanding your problem
///
/// Example: From 100 features, select the 10 most related to your target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectKBest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _k;
    private readonly SelectKBestScoreFunc _scoreFunc;

    // Fitted parameters
    private Vector<T>? _scores;
    private Vector<T>? _pvalues;
    private int[]? _selectedFeatures;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoreFunc => _scoreFunc;

    /// <summary>
    /// Gets the computed scores for each feature.
    /// </summary>
    public Vector<T>? Scores => _scores;

    /// <summary>
    /// Gets the p-values for each feature, or null when the scoring function has none (mutual information).
    /// </summary>
    public Vector<T>? PValues => _pvalues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedFeatures => _selectedFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectKBest{T}"/>.
    /// </summary>
    /// <param name="k">Number of features to select. Defaults to 10.</param>
    /// <param name="scoreFunc">Scoring function to use. Defaults to FClassif, scikit-learn's default (f_classif).</param>
    /// <param name="columnIndices">The column indices to consider, or null for all columns.</param>
    public SelectKBest(
        int k = 10,
        SelectKBestScoreFunc scoreFunc = SelectKBestScoreFunc.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
        {
            throw new ArgumentException("K must be at least 1.", nameof(k));
        }

        _k = k;
        _scoreFunc = scoreFunc;
    }

    /// <summary>
    /// Fits the selector using variance-based scoring when no target is provided.
    /// </summary>
    /// <remarks>
    /// When called without target values, this method falls back to variance-based
    /// feature selection. Features with higher variance are scored higher.
    /// For supervised selection, use Fit(Matrix&lt;T&gt; data, Vector&lt;T&gt; target) instead.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        int n = data.Rows;
        var engine = AiDotNetEngine.Current;

        // Use the sample variance (divisor n - 1) as the score for unsupervised selection; columns outside the
        // processed set score -infinity with p-value 1, processed ones have no p-value (0)
        var processed = FilterStatisticTensorOps.AsTensor(FilterStatisticTensorOps.SelectColumns(data, columnsToProcess));
        var variance = engine.TensorDivideScalar(
            FilterStatisticTensorOps.ColumnSumOfSquaredDeviations(processed), NumOps.FromDouble(n > 1 ? n - 1 : 1));
        var scores = FeatureSelectionTensorOps.ScatterToFeatures(
            variance, columnsToProcess, _nInputFeatures, NumOps.FromDouble(double.NegativeInfinity));
        _scores = scores.ToVector();
        _pvalues = FeatureSelectionTensorOps.ScatterToFeatures(
            FilterStatisticTensorOps.Filled(new[] { columnsToProcess.Length }, NumOps.Zero),
            columnsToProcess, _nInputFeatures, NumOps.One).ToVector();

        // Select top K features by variance
        _selectedFeatures = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_k, _nInputFeatures));

        IsFitted = true;
    }

    /// <summary>
    /// Fits the selector by computing feature scores based on the target.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Score the processed columns; the others score -infinity with p-value 1
        var (scores, pValues) = StatisticalTestHelper.ComputeScores(
            FilterStatisticTensorOps.SelectColumns(data, columnsToProcess), target, _scoreFunc);
        var allScores = FeatureSelectionTensorOps.ScatterToFeatures(
            FilterStatisticTensorOps.AsTensor(scores), columnsToProcess, _nInputFeatures,
            NumOps.FromDouble(double.NegativeInfinity));
        _scores = allScores.ToVector();
        _pvalues = pValues is null
            ? null
            : FeatureSelectionTensorOps.ScatterToFeatures(
                FilterStatisticTensorOps.AsTensor(pValues), columnsToProcess, _nInputFeatures, NumOps.One).ToVector();

        // Select top K features (equal scores go to the lower column)
        _selectedFeatures = FeatureSelectionTensorOps.TopKOfColumns(allScores, columnsToProcess, _k);

        IsFitted = true;
    }

    /// <summary>
    /// Fits the selector and transforms the data in one step.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Selects the top K features from the data.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        return FilterStatisticTensorOps.SelectColumns(data, _selectedFeatures);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectKBest does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedFeatures is null)
        {
            throw new InvalidOperationException("Selector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        var selectedSet = new HashSet<int>(_selectedFeatures);

        for (int i = 0; i < _nInputFeatures; i++)
        {
            mask[i] = selectedSet.Contains(i);
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedFeatures is null)
        {
            return Array.Empty<string>();
        }

        var names = new string[_selectedFeatures.Length];
        for (int i = 0; i < _selectedFeatures.Length; i++)
        {
            int col = _selectedFeatures[i];
            names[i] = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";
        }

        return names;
    }
}

/// <summary>
/// Scoring functions for SelectKBest.
/// </summary>
public enum SelectKBestScoreFunc
{
    /// <summary>
    /// F-value for regression (correlation-based).
    /// </summary>
    FRegression,

    /// <summary>
    /// Mutual information for regression (captures non-linear relationships).
    /// </summary>
    MutualInfoRegression,

    /// <summary>
    /// ANOVA F-value for classification.
    /// </summary>
    FClassif,

    /// <summary>
    /// Chi-squared statistic for classification (features must be non-negative).
    /// </summary>
    Chi2
}
