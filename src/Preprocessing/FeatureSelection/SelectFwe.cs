using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Selects features based on a family-wise error rate test.
/// </summary>
/// <remarks>
/// <para>
/// SelectFwe applies Bonferroni correction to control the probability of making
/// even one false positive among all selected features.
/// </para>
/// <para>
/// This is the most conservative multiple testing correction, dividing the
/// significance threshold by the number of tests.
/// </para>
/// <para><b>For Beginners:</b> FWER is the strictest correction:
/// - Controls the probability of ANY false positive
/// - Uses Bonferroni: alpha/number_of_features
/// - Very conservative: may miss true positives
/// - Best when false positives are costly
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SelectFwe<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _alpha;
    private readonly SelectKBestScoreFunc _scoringFunction;

    // Fitted parameters
    private double[]? _scores;
    private double[]? _pValues;
    private double[]? _adjustedPValues;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the family-wise significance level (alpha).
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoringFunction => _scoringFunction;

    /// <summary>
    /// Gets the scores for each feature.
    /// </summary>
    public double[]? Scores => _scores;

    /// <summary>
    /// Gets the original p-values for each feature.
    /// </summary>
    public double[]? PValues => _pValues;

    /// <summary>
    /// Gets the Bonferroni-adjusted p-values.
    /// </summary>
    public double[]? AdjustedPValues => _adjustedPValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectFwe{T}"/>.
    /// </summary>
    /// <param name="alpha">Family-wise error rate threshold. Defaults to 0.05.</param>
    /// <param name="scoringFunction">The scoring function to use. Defaults to FRegression.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SelectFwe(
        double alpha = 0.05,
        SelectKBestScoreFunc scoringFunction = SelectKBestScoreFunc.FRegression,
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
            "SelectFwe requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing Bonferroni-corrected p-values.
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
        int n = data.Rows;
        int p = data.Columns;

        // Convert to double arrays
        var X = new double[n, p];
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Compute scores and p-values for each feature
        _scores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
            {
                featureValues[i] = X[i, j];
            }

            var (score, pValue) = StatisticalTestHelper.ComputeScore(featureValues, y, _scoringFunction);
            _scores[j] = score;
            _pValues[j] = pValue;
        }

        // Apply Bonferroni correction
        _adjustedPValues = StatisticalTestHelper.BonferroniCorrection(_pValues);

        // Select features with adjusted p-value < alpha
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (_adjustedPValues[j] < _alpha)
            {
                selectedList.Add(j);
            }
        }

        // Ensure at least one feature is selected
        if (selectedList.Count == 0)
        {
            int bestIdx = 0;
            double minPValue = _adjustedPValues[0];
            for (int j = 1; j < p; j++)
            {
                if (_adjustedPValues[j] < minPValue)
                {
                    minPValue = _adjustedPValues[j];
                    bestIdx = j;
                }
            }
            selectedList.Add(bestIdx);
        }

        _selectedIndices = selectedList.OrderBy(i => i).ToArray();

        // Create support mask
        _supportMask = new bool[p];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }

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
    /// Transforms the data by selecting features passing FWER threshold.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SelectFwe has not been fitted.");
        }

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectFwe does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SelectFwe has not been fitted.");
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
