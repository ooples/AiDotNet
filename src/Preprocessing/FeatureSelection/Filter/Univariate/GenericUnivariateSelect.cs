using AiDotNet.Helpers;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Generic univariate feature selector with configurable mode.
/// </summary>
/// <remarks>
/// <para>
/// GenericUnivariateSelect provides a unified interface to all univariate
/// feature selection methods. You can choose between k_best, percentile,
/// fpr, fdr, and fwe modes.
/// </para>
/// <para>
/// This is useful when you want to experiment with different selection
/// strategies without changing your code structure.
/// </para>
/// <para><b>For Beginners:</b> This is a "swiss army knife" selector:
/// - k_best: Select exactly k features
/// - percentile: Select top X% of features
/// - fpr: Select based on false positive rate
/// - fdr: Select based on false discovery rate
/// - fwe: Select based on family-wise error rate
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class GenericUnivariateSelect<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly UnivariateSelectMode _mode;
    private readonly SelectKBestScoreFunc _scoringFunction;
    private readonly object _param;

    // Fitted parameters
    private double[]? _scores;
    private double[]? _pValues;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the selection mode.
    /// </summary>
    public UnivariateSelectMode Mode => _mode;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoringFunction => _scoringFunction;

    /// <summary>
    /// Gets the mode parameter (k, percentile, or alpha depending on mode).
    /// </summary>
    public object Param => _param;

    /// <summary>
    /// Gets the scores for each feature.
    /// </summary>
    public double[]? Scores => _scores;

    /// <summary>
    /// Gets the p-values for each feature.
    /// </summary>
    public double[]? PValues => _pValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="GenericUnivariateSelect{T}"/>.
    /// </summary>
    /// <param name="mode">The selection mode to use.</param>
    /// <param name="param">The mode-specific parameter (k for KBest, percentile for Percentile, alpha for FPR/FDR/FWE).</param>
    /// <param name="scoringFunction">The scoring function to use. Defaults to FRegression.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public GenericUnivariateSelect(
        UnivariateSelectMode mode = UnivariateSelectMode.Percentile,
        object? param = null,
        SelectKBestScoreFunc scoringFunction = SelectKBestScoreFunc.FRegression,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _mode = mode;
        _scoringFunction = scoringFunction;

        // Set default parameter based on mode
        if (param is null)
        {
            _param = mode switch
            {
                UnivariateSelectMode.KBest => 10,
                UnivariateSelectMode.Percentile => 10.0,
                UnivariateSelectMode.Fpr => 0.05,
                UnivariateSelectMode.Fdr => 0.05,
                UnivariateSelectMode.Fwe => 0.05,
                _ => 10
            };
        }
        else
        {
            _param = param;
        }

        ValidateParam();
    }

    private void ValidateParam()
    {
        switch (_mode)
        {
            case UnivariateSelectMode.KBest:
                if (_param is not int k || k < 1)
                {
                    throw new ArgumentException("For KBest mode, param must be a positive integer.", nameof(_param));
                }
                break;

            case UnivariateSelectMode.Percentile:
                double percentile = Convert.ToDouble(_param);
                if (percentile <= 0 || percentile > 100)
                {
                    throw new ArgumentException("For Percentile mode, param must be between 0 and 100.", nameof(_param));
                }
                break;

            case UnivariateSelectMode.Fpr:
            case UnivariateSelectMode.Fdr:
            case UnivariateSelectMode.Fwe:
                double alpha = Convert.ToDouble(_param);
                if (alpha <= 0 || alpha >= 1)
                {
                    throw new ArgumentException("For FPR/FDR/FWE modes, param (alpha) must be between 0 and 1.", nameof(_param));
                }
                break;
        }
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GenericUnivariateSelect requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing scores and selecting features.
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

        // Select features based on mode
        _selectedIndices = SelectFeatures(p);

        // Create support mask
        _supportMask = new bool[p];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }

        IsFitted = true;
    }

    private int[] SelectFeatures(int p)
    {
        if (_scores is null || _pValues is null)
        {
            throw new InvalidOperationException("Scores have not been computed.");
        }

        var selectedList = new List<int>();

        switch (_mode)
        {
            case UnivariateSelectMode.KBest:
                int k = Convert.ToInt32(_param);
                k = Math.Min(k, p);
                var sortedByScore = Enumerable.Range(0, p)
                    .OrderByDescending(i => _scores[i])
                    .Take(k)
                    .OrderBy(i => i)
                    .ToList();
                selectedList.AddRange(sortedByScore);
                break;

            case UnivariateSelectMode.Percentile:
                double percentile = Convert.ToDouble(_param);
                int numToSelect = Math.Max(1, (int)Math.Ceiling(p * percentile / 100.0));
                var sortedByScorePercentile = Enumerable.Range(0, p)
                    .OrderByDescending(i => _scores[i])
                    .Take(numToSelect)
                    .OrderBy(i => i)
                    .ToList();
                selectedList.AddRange(sortedByScorePercentile);
                break;

            case UnivariateSelectMode.Fpr:
                double alphaFpr = Convert.ToDouble(_param);
                for (int j = 0; j < p; j++)
                {
                    if (_pValues[j] < alphaFpr)
                    {
                        selectedList.Add(j);
                    }
                }
                break;

            case UnivariateSelectMode.Fdr:
                double alphaFdr = Convert.ToDouble(_param);
                var adjustedFdr = StatisticalTestHelper.BenjaminiHochbergCorrection(_pValues);
                for (int j = 0; j < p; j++)
                {
                    if (adjustedFdr[j] < alphaFdr)
                    {
                        selectedList.Add(j);
                    }
                }
                break;

            case UnivariateSelectMode.Fwe:
                double alphaFwe = Convert.ToDouble(_param);
                var adjustedFwe = StatisticalTestHelper.BonferroniCorrection(_pValues);
                for (int j = 0; j < p; j++)
                {
                    if (adjustedFwe[j] < alphaFwe)
                    {
                        selectedList.Add(j);
                    }
                }
                break;
        }

        // Ensure at least one feature is selected
        if (selectedList.Count == 0)
        {
            int bestIdx = 0;
            double bestScore = _scores[0];
            for (int j = 1; j < p; j++)
            {
                if (_scores[j] > bestScore)
                {
                    bestScore = _scores[j];
                    bestIdx = j;
                }
            }
            selectedList.Add(bestIdx);
        }

        return selectedList.OrderBy(i => i).ToArray();
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
    /// Transforms the data by selecting features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");
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
        throw new NotSupportedException("GenericUnivariateSelect does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");
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

/// <summary>
/// Specifies the mode for GenericUnivariateSelect.
/// </summary>
public enum UnivariateSelectMode
{
    /// <summary>
    /// Select the k highest-scoring features.
    /// </summary>
    KBest,

    /// <summary>
    /// Select the top percentile of features.
    /// </summary>
    Percentile,

    /// <summary>
    /// Select features based on false positive rate threshold.
    /// </summary>
    Fpr,

    /// <summary>
    /// Select features based on false discovery rate (Benjamini-Hochberg).
    /// </summary>
    Fdr,

    /// <summary>
    /// Select features based on family-wise error rate (Bonferroni).
    /// </summary>
    Fwe
}
