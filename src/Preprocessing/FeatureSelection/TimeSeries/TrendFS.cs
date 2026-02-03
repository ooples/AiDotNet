using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Trend-based feature selection for time series data.
/// </summary>
/// <remarks>
/// <para>
/// Trend-based feature selection identifies features that exhibit significant
/// temporal trends (upward or downward movement over time). Features with strong
/// trends may be useful for long-term forecasting or detecting changes.
/// </para>
/// <para><b>For Beginners:</b> This method looks for features that are consistently
/// going up or down over time, like a stock price in a bull market. Features without
/// clear direction (just random fluctuation) score low. It's useful when you care
/// about the overall trajectory, not just short-term patterns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TrendFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _absoluteTrend;

    private double[]? _trendScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool AbsoluteTrend => _absoluteTrend;
    public double[]? TrendScores => _trendScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TrendFS(
        int nFeaturesToSelect = 10,
        bool absoluteTrend = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _absoluteTrend = absoluteTrend;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
        {
            _trendScores = new double[p];
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        _trendScores = new double[p];

        // Time variable (0, 1, 2, ..., n-1)
        double tMean = (n - 1) / 2.0;
        double tVar = 0;
        for (int i = 0; i < n; i++)
            tVar += (i - tMean) * (i - tMean);

        for (int j = 0; j < p; j++)
        {
            // Compute feature mean
            double yMean = 0;
            for (int i = 0; i < n; i++)
                yMean += NumOps.ToDouble(data[i, j]);
            yMean /= n;

            // Compute linear regression slope (trend)
            double covariance = 0;
            for (int i = 0; i < n; i++)
            {
                double yDiff = NumOps.ToDouble(data[i, j]) - yMean;
                covariance += (i - tMean) * yDiff;
            }

            double slope = tVar > 1e-10 ? covariance / tVar : 0;

            // Compute R-squared to measure trend strength
            double ssTotal = 0;
            double ssResidual = 0;
            double intercept = yMean - slope * tMean;

            for (int i = 0; i < n; i++)
            {
                double actual = NumOps.ToDouble(data[i, j]);
                double predicted = intercept + slope * i;
                double diff = actual - yMean;
                ssTotal += diff * diff;
                ssResidual += (actual - predicted) * (actual - predicted);
            }

            double rSquared = ssTotal > 1e-10 ? 1 - (ssResidual / ssTotal) : 0;

            // Score combines trend magnitude and fit quality
            double trendMagnitude = _absoluteTrend ? Math.Abs(slope) : slope;
            _trendScores[j] = trendMagnitude * Math.Sqrt(rSquared);
        }

        // Select top features by trend score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _trendScores
            .Select((s, idx) => (Score: _absoluteTrend ? s : Math.Abs(s), Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendFS has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TrendFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrendFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
