using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Lag-based Feature Selection for Time Series.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their lagged correlations with the target,
/// identifying features that have predictive power at different time lags.
/// </para>
/// <para><b>For Beginners:</b> In time series, past values often predict future
/// values. This selector finds features whose past values (lags) are strongly
/// correlated with the target, helping you identify which features are most
/// useful for forecasting.
/// </para>
/// </remarks>
public class LagFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

    private double[]? _bestLagCorrelations;
    private int[]? _bestLags;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public double[]? BestLagCorrelations => _bestLagCorrelations;
    public int[]? BestLags => _bestLags;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LagFeatureSelector(
        int nFeaturesToSelect = 10,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LagFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _bestLagCorrelations = new double[p];
        _bestLags = new int[p];

        for (int j = 0; j < p; j++)
        {
            double bestCorr = 0;
            int bestLag = 0;

            for (int lag = 0; lag <= Math.Min(_maxLag, n - 10); lag++)
            {
                double corr = ComputeLaggedCorrelation(X, y, j, lag, n);
                if (Math.Abs(corr) > Math.Abs(bestCorr))
                {
                    bestCorr = corr;
                    bestLag = lag;
                }
            }

            _bestLagCorrelations[j] = Math.Abs(bestCorr);
            _bestLags[j] = bestLag;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _bestLagCorrelations[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeLaggedCorrelation(double[,] X, double[] y, int j, int lag, int n)
    {
        int effectiveN = n - lag;
        if (effectiveN < 2) return 0;

        double xMean = 0, yMean = 0;
        for (int i = 0; i < effectiveN; i++)
        {
            xMean += X[i, j];       // x at time t
            yMean += y[i + lag];    // y at time t + lag
        }
        xMean /= effectiveN;
        yMean /= effectiveN;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < effectiveN; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i + lag] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LagFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("LagFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LagFeatureSelector has not been fitted.");

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
