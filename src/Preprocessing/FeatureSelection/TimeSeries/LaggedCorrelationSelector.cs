using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Lagged correlation selector for time series feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Computes cross-correlation between features and target at various lags
/// to find features that predict future target values.
/// </para>
/// <para><b>For Beginners:</b> This finds features whose current values
/// predict future values of the target. A feature might correlate with
/// the target 3 time steps later, making it useful for forecasting.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LaggedCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _minLag;
    private readonly int _maxLag;

    private double[]? _correlationScores;
    private int[]? _bestLags;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CorrelationScores => _correlationScores;
    public int[]? BestLags => _bestLags;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LaggedCorrelationSelector(
        int nFeaturesToSelect = 10,
        int minLag = 0,
        int maxLag = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (minLag < 0)
            throw new ArgumentException("Min lag must be non-negative.", nameof(minLag));
        if (maxLag < minLag)
            throw new ArgumentException("Max lag must be >= min lag.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minLag = minLag;
        _maxLag = maxLag;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LaggedCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int maxLag = Math.Min(_maxLag, n / 3);

        _correlationScores = new double[p];
        _bestLags = new int[p];

        // Extract target values
        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]);

        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++)
                x[i] = NumOps.ToDouble(data[i, j]);

            double maxCorr = 0;
            int bestLag = 0;

            for (int lag = _minLag; lag <= maxLag; lag++)
            {
                // Correlate x[0:n-lag] with y[lag:n]
                int nValid = n - lag;
                if (nValid < 3) continue;

                double xMean = 0, yMean = 0;
                for (int i = 0; i < nValid; i++)
                {
                    xMean += x[i];
                    yMean += y[i + lag];
                }
                xMean /= nValid;
                yMean /= nValid;

                double ssXY = 0, ssXX = 0, ssYY = 0;
                for (int i = 0; i < nValid; i++)
                {
                    double dx = x[i] - xMean;
                    double dy = y[i + lag] - yMean;
                    ssXY += dx * dy;
                    ssXX += dx * dx;
                    ssYY += dy * dy;
                }

                if (ssXX > 1e-10 && ssYY > 1e-10)
                {
                    double corr = Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
                    if (corr > maxCorr)
                    {
                        maxCorr = corr;
                        bestLag = lag;
                    }
                }
            }

            _correlationScores[j] = maxCorr;
            _bestLags[j] = bestLag;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _correlationScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaggedCorrelationSelector has not been fitted.");

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
        throw new NotSupportedException("LaggedCorrelationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LaggedCorrelationSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
