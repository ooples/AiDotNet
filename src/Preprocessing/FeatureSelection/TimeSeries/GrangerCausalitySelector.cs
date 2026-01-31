using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Granger Causality-based feature selection for time series.
/// </summary>
/// <remarks>
/// <para>
/// Tests whether past values of a feature help predict future values of
/// the target beyond what past target values alone can predict.
/// </para>
/// <para><b>For Beginners:</b> Granger causality asks: "Does knowing the
/// past of feature X help predict Y's future better than just knowing Y's
/// past?" Features that pass this test are likely truly predictive.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GrangerCausalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;
    private readonly double _significanceLevel;

    private double[]? _fStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GrangerCausalitySelector(
        int nFeaturesToSelect = 10,
        int maxLag = 5,
        double significanceLevel = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxLag < 1)
            throw new ArgumentException("Max lag must be at least 1.", nameof(maxLag));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxLag = maxLag;
        _significanceLevel = significanceLevel;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GrangerCausalitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int lag = Math.Min(_maxLag, (n - 10) / 3);  // Leave enough samples

        if (lag < 1) lag = 1;

        _fStatistics = new double[p];
        _pValues = new double[p];

        // Extract target values
        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]);

        int nValid = n - lag;
        if (nValid < 5)
            throw new ArgumentException("Not enough samples for Granger causality test.");

        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++)
                x[i] = NumOps.ToDouble(data[i, j]);

            // Restricted model: y[t] ~ y[t-1], ..., y[t-lag]
            var restrictedSSR = ComputeAutoRegressiveSSR(y, lag, nValid);

            // Unrestricted model: y[t] ~ y[t-1], ..., y[t-lag], x[t-1], ..., x[t-lag]
            var unrestrictedSSR = ComputeVARSSR(y, x, lag, nValid);

            // F-statistic
            int dfRestricted = nValid - lag;
            int dfUnrestricted = nValid - 2 * lag;

            if (dfUnrestricted > 0 && unrestrictedSSR > 1e-10)
            {
                double fStat = ((restrictedSSR - unrestrictedSSR) / lag) /
                              (unrestrictedSSR / dfUnrestricted);
                _fStatistics[j] = fStat;
                _pValues[j] = 1 - FDistributionCDF(fStat, lag, dfUnrestricted);
            }
            else
            {
                _fStatistics[j] = 0;
                _pValues[j] = 1;
            }
        }

        // Select features with significant Granger causality
        var significant = new List<int>();
        for (int j = 0; j < p; j++)
            if (_pValues[j] < _significanceLevel)
                significant.Add(j);

        if (significant.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = significant
                .OrderBy(j => _pValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            // Fall back to top by F-statistic
            _selectedIndices = _fStatistics
                .Select((f, idx) => (F: f, Index: idx))
                .OrderByDescending(x => x.F)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double ComputeAutoRegressiveSSR(double[] y, int lag, int nValid)
    {
        // Simple AR regression: y[t] = a0 + a1*y[t-1] + ... + alag*y[t-lag]
        // Using normal equations

        // Build design matrix
        var X = new double[nValid, lag + 1];  // +1 for intercept
        var Y = new double[nValid];

        for (int i = 0; i < nValid; i++)
        {
            int t = i + lag;
            Y[i] = y[t];
            X[i, 0] = 1;  // Intercept
            for (int l = 1; l <= lag; l++)
                X[i, l] = y[t - l];
        }

        var predicted = FitAndPredict(X, Y, nValid, lag + 1);
        double ssr = 0;
        for (int i = 0; i < nValid; i++)
            ssr += Math.Pow(Y[i] - predicted[i], 2);

        return ssr;
    }

    private double ComputeVARSSR(double[] y, double[] x, int lag, int nValid)
    {
        // VAR regression: y[t] = a0 + a1*y[t-1] + ... + alag*y[t-lag] + b1*x[t-1] + ... + blag*x[t-lag]
        int nCols = 2 * lag + 1;  // intercept + lag y's + lag x's

        var X = new double[nValid, nCols];
        var Y = new double[nValid];

        for (int i = 0; i < nValid; i++)
        {
            int t = i + lag;
            Y[i] = y[t];
            X[i, 0] = 1;
            for (int l = 1; l <= lag; l++)
            {
                X[i, l] = y[t - l];
                X[i, lag + l] = x[t - l];
            }
        }

        var predicted = FitAndPredict(X, Y, nValid, nCols);
        double ssr = 0;
        for (int i = 0; i < nValid; i++)
            ssr += Math.Pow(Y[i] - predicted[i], 2);

        return ssr;
    }

    private double[] FitAndPredict(double[,] X, double[] Y, int n, int p)
    {
        // OLS: beta = (X'X)^-1 X'Y
        // Simplified implementation

        var XtX = new double[p, p];
        var XtY = new double[p];

        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                    XtX[i, j] += X[k, i] * X[k, j];
            }
            for (int k = 0; k < n; k++)
                XtY[i] += X[k, i] * Y[k];
        }

        // Regularize
        for (int i = 0; i < p; i++)
            XtX[i, i] += 1e-6;

        var beta = SolveLinearSystem(XtX, XtY, p);

        var predicted = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                predicted[i] += X[i, j] * beta[j];
        }

        return predicted;
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Gaussian elimination
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                augmented[i, j] = A[i, j];
            augmented[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
            {
                double temp = augmented[col, j];
                augmented[col, j] = augmented[maxRow, j];
                augmented[maxRow, j] = temp;
            }

            if (Math.Abs(augmented[col, col]) < 1e-10)
                continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = augmented[row, col] / augmented[col, col];
                for (int j = col; j <= n; j++)
                    augmented[row, j] -= factor * augmented[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = augmented[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= augmented[i, j] * x[j];
            if (Math.Abs(augmented[i, i]) > 1e-10)
                x[i] /= augmented[i, i];
        }

        return x;
    }

    private double FDistributionCDF(double x, int d1, int d2)
    {
        // Approximation using incomplete beta function relationship
        if (x <= 0) return 0;
        double a = d1 / 2.0;
        double b = d2 / 2.0;
        double z = d1 * x / (d1 * x + d2);
        return IncompleteBeta(z, a, b);
    }

    private double IncompleteBeta(double x, double a, double b)
    {
        // Continued fraction approximation
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;

        // Use continued fraction
        double bt = Math.Exp(a * Math.Log(x) + b * Math.Log(1 - x));
        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(x, a, b) / a;
        else
            return 1 - bt * BetaCF(1 - x, b, a) / b;
    }

    private double BetaCF(double x, double a, double b)
    {
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= 100; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < 1e-7) break;
        }

        return h;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GrangerCausalitySelector has not been fitted.");

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
        throw new NotSupportedException("GrangerCausalitySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GrangerCausalitySelector has not been fitted.");

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
