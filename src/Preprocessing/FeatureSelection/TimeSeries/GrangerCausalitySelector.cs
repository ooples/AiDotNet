using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.TimeSeries;

/// <summary>
/// Granger Causality-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Granger causality tests to select features that help predict the target
/// beyond what the target's own history can predict.
/// </para>
/// <para><b>For Beginners:</b> Granger causality asks: does knowing a feature's
/// past values help predict the target better than just knowing the target's own
/// past? Features that "Granger-cause" the target have genuine predictive value
/// for forecasting.
/// </para>
/// </remarks>
public class GrangerCausalitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxLag;

    private double[]? _grangerFScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxLag => _maxLag;
    public double[]? GrangerFScores => _grangerFScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GrangerCausalitySelector(
        int nFeaturesToSelect = 10,
        int maxLag = 5,
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
            "GrangerCausalitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _grangerFScores = new double[p];
        int effectiveN = n - _maxLag;

        // Compute restricted model RSS (autoregressive model on Y only)
        double rssRestricted = ComputeARModelRSS(y, _maxLag, effectiveN);

        // For each feature, compute unrestricted model RSS
        for (int j = 0; j < p; j++)
        {
            double rssUnrestricted = ComputeGrangerModelRSS(X, y, j, _maxLag, effectiveN);

            // Granger F-statistic
            int dfRestricted = effectiveN - _maxLag;
            int dfUnrestricted = effectiveN - 2 * _maxLag;

            if (dfUnrestricted > 0 && rssUnrestricted > 1e-10)
            {
                double fStat = ((rssRestricted - rssUnrestricted) / _maxLag) /
                              (rssUnrestricted / dfUnrestricted);
                _grangerFScores[j] = Math.Max(0, fStat);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _grangerFScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeARModelRSS(double[] y, int lag, int effectiveN)
    {
        // Build design matrix for AR model
        int n = y.Length;
        var X = new double[effectiveN, lag];
        var yTarget = new double[effectiveN];

        for (int i = 0; i < effectiveN; i++)
        {
            yTarget[i] = y[i + lag];
            for (int l = 0; l < lag; l++)
                X[i, l] = y[i + lag - l - 1];
        }

        return ComputeRSS(X, yTarget, effectiveN, lag);
    }

    private double ComputeGrangerModelRSS(double[,] Xall, double[] y, int j, int lag, int effectiveN)
    {
        // Build design matrix for Granger model (Y lags + X_j lags)
        int n = y.Length;
        var X = new double[effectiveN, 2 * lag];
        var yTarget = new double[effectiveN];

        for (int i = 0; i < effectiveN; i++)
        {
            yTarget[i] = y[i + lag];
            for (int l = 0; l < lag; l++)
            {
                X[i, l] = y[i + lag - l - 1];
                X[i, lag + l] = Xall[i + lag - l - 1, j];
            }
        }

        return ComputeRSS(X, yTarget, effectiveN, 2 * lag);
    }

    private double ComputeRSS(double[,] X, double[] y, int n, int p)
    {
        // Solve OLS and compute RSS
        var XtX = new double[p, p];
        var Xty = new double[p];

        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < p; j++)
            {
                for (int k = 0; k < n; k++)
                    XtX[i, j] += X[k, i] * X[k, j];
            }
            for (int k = 0; k < n; k++)
                Xty[i] += X[k, i] * y[k];
        }

        // Regularization
        for (int i = 0; i < p; i++)
            XtX[i, i] += 1e-6;

        var beta = SolveSystem(XtX, Xty, p);

        // Compute RSS
        double rss = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j < p; j++)
                pred += beta[j] * X[i, j];
            double residual = y[i] - pred;
            rss += residual * residual;
        }

        return rss;
    }

    private double[] SolveSystem(double[,] A, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-10) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
