using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparse;

/// <summary>
/// Fused Lasso (Total Variation) for feature selection with smoothness constraints.
/// </summary>
/// <remarks>
/// <para>
/// Fused Lasso adds a penalty on the differences between consecutive coefficients,
/// encouraging nearby features to have similar coefficients. This is useful when
/// features have a natural ordering (e.g., spectral data, time series).
/// </para>
/// <para><b>For Beginners:</b> When features are ordered (like wavelengths or time points),
/// you often expect nearby features to behave similarly. Fused Lasso encourages smooth
/// coefficient patterns while still allowing for sparse solutions where coefficients
/// jump abruptly only at important boundaries.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FusedLasso<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _fusedPenalty;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double FusedPenalty => _fusedPenalty;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FusedLasso(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double fusedPenalty = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _fusedPenalty = fusedPenalty;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FusedLasso requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Standardize features
        var means = new double[p];
        var stds = new double[p];
        var X = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;
            means[j] = mean;

            double variance = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            double std = Math.Sqrt(variance / n);
            stds[j] = std > 1e-10 ? std : 1.0;

            for (int i = 0; i < n; i++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - mean) / stds[j];
        }

        // Standardize target
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]) - yMean;

        // Initialize
        _coefficients = new double[p];
        var residuals = (double[])y.Clone();

        // ADMM for Fused Lasso
        var u = new double[p]; // Dual variable for sparsity
        var v = new double[p - 1]; // Dual variable for fusion
        var z1 = new double[p]; // Auxiliary for sparsity
        var z2 = new double[p - 1]; // Auxiliary for fusion

        double rho = 1.0; // ADMM penalty parameter

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var oldCoef = (double[])_coefficients.Clone();

            // Update beta (coefficient) - solve least squares with regularization
            for (int j = 0; j < p; j++)
            {
                // Gradient from data
                double grad = 0;
                double hess = 0;
                for (int i = 0; i < n; i++)
                {
                    grad += X[i, j] * residuals[i];
                    hess += X[i, j] * X[i, j];
                }

                // Add ADMM terms
                double target_val = z1[j] - u[j];
                if (j < p - 1)
                    target_val += z2[j] - v[j];
                if (j > 0)
                    target_val += _coefficients[j - 1] + z2[j - 1] - v[j - 1];

                grad += rho * (_coefficients[j] - target_val);
                hess += rho * (1 + (j < p - 1 ? 1 : 0) + (j > 0 ? 1 : 0));

                double oldVal = _coefficients[j];
                _coefficients[j] = hess > 1e-10 ? _coefficients[j] + grad / hess : _coefficients[j];

                // Update residuals
                for (int i = 0; i < n; i++)
                    residuals[i] += X[i, j] * (oldVal - _coefficients[j]);
            }

            // Update z1 (sparsity constraint) with soft thresholding
            for (int j = 0; j < p; j++)
            {
                double val = _coefficients[j] + u[j];
                double threshold = _alpha / rho;
                if (val > threshold)
                    z1[j] = val - threshold;
                else if (val < -threshold)
                    z1[j] = val + threshold;
                else
                    z1[j] = 0;
            }

            // Update z2 (fusion constraint) with soft thresholding
            for (int j = 0; j < p - 1; j++)
            {
                double val = _coefficients[j + 1] - _coefficients[j] + v[j];
                double threshold = _fusedPenalty / rho;
                if (val > threshold)
                    z2[j] = val - threshold;
                else if (val < -threshold)
                    z2[j] = val + threshold;
                else
                    z2[j] = 0;
            }

            // Update dual variables
            for (int j = 0; j < p; j++)
                u[j] += _coefficients[j] - z1[j];
            for (int j = 0; j < p - 1; j++)
                v[j] += _coefficients[j + 1] - _coefficients[j] - z2[j];

            // Check convergence
            double maxChange = 0;
            for (int j = 0; j < p; j++)
                maxChange = Math.Max(maxChange, Math.Abs(_coefficients[j] - oldCoef[j]));

            if (maxChange < _tolerance)
                break;
        }

        // Select features by absolute coefficient magnitude
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _coefficients
            .Select((c, idx) => (Coef: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.Coef)
            .Take(numToSelect)
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
            throw new InvalidOperationException("FusedLasso has not been fitted.");

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
        throw new NotSupportedException("FusedLasso does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FusedLasso has not been fitted.");

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
