using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Smoothly Clipped Absolute Deviation (SCAD) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SCAD is a nonconvex penalty that reduces bias for large coefficients while
/// maintaining sparsity. Unlike Lasso, SCAD doesn't overshrink large coefficients,
/// providing near-unbiased estimates for significant features.
/// </para>
/// <para><b>For Beginners:</b> SCAD is like an improved version of Lasso. Lasso
/// tends to shrink all coefficients, even important ones. SCAD lets truly important
/// features keep their full strength while still zeroing out unimportant ones.
/// This gives you better feature selection with less bias.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SCADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _a;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SCADSelector(
        int nFeaturesToSelect = 10,
        double lambda = 1.0,
        double a = 3.7,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (a <= 2)
            throw new ArgumentException("SCAD parameter 'a' must be greater than 2.", nameof(a));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _a = a;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SCADSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Standardize data
        var X = new double[n, p];
        var y = new double[n];
        var means = new double[p];
        var stds = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += Math.Pow(NumOps.ToDouble(data[i, j]) - means[j], 2);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - means[j]) / stds[j];
            y[i] = NumOps.ToDouble(target[i]) - yMean;
        }

        // Coordinate descent with SCAD penalty
        _coefficients = new double[p];
        var residuals = new double[n];
        Array.Copy(y, residuals, n);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                double oldBeta = _coefficients[j];

                // Compute partial residual
                double partial = 0;
                for (int i = 0; i < n; i++)
                    partial += X[i, j] * (residuals[i] + oldBeta * X[i, j]);
                partial /= n;

                // Apply SCAD thresholding
                double newBeta = SCADThreshold(partial, _lambda, _a);

                if (Math.Abs(newBeta - oldBeta) > 1e-10)
                {
                    // Update residuals
                    for (int i = 0; i < n; i++)
                        residuals[i] -= (newBeta - oldBeta) * X[i, j];
                }

                _coefficients[j] = newBeta;
                maxChange = Math.Max(maxChange, Math.Abs(newBeta - oldBeta));
            }

            if (maxChange < _tolerance)
                break;
        }

        // Select top features by absolute coefficient
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _coefficients
            .Select((c, idx) => (Coef: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.Coef)
            .Take(nToSelect)
            .Where(x => x.Coef > 1e-10)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        if (_selectedIndices.Length == 0)
        {
            // Fall back to top by absolute coefficient even if small
            _selectedIndices = _coefficients
                .Select((c, idx) => (Coef: Math.Abs(c), Index: idx))
                .OrderByDescending(x => x.Coef)
                .Take(nToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double SCADThreshold(double z, double lambda, double a)
    {
        double absZ = Math.Abs(z);

        if (absZ < 2 * lambda)
        {
            // Soft thresholding region (like Lasso)
            return Math.Sign(z) * Math.Max(0, absZ - lambda);
        }
        else if (absZ <= a * lambda)
        {
            // Transition region
            double numer = (a - 1) * z - Math.Sign(z) * a * lambda;
            return numer / (a - 2);
        }
        else
        {
            // No shrinkage for large coefficients
            return z;
        }
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SCADSelector has not been fitted.");

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
        throw new NotSupportedException("SCADSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SCADSelector has not been fitted.");

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
