using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Minimax Concave Penalty (MCP) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// MCP is a nonconvex penalty similar to SCAD that provides nearly unbiased
/// estimates for large coefficients while inducing sparsity. It has a concave
/// shape that reduces shrinkage bias.
/// </para>
/// <para><b>For Beginners:</b> Like SCAD, MCP is an improvement over Lasso.
/// It strongly penalizes small coefficients (pushing them to zero) but
/// gradually reduces the penalty for larger coefficients. This means truly
/// important features keep their full effect while noise features are eliminated.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MCPSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _gamma;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MCPSelector(
        int nFeaturesToSelect = 10,
        double lambda = 1.0,
        double gamma = 3.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (gamma <= 1)
            throw new ArgumentException("MCP parameter 'gamma' must be greater than 1.", nameof(gamma));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _gamma = gamma;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MCPSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Coordinate descent with MCP penalty
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

                // Apply MCP thresholding
                double newBeta = MCPThreshold(partial, _lambda, _gamma);

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

    private double MCPThreshold(double z, double lambda, double gamma)
    {
        double absZ = Math.Abs(z);

        if (absZ <= gamma * lambda)
        {
            // Soft thresholding with MCP adjustment
            double threshold = lambda * (1 - absZ / (gamma * lambda));
            return Math.Sign(z) * Math.Max(0, absZ - threshold) * gamma / (gamma - 1);
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
            throw new InvalidOperationException("MCPSelector has not been fitted.");

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
        throw new NotSupportedException("MCPSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCPSelector has not been fitted.");

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
