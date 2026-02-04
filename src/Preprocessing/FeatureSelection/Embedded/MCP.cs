using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Minimax Concave Penalty (MCP) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// MCP is a non-convex penalty that provides an unbiased selection of significant
/// variables while controlling model complexity. Unlike LASSO, MCP does not over-penalize
/// large coefficients, providing nearly unbiased estimates for important features.
/// </para>
/// <para><b>For Beginners:</b> MCP is like LASSO but smarter about large coefficients.
/// LASSO shrinks everything, even important features. MCP says "if a coefficient is
/// big enough, stop penalizing it" - so truly important features keep their full effect
/// while unimportant ones are still shrunk to zero.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MCP<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lambda;
    private readonly double _gamma; // MCP parameter (typically > 1)
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Lambda => _lambda;
    public double Gamma => _gamma;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MCP(
        int nFeaturesToSelect = 10,
        double lambda = 0.1,
        double gamma = 3.0,
        int maxIterations = 100,
        double tolerance = 1e-6,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (lambda < 0)
            throw new ArgumentException("Lambda must be non-negative.", nameof(lambda));
        if (gamma <= 1)
            throw new ArgumentException("Gamma must be greater than 1.", nameof(gamma));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lambda = lambda;
        _gamma = gamma;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MCP requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Standardize data
        var means = new double[p];
        var stds = new double[p];
        var X = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += NumOps.ToDouble(data[i, j]);
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - means[j];
                stds[j] += diff * diff;
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (NumOps.ToDouble(data[i, j]) - means[j]) / stds[j];
        }

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]) - yMean;

        // Initialize coefficients
        _coefficients = new double[p];

        // Coordinate descent with MCP penalty
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                // Compute partial residual
                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = y[i];
                    for (int k = 0; k < p; k++)
                    {
                        if (k != j)
                            residual -= X[i, k] * _coefficients[k];
                    }
                    rho += X[i, j] * residual;
                }
                rho /= n;

                // MCP threshold
                double newCoef = MCPThreshold(rho, _lambda, _gamma);

                double change = Math.Abs(newCoef - _coefficients[j]);
                if (change > maxChange) maxChange = change;

                _coefficients[j] = newCoef;
            }

            if (maxChange < _tolerance)
                break;
        }

        // Select features with non-zero coefficients
        var nonZero = new List<(int Index, double AbsCoef)>();
        for (int j = 0; j < p; j++)
        {
            if (Math.Abs(_coefficients[j]) > 1e-10)
                nonZero.Add((j, Math.Abs(_coefficients[j])));
        }

        if (nonZero.Count == 0)
        {
            // Fallback: select top by correlation
            _selectedIndices = SelectByCorrelation(data, target, p);
        }
        else
        {
            _selectedIndices = nonZero
                .OrderByDescending(x => x.AbsCoef)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double MCPThreshold(double z, double lambda, double gamma)
    {
        double absZ = Math.Abs(z);

        if (absZ <= lambda)
        {
            // Soft threshold region
            return 0;
        }
        else if (absZ <= gamma * lambda)
        {
            // MCP region with partial shrinkage
            return Math.Sign(z) * (absZ - lambda) / (1 - 1 / gamma);
        }
        else
        {
            // No shrinkage for large values
            return z;
        }
    }

    private int[] SelectByCorrelation(Matrix<T> data, Vector<T> target, int p)
    {
        int n = data.Rows;
        var scores = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(Math.Min(_nFeaturesToSelect, p))
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCP has not been fitted.");

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
        throw new NotSupportedException("MCP does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCP has not been fitted.");

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
