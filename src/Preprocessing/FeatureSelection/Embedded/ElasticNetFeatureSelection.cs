using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Elastic Net regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization. Like Lasso, it can
/// drive coefficients to zero for feature selection, but the L2 component helps when
/// features are correlated. The l1_ratio parameter controls the mix between L1 and L2.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is a compromise between Lasso and Ridge
/// regression. When features are highly correlated, Lasso tends to pick one and ignore
/// the others. Elastic Net groups correlated features together, selecting them as a
/// group rather than arbitrarily picking one.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ElasticNetFeatureSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ElasticNetFeatureSelection(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        double l1Ratio = 0.5,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        if (l1Ratio < 0 || l1Ratio > 1)
            throw new ArgumentException("L1 ratio must be between 0 and 1.", nameof(l1Ratio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ElasticNetFeatureSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Coordinate descent for Elastic Net
        _coefficients = new double[p];
        var residuals = (double[])y.Clone();

        double l1Penalty = _alpha * _l1Ratio * n;
        double l2Penalty = _alpha * (1 - _l1Ratio) * n;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                double oldCoef = _coefficients[j];

                // Add back contribution of current feature to residuals
                for (int i = 0; i < n; i++)
                    residuals[i] += X[i, j] * oldCoef;

                // Compute raw update
                double rho = 0;
                double sumXjSq = 0;
                for (int i = 0; i < n; i++)
                {
                    rho += X[i, j] * residuals[i];
                    sumXjSq += X[i, j] * X[i, j];
                }

                // Elastic Net update with soft thresholding
                double denom = sumXjSq + l2Penalty;
                double newCoef;
                if (denom < 1e-10)
                {
                    newCoef = 0;
                }
                else
                {
                    if (rho > l1Penalty)
                        newCoef = (rho - l1Penalty) / denom;
                    else if (rho < -l1Penalty)
                        newCoef = (rho + l1Penalty) / denom;
                    else
                        newCoef = 0;
                }

                _coefficients[j] = newCoef;

                // Update residuals
                for (int i = 0; i < n; i++)
                    residuals[i] -= X[i, j] * newCoef;

                maxChange = Math.Max(maxChange, Math.Abs(newCoef - oldCoef));
            }

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
            throw new InvalidOperationException("ElasticNetFeatureSelection has not been fitted.");

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
        throw new NotSupportedException("ElasticNetFeatureSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetFeatureSelection has not been fitted.");

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
