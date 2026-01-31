using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Elastic Net-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Elastic Net combines L1 (LASSO) and L2 (Ridge) regularization to select
/// features while handling correlated features better than pure LASSO.
/// Features with non-zero coefficients are selected.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is like LASSO but smarter about groups
/// of related features. Instead of picking just one from a correlated group (like
/// LASSO does), it can keep several if they're all useful. This is great when you
/// have features that measure similar things.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ElasticNetFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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
    public int MaxIterations => _maxIterations;
    public double Tolerance => _tolerance;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ElasticNetFS(
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
        if (alpha <= 0)
            throw new ArgumentException("Alpha must be positive.", nameof(alpha));
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
            "ElasticNetFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Standardize features and target
        var xMeans = new double[p];
        var xStds = new double[p];
        var xStandardized = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                xMeans[j] += NumOps.ToDouble(data[i, j]);
            xMeans[j] /= n;

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - xMeans[j];
                xStds[j] += diff * diff;
            }
            xStds[j] = Math.Sqrt(xStds[j] / n);
            if (xStds[j] < 1e-10) xStds[j] = 1;

            for (int i = 0; i < n; i++)
                xStandardized[i, j] = (NumOps.ToDouble(data[i, j]) - xMeans[j]) / xStds[j];
        }

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        var yCentered = new double[n];
        for (int i = 0; i < n; i++)
            yCentered[i] = NumOps.ToDouble(target[i]) - yMean;

        // Coordinate descent for Elastic Net
        _coefficients = new double[p];
        var residual = new double[n];
        Array.Copy(yCentered, residual, n);

        double l1Penalty = _alpha * _l1Ratio;
        double l2Penalty = _alpha * (1 - _l1Ratio);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                double oldCoef = _coefficients[j];

                // Add back the effect of current coefficient
                for (int i = 0; i < n; i++)
                    residual[i] += oldCoef * xStandardized[i, j];

                // Compute raw update
                double rho = 0;
                for (int i = 0; i < n; i++)
                    rho += xStandardized[i, j] * residual[i];
                rho /= n;

                // Soft thresholding with L2 denominator
                double newCoef;
                if (rho > l1Penalty)
                    newCoef = (rho - l1Penalty) / (1 + l2Penalty);
                else if (rho < -l1Penalty)
                    newCoef = (rho + l1Penalty) / (1 + l2Penalty);
                else
                    newCoef = 0;

                _coefficients[j] = newCoef;

                // Update residual
                for (int i = 0; i < n; i++)
                    residual[i] -= newCoef * xStandardized[i, j];

                maxChange = Math.Max(maxChange, Math.Abs(newCoef - oldCoef));
            }

            if (maxChange < _tolerance)
                break;
        }

        // Rescale coefficients back to original scale
        for (int j = 0; j < p; j++)
            _coefficients[j] /= xStds[j];

        // Select features with largest absolute coefficients
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
            .Take(numToSelect)
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
            throw new InvalidOperationException("ElasticNetFS has not been fitted.");

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
        throw new NotSupportedException("ElasticNetFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetFS has not been fitted.");

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
