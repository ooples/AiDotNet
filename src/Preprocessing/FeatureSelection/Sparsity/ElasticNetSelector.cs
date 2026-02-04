using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sparsity;

/// <summary>
/// Elastic Net based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using Elastic Net regularization, which combines
/// L1 and L2 penalties for better stability with correlated features.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net combines Lasso (L1) and Ridge (L2)
/// penalties. L1 selects features by zeroing coefficients; L2 groups correlated
/// features together. The mix gives stable feature selection even when features
/// are correlated, which pure Lasso struggles with.
/// </para>
/// </remarks>
public class ElasticNetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double L1Ratio => _l1Ratio;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ElasticNetSelector(
        int nFeaturesToSelect = 10,
        double alpha = 0.1,
        double l1Ratio = 0.5,
        int maxIterations = 1000,
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
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ElasticNetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += X[i, j];
            means[j] = sum / n;

            double sumSq = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = X[i, j] - means[j];
                sumSq += diff * diff;
            }
            stds[j] = Math.Sqrt(sumSq / n);
            if (stds[j] < 1e-10) stds[j] = 1;

            for (int i = 0; i < n; i++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        // Coordinate descent for Elastic Net
        _coefficients = new double[p];
        double l1 = _alpha * _l1Ratio;
        double l2 = _alpha * (1 - _l1Ratio);

        var xNorms = new double[p];
        for (int j = 0; j < p; j++)
        {
            double norm = 0;
            for (int i = 0; i < n; i++)
                norm += X[i, j] * X[i, j];
            xNorms[j] = norm;
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                double denom = xNorms[j] + l2 * n;
                if (denom < 1e-10) continue;

                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = y[i];
                    for (int k = 0; k < p; k++)
                        if (k != j)
                            residual -= X[i, k] * _coefficients[k];
                    rho += X[i, j] * residual;
                }

                double oldCoef = _coefficients[j];
                _coefficients[j] = SoftThreshold(rho, l1 * n) / denom;

                maxChange = Math.Max(maxChange, Math.Abs(_coefficients[j] - oldCoef));
            }

            if (maxChange < 1e-6)
                break;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_coefficients[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double SoftThreshold(double x, double lambda)
    {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetSelector has not been fitted.");

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
        throw new NotSupportedException("ElasticNetSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ElasticNetSelector has not been fitted.");

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
