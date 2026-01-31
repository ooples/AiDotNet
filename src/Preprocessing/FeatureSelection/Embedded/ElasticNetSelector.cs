using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Elastic Net (L1+L2) regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines L1 (Lasso) and L2 (Ridge) penalties to perform feature selection
/// while handling correlated features better than pure Lasso. The l1_ratio
/// parameter controls the mix of L1 and L2 penalties.
/// </para>
/// <para><b>For Beginners:</b> Elastic Net is like having both a strict budget (L1)
/// and a preference for simpler solutions (L2). This combination works well when
/// features are correlated - pure Lasso might arbitrarily pick one from a group
/// of correlated features, but Elastic Net tends to select or exclude them together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ElasticNetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ElasticNetSelector(
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
            "ElasticNetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = new double[n];

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

        // Elastic Net penalties
        double l1Penalty = _alpha * _l1Ratio * n;
        double l2Penalty = _alpha * (1 - _l1Ratio) * n;

        // Coordinate descent
        var beta = new double[p];
        var residuals = new double[n];
        Array.Copy(y, residuals, n);

        // Precompute X'X diagonals
        var Xj2 = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                Xj2[j] += X[i, j] * X[i, j];
        }

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                // Compute partial residual
                double rhoJ = 0;
                for (int i = 0; i < n; i++)
                    rhoJ += X[i, j] * (residuals[i] + beta[j] * X[i, j]);

                // Elastic Net update
                double betaOld = beta[j];
                double denom = Xj2[j] + l2Penalty;
                if (denom > 1e-10)
                {
                    beta[j] = SoftThreshold(rhoJ, l1Penalty) / denom;
                }

                // Update residuals
                if (Math.Abs(beta[j] - betaOld) > 1e-10)
                {
                    for (int i = 0; i < n; i++)
                        residuals[i] -= (beta[j] - betaOld) * X[i, j];
                }

                maxChange = Math.Max(maxChange, Math.Abs(beta[j] - betaOld));
            }

            if (maxChange < _tolerance)
                break;
        }

        // Un-standardize coefficients
        _coefficients = new double[p];
        for (int j = 0; j < p; j++)
            _coefficients[j] = beta[j] / stds[j];

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _coefficients
            .Select((c, idx) => (Coef: Math.Abs(c), Index: idx))
            .OrderByDescending(x => x.Coef)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private static double SoftThreshold(double x, double lambda)
    {
        if (x > lambda)
            return x - lambda;
        else if (x < -lambda)
            return x + lambda;
        else
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
