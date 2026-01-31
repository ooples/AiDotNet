using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Embedded;

/// <summary>
/// Lasso (L1) regularization-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses L1-regularized linear regression (Lasso) which drives some coefficients
/// exactly to zero, performing automatic feature selection. The regularization
/// parameter controls the sparsity of the solution.
/// </para>
/// <para><b>For Beginners:</b> Lasso is like a strict budget for your model. It
/// forces the model to "spend" wisely on features, often setting unimportant
/// feature weights to exactly zero. Features with non-zero weights after this
/// "budgeting" process are considered important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LassoSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LassoSelector(
        int nFeaturesToSelect = 10,
        double alpha = 1.0,
        int maxIterations = 1000,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha < 0)
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LassoSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Coordinate descent for Lasso
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

                // Soft thresholding
                double betaOld = beta[j];
                if (Xj2[j] > 1e-10)
                {
                    double betaNew = SoftThreshold(rhoJ, _alpha * n) / Xj2[j];
                    beta[j] = betaNew;
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

        // Select top features by coefficient magnitude
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
            throw new InvalidOperationException("LassoSelector has not been fitted.");

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
        throw new NotSupportedException("LassoSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LassoSelector has not been fitted.");

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
