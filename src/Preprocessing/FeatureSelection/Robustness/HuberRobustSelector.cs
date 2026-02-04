using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Robustness;

/// <summary>
/// Huber Robust Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their coefficients in a Huber regression, which
/// downweights the influence of outliers using the Huber loss function.
/// </para>
/// <para><b>For Beginners:</b> The Huber loss combines squared error for small
/// residuals with absolute error for large ones. This makes the regression less
/// sensitive to outliers while still being efficient for normal data.
/// </para>
/// </remarks>
public class HuberRobustSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _delta;
    private readonly int _maxIterations;
    private readonly double _learningRate;

    private double[]? _huberCoefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Delta => _delta;
    public double[]? HuberCoefficients => _huberCoefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HuberRobustSelector(
        int nFeaturesToSelect = 10,
        double delta = 1.35,
        int maxIterations = 100,
        double learningRate = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _delta = delta;
        _maxIterations = maxIterations;
        _learningRate = learningRate;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HuberRobustSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Standardize features
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;

            for (int i = 0; i < n; i++)
                stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = (X[i, j] - means[j]) / stds[j];

        // Standardize target
        double yMean = y.Average();
        double yStd = Math.Sqrt(y.Select(v => (v - yMean) * (v - yMean)).Average());
        if (yStd < 1e-10) yStd = 1;
        for (int i = 0; i < n; i++)
            y[i] = (y[i] - yMean) / yStd;

        // Fit Huber regression via gradient descent
        var weights = new double[p];
        double bias = 0;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var gradW = new double[p];
            double gradB = 0;

            for (int i = 0; i < n; i++)
            {
                double pred = bias;
                for (int j = 0; j < p; j++)
                    pred += X[i, j] * weights[j];

                double residual = pred - y[i];
                double absResidual = Math.Abs(residual);

                // Huber loss gradient
                double factor;
                if (absResidual <= _delta)
                {
                    factor = residual;
                }
                else
                {
                    factor = _delta * Math.Sign(residual);
                }

                for (int j = 0; j < p; j++)
                    gradW[j] += factor * X[i, j];
                gradB += factor;
            }

            // Update weights
            for (int j = 0; j < p; j++)
                weights[j] -= _learningRate * gradW[j] / n;
            bias -= _learningRate * gradB / n;
        }

        _huberCoefficients = weights.Select(Math.Abs).ToArray();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _huberCoefficients[j])
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
            throw new InvalidOperationException("HuberRobustSelector has not been fitted.");

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
        throw new NotSupportedException("HuberRobustSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HuberRobustSelector has not been fitted.");

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
