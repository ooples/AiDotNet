using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Automatic Relevance Determination (ARD) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Bayesian ARD to automatically determine which features are relevant
/// by learning individual precision (inverse variance) parameters for each
/// feature's coefficient. Features with high precision (low variance) are
/// effectively pruned.
/// </para>
/// <para><b>For Beginners:</b> ARD is like giving each feature its own "importance dial."
/// During training, the method automatically turns down the dial for irrelevant features
/// (making their effect nearly zero) while keeping the dial up for important ones.
/// Features whose dials get turned all the way down are removed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ARDSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alphaThreshold;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    private double[]? _alphas;
    private double[]? _coefficients;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Alphas => _alphas;
    public double[]? Coefficients => _coefficients;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ARDSelector(
        int nFeaturesToSelect = 10,
        double alphaThreshold = 1e8,
        int maxIterations = 300,
        double tolerance = 1e-4,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alphaThreshold <= 0)
            throw new ArgumentOutOfRangeException(nameof(alphaThreshold), "Alpha threshold must be positive.");
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "Max iterations must be at least 1.");
        if (tolerance <= 0)
            throw new ArgumentOutOfRangeException(nameof(tolerance), "Tolerance must be positive.");

        _nFeaturesToSelect = nFeaturesToSelect;
        _alphaThreshold = alphaThreshold;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ARDSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");
        if (data.Rows == 0)
            throw new ArgumentException("Data must have at least one row.", nameof(data));
        if (data.Columns == 0)
            throw new ArgumentException("Data must have at least one column.", nameof(data));
        if (_nFeaturesToSelect > data.Columns)
            throw new ArgumentException(
                $"Number of features to select ({_nFeaturesToSelect}) cannot exceed number of columns ({data.Columns}).",
                nameof(data));

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize alphas (precision parameters)
        _alphas = new double[p];
        _coefficients = new double[p];
        for (int j = 0; j < p; j++)
            _alphas[j] = 1.0;

        double beta = 1.0; // Noise precision

        // Precompute X^T X and X^T y
        var XtX = new double[p, p];
        var Xty = new double[p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int i = 0; i < n; i++)
                Xty[j1] += X[i, j1] * y[i];

            for (int j2 = j1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += X[i, j1] * X[i, j2];
                XtX[j1, j2] = sum;
                XtX[j2, j1] = sum;
            }
        }

        // EM iterations
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var oldAlphas = (double[])_alphas.Clone();

            // Compute posterior covariance (simplified diagonal approximation)
            var sigma = new double[p];
            for (int j = 0; j < p; j++)
            {
                double denom = _alphas[j] + beta * XtX[j, j];
                sigma[j] = denom > 1e-10 ? 1.0 / denom : 1e10;
            }

            // Compute posterior mean (coefficients)
            for (int j = 0; j < p; j++)
                _coefficients[j] = beta * sigma[j] * Xty[j];

            // Compute gamma (effective number of parameters per feature)
            var gamma = new double[p];
            for (int j = 0; j < p; j++)
                gamma[j] = 1 - _alphas[j] * sigma[j];

            // Update alphas
            for (int j = 0; j < p; j++)
            {
                double w2 = _coefficients[j] * _coefficients[j];
                if (w2 > 1e-10 && gamma[j] > 1e-10)
                    _alphas[j] = gamma[j] / w2;
                else
                    _alphas[j] = _alphaThreshold;
            }

            // Update beta (noise precision)
            double residualSum = 0;
            for (int i = 0; i < n; i++)
            {
                double pred = 0;
                for (int j = 0; j < p; j++)
                    pred += X[i, j] * _coefficients[j];
                residualSum += (y[i] - pred) * (y[i] - pred);
            }
            double gammaSum = gamma.Sum();
            beta = Math.Max((n - gammaSum) / (residualSum + 1e-10), 1e-10);

            // Check convergence
            double maxChange = 0;
            for (int j = 0; j < p; j++)
            {
                double change = Math.Abs(_alphas[j] - oldAlphas[j]) / (Math.Abs(oldAlphas[j]) + 1e-10);
                maxChange = Math.Max(maxChange, change);
            }
            if (maxChange < _tolerance)
                break;
        }

        // Select features with low alpha (high relevance)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => _alphas[j] < _alphaThreshold)
            .OrderBy(j => _alphas[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If not enough features below threshold, take top by coefficient magnitude
        if (_selectedIndices.Length < numToSelect)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => Math.Abs(_coefficients[j]))
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("ARDSelector has not been fitted.");
        if (data.Columns != _nInputFeatures)
            throw new ArgumentException(
                $"Expected {_nInputFeatures} columns but got {data.Columns}.", nameof(data));

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
        throw new NotSupportedException("ARDSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ARDSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ARDSelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        if (inputFeatureNames.Length < _nInputFeatures)
            throw new ArgumentException(
                $"Expected at least {_nInputFeatures} feature names, but got {inputFeatureNames.Length}.",
                nameof(inputFeatureNames));

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
