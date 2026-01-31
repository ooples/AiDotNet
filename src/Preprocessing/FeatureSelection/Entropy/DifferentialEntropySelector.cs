using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Differential Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their differential (continuous) entropy,
/// estimated using nearest-neighbor methods.
/// </para>
/// <para><b>For Beginners:</b> Differential entropy measures uncertainty in
/// continuous distributions. Unlike discrete entropy, it can be negative.
/// Features with higher differential entropy contain more unpredictable
/// information, which may indicate useful variability.
/// </para>
/// </remarks>
public class DifferentialEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _kNeighbors;

    private double[]? _entropyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int KNeighbors => _kNeighbors;
    public double[]? EntropyValues => _entropyValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DifferentialEntropySelector(
        int nFeaturesToSelect = 10,
        int kNeighbors = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(kNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kNeighbors = kNeighbors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n <= _kNeighbors)
            throw new ArgumentException($"Number of samples ({n}) must be greater than k ({_kNeighbors}).");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _entropyValues = new double[p];

        // Kozachenko-Leonenko estimator for 1D differential entropy
        // H = log(n-1) - digamma(k) + log(2) + (1/n) * sum(log(2 * epsilon_i))
        double digammaK = Digamma(_kNeighbors);
        double logN = Math.Log(n - 1);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Find k-th nearest neighbor distance for each point
            double sumLogEpsilon = 0;
            for (int i = 0; i < n; i++)
            {
                var distances = new List<double>();
                for (int ii = 0; ii < n; ii++)
                {
                    if (ii != i)
                        distances.Add(Math.Abs(col[i] - col[ii]));
                }
                distances.Sort();

                double epsilon = distances[_kNeighbors - 1];
                if (epsilon < 1e-10)
                    epsilon = 1e-10;

                sumLogEpsilon += Math.Log(2 * epsilon);
            }

            _entropyValues[j] = logN - digammaK + sumLogEpsilon / n;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _entropyValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double Digamma(int k)
    {
        // Approximation of digamma function for positive integers
        // psi(k) = -gamma + sum(1/i for i in 1..k-1) where gamma ≈ 0.5772
        if (k == 1) return -0.5772156649;
        double result = -0.5772156649;
        for (int i = 1; i < k; i++)
            result += 1.0 / i;
        return result;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEntropySelector has not been fitted.");

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
        throw new NotSupportedException("DifferentialEntropySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEntropySelector has not been fitted.");

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
