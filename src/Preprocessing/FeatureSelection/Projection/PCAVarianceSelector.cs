using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Projection;

/// <summary>
/// PCA Variance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to principal components,
/// measuring how much variance each feature contributes to the data.
/// </para>
/// <para><b>For Beginners:</b> PCA finds the main directions of variation in data.
/// This selector identifies which original features contribute most to those
/// main directions, keeping features that explain the most variance.
/// </para>
/// </remarks>
public class PCAVarianceSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private double[]? _varianceContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public double[]? VarianceContributions => _varianceContributions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PCAVarianceSelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Center the data
        var means = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                means[j] += X[i, j];
            means[j] /= n;
        }

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] -= means[j];

        // Compute covariance matrix
        var covariance = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += X[i, j1] * X[i, j2];
                covariance[j1, j2] = sum / (n - 1);
                covariance[j2, j1] = covariance[j1, j2];
            }
        }

        // Power iteration to find top components
        int numComponents = Math.Min(_nComponents, p);
        var loadings = new double[numComponents, p];
        var eigenvalues = new double[numComponents];

        var covCopy = (double[,])covariance.Clone();

        for (int c = 0; c < numComponents; c++)
        {
            var eigenvector = PowerIteration(covCopy, p, out double eigenvalue);
            eigenvalues[c] = eigenvalue;
            for (int j = 0; j < p; j++)
                loadings[c, j] = eigenvector[j];

            // Deflate covariance matrix
            for (int j1 = 0; j1 < p; j1++)
                for (int j2 = 0; j2 < p; j2++)
                    covCopy[j1, j2] -= eigenvalue * eigenvector[j1] * eigenvector[j2];
        }

        // Compute contribution of each feature (sum of squared loadings weighted by variance explained)
        double totalVariance = eigenvalues.Sum();
        _varianceContributions = new double[p];

        for (int j = 0; j < p; j++)
        {
            for (int c = 0; c < numComponents; c++)
            {
                double weight = totalVariance > 0 ? eigenvalues[c] / totalVariance : 1.0 / numComponents;
                _varianceContributions[j] += weight * loadings[c, j] * loadings[c, j];
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _varianceContributions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] PowerIteration(double[,] matrix, int size, out double eigenvalue)
    {
        var vector = new double[size];
        for (int i = 0; i < size; i++)
            vector[i] = 1.0 / Math.Sqrt(size);

        for (int iter = 0; iter < 100; iter++)
        {
            var newVector = new double[size];
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    newVector[i] += matrix[i, j] * vector[j];

            double norm = Math.Sqrt(newVector.Sum(x => x * x));
            if (norm > 1e-10)
                for (int i = 0; i < size; i++)
                    newVector[i] /= norm;

            double change = 0;
            for (int i = 0; i < size; i++)
                change += (newVector[i] - vector[i]) * (newVector[i] - vector[i]);

            vector = newVector;
            if (change < 1e-10) break;
        }

        // Compute eigenvalue
        eigenvalue = 0;
        for (int i = 0; i < size; i++)
        {
            double mv = 0;
            for (int j = 0; j < size; j++)
                mv += matrix[i, j] * vector[j];
            eigenvalue += mv * vector[i];
        }

        return vector;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCAVarianceSelector has not been fitted.");

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
        throw new NotSupportedException("PCAVarianceSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PCAVarianceSelector has not been fitted.");

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
