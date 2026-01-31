using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel PCA-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses kernel PCA to map features to a higher-dimensional space and selects
/// features that contribute most to the principal components in kernel space.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA finds linear patterns. Kernel PCA can
/// find non-linear patterns by implicitly mapping data to a higher dimension.
/// We select features that are most important in this richer representation,
/// capturing complex relationships that linear methods might miss.
/// </para>
/// </remarks>
public class KernelPCASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;
    private readonly int _nComponents;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KernelPCASelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
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

        // Compute kernel matrix (RBF kernel)
        var K = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = X[i, k] - X[j, k];
                    dist += diff * diff;
                }
                double kval = Math.Exp(-_gamma * dist);
                K[i, j] = kval;
                K[j, i] = kval;
            }
        }

        // Center kernel matrix
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double totalMean = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += K[i, j];
                colMeans[j] += K[i, j];
                totalMean += K[i, j];
            }
            rowMeans[i] /= n;
        }
        for (int j = 0; j < n; j++) colMeans[j] /= n;
        totalMean /= (n * n);

        var Kc = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Kc[i, j] = K[i, j] - rowMeans[i] - colMeans[j] + totalMean;

        // Get top eigenvectors (power iteration)
        int nComp = Math.Min(_nComponents, Math.Min(n, p));
        var eigenvectors = PowerIterationTopK(Kc, n, nComp, 50);

        // Compute feature importance by reconstructing feature sensitivity
        _featureImportances = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < nComp; k++)
            {
                // Sensitivity: how much does changing feature j affect kernel PCA projection
                double sensitivity = 0;
                for (int i = 0; i < n; i++)
                {
                    for (int i2 = 0; i2 < n; i2++)
                    {
                        double diff = X[i, j] - X[i2, j];
                        sensitivity += Math.Abs(eigenvectors[i, k] * eigenvectors[i2, k] * diff * diff);
                    }
                }
                _featureImportances[j] += sensitivity / (n * n);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] PowerIterationTopK(double[,] A, int n, int k, int maxIter)
    {
        var result = new double[n, k];
        var rand = RandomHelper.CreateSecureRandom();

        for (int ki = 0; ki < k; ki++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++) v[i] = rand.NextDouble();

            for (int iter = 0; iter < maxIter; iter++)
            {
                // Orthogonalize against previous
                for (int prev = 0; prev < ki; prev++)
                {
                    double dot = 0;
                    for (int i = 0; i < n; i++) dot += v[i] * result[i, prev];
                    for (int i = 0; i < n; i++) v[i] -= dot * result[i, prev];
                }

                var newV = new double[n];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        newV[i] += A[i, j] * v[j];

                double norm = Math.Sqrt(newV.Sum(x => x * x)) + 1e-10;
                for (int i = 0; i < n; i++) v[i] = newV[i] / norm;
            }

            for (int i = 0; i < n; i++) result[i, ki] = v[i];
        }

        return result;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelPCASelector has not been fitted.");

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
        throw new NotSupportedException("KernelPCASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelPCASelector has not been fitted.");

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
