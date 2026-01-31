using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel PCA-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Applies kernel PCA to map features into a higher-dimensional space, then
/// selects features based on their contribution to the principal components
/// in the kernel space.
/// </para>
/// <para><b>For Beginners:</b> Regular PCA finds linear patterns, but some patterns
/// are curved or complex. Kernel PCA uses a mathematical trick to find these
/// non-linear patterns without actually computing in high dimensions. We then
/// select features that contribute most to these important non-linear patterns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KernelPCASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;
    private readonly double _gamma;
    private readonly string _kernelType;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KernelPCASelector(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        double gamma = 1.0,
        string kernelType = "rbf",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nComponents < 1)
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
        _gamma = gamma;
        _kernelType = kernelType.ToLowerInvariant();
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Compute kernel matrix
        var K = ComputeKernelMatrix(X, n, p);

        // Center kernel matrix
        CenterKernelMatrix(K, n);

        // Compute top eigenvectors using power iteration
        var eigenvalues = new double[_nComponents];
        var eigenvectors = new double[_nComponents, n];
        var deflatedK = (double[,])K.Clone();

        for (int k = 0; k < Math.Min(_nComponents, n); k++)
        {
            var (eigenvalue, eigenvector) = PowerIteration(deflatedK, n, 100);
            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++)
                eigenvectors[k, i] = eigenvector[i];

            // Deflate
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    deflatedK[i, j] -= eigenvalue * eigenvector[i] * eigenvector[j];
        }

        // Compute feature importance via sensitivity analysis
        _featureImportances = new double[p];
        double totalEigenvalue = eigenvalues.Sum();

        for (int j = 0; j < p; j++)
        {
            // Compute gradient of kernel w.r.t. feature j
            double importance = 0;
            for (int k = 0; k < _nComponents; k++)
            {
                if (totalEigenvalue < 1e-10) continue;
                double weight = eigenvalues[k] / totalEigenvalue;

                // Approximate importance via feature variance contribution
                double variance = 0;
                for (int i = 0; i < n; i++)
                {
                    double contribution = 0;
                    for (int i2 = 0; i2 < n; i2++)
                        contribution += eigenvectors[k, i2] * ComputeKernelGradient(X, i, i2, j, p);
                    variance += contribution * contribution;
                }
                importance += weight * variance;
            }
            _featureImportances[j] = importance;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] ComputeKernelMatrix(double[,] X, int n, int p)
    {
        var K = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double k = _kernelType switch
                {
                    "rbf" => ComputeRBFKernel(X, i, j, p),
                    "poly" => ComputePolyKernel(X, i, j, p),
                    "linear" => ComputeLinearKernel(X, i, j, p),
                    _ => ComputeRBFKernel(X, i, j, p)
                };
                K[i, j] = k;
                K[j, i] = k;
            }
        }
        return K;
    }

    private double ComputeRBFKernel(double[,] X, int i, int j, int p)
    {
        double sqDist = 0;
        for (int f = 0; f < p; f++)
        {
            double diff = X[i, f] - X[j, f];
            sqDist += diff * diff;
        }
        return Math.Exp(-_gamma * sqDist);
    }

    private double ComputePolyKernel(double[,] X, int i, int j, int p)
    {
        double dot = 0;
        for (int f = 0; f < p; f++)
            dot += X[i, f] * X[j, f];
        return Math.Pow(dot + 1, 3);
    }

    private double ComputeLinearKernel(double[,] X, int i, int j, int p)
    {
        double dot = 0;
        for (int f = 0; f < p; f++)
            dot += X[i, f] * X[j, f];
        return dot;
    }

    private double ComputeKernelGradient(double[,] X, int i, int j, int featureIdx, int p)
    {
        if (_kernelType == "rbf")
        {
            double k = ComputeRBFKernel(X, i, j, p);
            return -2 * _gamma * (X[i, featureIdx] - X[j, featureIdx]) * k;
        }
        else if (_kernelType == "poly")
        {
            double dot = 0;
            for (int f = 0; f < p; f++)
                dot += X[i, f] * X[j, f];
            return 3 * Math.Pow(dot + 1, 2) * X[j, featureIdx];
        }
        return X[j, featureIdx];
    }

    private void CenterKernelMatrix(double[,] K, int n)
    {
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
        for (int j = 0; j < n; j++)
            colMeans[j] /= n;
        totalMean /= (n * n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                K[i, j] = K[i, j] - rowMeans[i] - colMeans[j] + totalMean;
    }

    private (double eigenvalue, double[] eigenvector) PowerIteration(double[,] matrix, int size, int maxIter)
    {
        var v = new double[size];
        for (int i = 0; i < size; i++)
            v[i] = 1.0 / size;

        for (int iter = 0; iter < maxIter; iter++)
        {
            var newV = new double[size];
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    newV[i] += matrix[i, j] * v[j];

            double norm = Math.Sqrt(newV.Sum(x => x * x));
            if (norm > 1e-10)
                for (int i = 0; i < size; i++)
                    newV[i] /= norm;

            v = newV;
        }

        // Compute eigenvalue
        var mv = new double[size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                mv[i] += matrix[i, j] * v[j];

        double eigenvalue = 0;
        for (int i = 0; i < size; i++)
            eigenvalue += v[i] * mv[i];

        return (eigenvalue, v);
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
