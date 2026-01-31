using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Dimensionality;

/// <summary>
/// Truncated SVD Feature Selection using singular value decomposition.
/// </summary>
/// <remarks>
/// <para>
/// Uses Singular Value Decomposition to identify which original features
/// contribute most to the principal directions of variation. Unlike PCA,
/// this works directly on the data matrix without centering.
/// </para>
/// <para><b>For Beginners:</b> SVD breaks down your data into components
/// that explain different directions of variation. By looking at which
/// original features contribute to the most important directions, we can
/// select features that capture the most information about your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TruncatedSVDSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nComponents;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NComponents => _nComponents;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public TruncatedSVDSelection(
        int nFeaturesToSelect = 10,
        int nComponents = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nComponents < 1)
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nComponents = nComponents;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute X^T X for right singular vectors
        var xTx = new double[p, p];
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1; j2 < p; j2++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                    sum += NumOps.ToDouble(data[i, j1]) * NumOps.ToDouble(data[i, j2]);
                xTx[j1, j2] = sum;
                xTx[j2, j1] = sum;
            }
        }

        // Power iteration for top eigenvectors of X^T X
        int nComp = Math.Min(_nComponents, p);
        var loadings = new double[nComp, p];
        var singularValues = new double[nComp];
        var deflatedXtX = (double[,])xTx.Clone();

        for (int k = 0; k < nComp; k++)
        {
            var eigenvector = PowerIteration(deflatedXtX, p, 100);
            singularValues[k] = Math.Sqrt(ComputeEigenvalue(deflatedXtX, eigenvector, p));

            for (int j = 0; j < p; j++)
                loadings[k, j] = eigenvector[j];

            // Deflate
            double eigenvalue = ComputeEigenvalue(deflatedXtX, eigenvector, p);
            for (int j1 = 0; j1 < p; j1++)
                for (int j2 = 0; j2 < p; j2++)
                    deflatedXtX[j1, j2] -= eigenvalue * eigenvector[j1] * eigenvector[j2];
        }

        // Compute feature importance as weighted sum of squared loadings
        _featureImportances = new double[p];
        double totalVariance = singularValues.Sum(s => s * s);

        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < nComp; k++)
            {
                double weight = (singularValues[k] * singularValues[k]) / (totalVariance + 1e-10);
                _featureImportances[j] += weight * loadings[k, j] * loadings[k, j];
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

    private double[] PowerIteration(double[,] matrix, int size, int maxIter)
    {
        var v = new double[size];
        for (int i = 0; i < size; i++)
            v[i] = 1.0 / size;

        for (int iter = 0; iter < maxIter; iter++)
        {
            var newV = new double[size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                    newV[i] += matrix[i, j] * v[j];
            }

            double norm = Math.Sqrt(newV.Sum(x => x * x));
            if (norm > 1e-10)
            {
                for (int i = 0; i < size; i++)
                    newV[i] /= norm;
            }

            v = newV;
        }

        return v;
    }

    private double ComputeEigenvalue(double[,] matrix, double[] eigenvector, int size)
    {
        var mv = new double[size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                mv[i] += matrix[i, j] * eigenvector[j];

        double numerator = 0, denominator = 0;
        for (int i = 0; i < size; i++)
        {
            numerator += eigenvector[i] * mv[i];
            denominator += eigenvector[i] * eigenvector[i];
        }

        return denominator > 1e-10 ? numerator / denominator : 0;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TruncatedSVDSelection has not been fitted.");

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
        throw new NotSupportedException("TruncatedSVDSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TruncatedSVDSelection has not been fitted.");

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
