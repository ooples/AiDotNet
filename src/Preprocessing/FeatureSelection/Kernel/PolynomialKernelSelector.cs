using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Polynomial Kernel based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their importance in a polynomial kernel space,
/// capturing polynomial relationships with the target.
/// </para>
/// <para><b>For Beginners:</b> The polynomial kernel can detect relationships
/// that are polynomial in nature (like x^2, x*y, etc.). This selector finds
/// features that have strong polynomial relationships with the target.
/// </para>
/// </remarks>
public class PolynomialKernelSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _degree;
    private readonly double _coef0;

    private double[]? _kernelScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int Degree => _degree;
    public double Coef0 => _coef0;
    public double[]? KernelScores => _kernelScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public PolynomialKernelSelector(
        int nFeaturesToSelect = 10,
        int degree = 3,
        double coef0 = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _degree = degree;
        _coef0 = coef0;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PolynomialKernelSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _kernelScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Standardize feature
            double mean = col.Average();
            double std = Math.Sqrt(col.Select(v => (v - mean) * (v - mean)).Average());
            if (std > 1e-10)
                for (int i = 0; i < n; i++)
                    col[i] = (col[i] - mean) / std;

            // Compute polynomial kernel matrix for this feature
            // k(x, y) = (x*y + coef0)^degree
            var kernel = new double[n, n];
            for (int i1 = 0; i1 < n; i1++)
            {
                for (int i2 = i1; i2 < n; i2++)
                {
                    double k = Math.Pow(col[i1] * col[i2] + _coef0, _degree);
                    kernel[i1, i2] = k;
                    kernel[i2, i1] = k;
                }
            }

            // Kernel-target alignment
            double alignmentNumerator = 0;
            double alignmentDenominator = 0;

            for (int i1 = 0; i1 < n; i1++)
            {
                for (int i2 = i1 + 1; i2 < n; i2++)
                {
                    double targetSimilarity = 1 - Math.Abs(y[i1] - y[i2]) / (1 + Math.Abs(y[i1] - y[i2]));
                    alignmentNumerator += kernel[i1, i2] * targetSimilarity;
                    alignmentDenominator += kernel[i1, i2] * kernel[i1, i2];
                }
            }

            _kernelScores[j] = alignmentDenominator > 1e-10
                ? alignmentNumerator / Math.Sqrt(alignmentDenominator)
                : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _kernelScores[j])
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
            throw new InvalidOperationException("PolynomialKernelSelector has not been fitted.");

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
        throw new NotSupportedException("PolynomialKernelSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PolynomialKernelSelector has not been fitted.");

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
