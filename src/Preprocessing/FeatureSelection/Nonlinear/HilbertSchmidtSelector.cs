using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonlinear;

/// <summary>
/// Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on HSIC, a kernel-based measure of independence
/// that can detect complex nonlinear relationships.
/// </para>
/// <para><b>For Beginners:</b> HSIC uses kernel methods to measure dependence
/// between variables in a high-dimensional space. It can detect subtle nonlinear
/// relationships that other methods might miss. Higher HSIC means stronger
/// dependence between the feature and target.
/// </para>
/// </remarks>
public class HilbertSchmidtSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _kernelWidth;

    private double[]? _hsicValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double KernelWidth => _kernelWidth;
    public double[]? HSICValues => _hsicValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HilbertSchmidtSelector(
        int nFeaturesToSelect = 10,
        double kernelWidth = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kernelWidth <= 0)
            throw new ArgumentException("Kernel width must be positive.", nameof(kernelWidth));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kernelWidth = kernelWidth;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HilbertSchmidtSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _hsicValues = new double[p];

        // Compute centering matrix H = I - 1/n * 11^T
        // For centered kernel: K_c = HKH

        // Kernel matrix for Y
        double yWidth = ComputeMedianWidth(y) * _kernelWidth;
        var Ky = ComputeGaussianKernel(y, yWidth);
        var KyC = CenterKernel(Ky);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double xWidth = ComputeMedianWidth(col) * _kernelWidth;
            var Kx = ComputeGaussianKernel(col, xWidth);
            var KxC = CenterKernel(Kx);

            // HSIC = (1/n²) * trace(Kx_c * Ky_c)
            double hsic = 0;
            for (int i = 0; i < n; i++)
                for (int k = 0; k < n; k++)
                    hsic += KxC[i, k] * KyC[k, i];
            hsic /= (n * n);

            _hsicValues[j] = Math.Max(0, hsic);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hsicValues[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeMedianWidth(double[] data)
    {
        int n = data.Length;
        var distances = new List<double>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                distances.Add(Math.Abs(data[i] - data[j]));

        if (distances.Count == 0) return 1.0;

        distances.Sort();
        int mid = distances.Count / 2;
        double median = distances.Count % 2 == 0
            ? (distances[mid - 1] + distances[mid]) / 2
            : distances[mid];

        return Math.Max(median, 1e-10);
    }

    private double[,] ComputeGaussianKernel(double[] data, double width)
    {
        int n = data.Length;
        var K = new double[n, n];
        double gamma = 1.0 / (2 * width * width);

        for (int i = 0; i < n; i++)
        {
            K[i, i] = 1.0;
            for (int j = i + 1; j < n; j++)
            {
                double diff = data[i] - data[j];
                double val = Math.Exp(-gamma * diff * diff);
                K[i, j] = val;
                K[j, i] = val;
            }
        }

        return K;
    }

    private double[,] CenterKernel(double[,] K)
    {
        int n = K.GetLength(0);
        var Kc = new double[n, n];

        var rowMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += K[i, j];
                grandMean += K[i, j];
            }
            rowMeans[i] /= n;
        }
        grandMean /= (n * n);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Kc[i, j] = K[i, j] - rowMeans[i] - rowMeans[j] + grandMean;

        return Kc;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HilbertSchmidtSelector has not been fitted.");

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
        throw new NotSupportedException("HilbertSchmidtSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HilbertSchmidtSelector has not been fitted.");

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
