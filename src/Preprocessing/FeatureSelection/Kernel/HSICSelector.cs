using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Hilbert-Schmidt Independence Criterion (HSIC) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses HSIC to measure the statistical dependency between features and target
/// in a reproducing kernel Hilbert space, capturing non-linear dependencies.
/// </para>
/// <para><b>For Beginners:</b> HSIC is a powerful way to measure if two variables
/// are related, even in complex non-linear ways. By computing HSIC between each
/// feature and the target, we find features that are truly informative, even when
/// the relationship isn't a straight line.
/// </para>
/// </remarks>
public class HSICSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gammaX;
    private readonly double _gammaY;

    private double[]? _hsicScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double GammaX => _gammaX;
    public double GammaY => _gammaY;
    public double[]? HSICScores => _hsicScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public HSICSelector(
        int nFeaturesToSelect = 10,
        double gammaX = 1.0,
        double gammaY = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gammaX = gammaX;
        _gammaY = gammaY;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "HSICSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Compute kernel matrix for target
        var Ky = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double diff = y[i] - y[j];
                double kval = Math.Exp(-_gammaY * diff * diff);
                Ky[i, j] = kval;
                Ky[j, i] = kval;
            }
        }

        // Center Ky
        var H = CreateCenteringMatrix(n);
        var HKy = Multiply(H, Ky, n);
        var HKyH = Multiply(HKy, H, n);

        _hsicScores = new double[p];

        for (int feat = 0; feat < p; feat++)
        {
            // Compute kernel matrix for this feature
            var Kx = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++)
                {
                    double diff = X[i, feat] - X[j, feat];
                    double kval = Math.Exp(-_gammaX * diff * diff);
                    Kx[i, j] = kval;
                    Kx[j, i] = kval;
                }
            }

            // Center Kx
            var HKx = Multiply(H, Kx, n);
            var HKxH = Multiply(HKx, H, n);

            // HSIC = trace(HKxH * HKyH) / (n-1)^2
            double hsic = 0;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    hsic += HKxH[i, j] * HKyH[j, i];
            hsic /= ((n - 1) * (n - 1));

            _hsicScores[feat] = Math.Max(0, hsic);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hsicScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[,] CreateCenteringMatrix(int n)
    {
        var H = new double[n, n];
        double val = 1.0 / n;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                H[i, j] = (i == j ? 1 : 0) - val;
        }
        return H;
    }

    private double[,] Multiply(double[,] A, double[,] B, int n)
    {
        var C = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    C[i, j] += A[i, k] * B[k, j];
        return C;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HSICSelector has not been fitted.");

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
        throw new NotSupportedException("HSICSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HSICSelector has not been fitted.");

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
