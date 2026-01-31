using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Complexity;

/// <summary>
/// Fractal Dimension based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their estimated fractal dimension using
/// the Higuchi algorithm, measuring signal complexity.
/// </para>
/// <para><b>For Beginners:</b> Fractal dimension measures how "rough" or complex
/// a signal is. A straight line has dimension 1, a completely random signal
/// approaches dimension 2. Complex, irregular patterns have higher fractal
/// dimensions. This is useful for finding features with interesting structure.
/// </para>
/// </remarks>
public class FractalDimensionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _kMax;

    private double[]? _fractalDimensions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int KMax => _kMax;
    public double[]? FractalDimensions => _fractalDimensions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FractalDimensionSelector(
        int nFeaturesToSelect = 10,
        int kMax = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (kMax < 2)
            throw new ArgumentException("kMax must be at least 2.", nameof(kMax));

        _nFeaturesToSelect = nFeaturesToSelect;
        _kMax = kMax;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < _kMax + 1)
            throw new ArgumentException($"Need at least {_kMax + 1} samples for kMax = {_kMax}.");

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        _fractalDimensions = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _fractalDimensions[j] = ComputeHiguchiFD(col);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fractalDimensions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeHiguchiFD(double[] data)
    {
        int n = data.Length;
        var logK = new List<double>();
        var logL = new List<double>();

        for (int k = 1; k <= _kMax; k++)
        {
            var lmValues = new List<double>();

            for (int m = 1; m <= k; m++)
            {
                // Compute L_m(k)
                int numSegments = (n - m) / k;
                if (numSegments < 1) continue;

                double sumDiff = 0;
                for (int i = 1; i <= numSegments; i++)
                {
                    int idx1 = m + i * k - 1;
                    int idx0 = m + (i - 1) * k - 1;
                    if (idx1 < n && idx0 >= 0)
                        sumDiff += Math.Abs(data[idx1] - data[idx0]);
                }

                double normFactor = (n - 1.0) / (k * k * numSegments);
                double lm = sumDiff * normFactor;
                lmValues.Add(lm);
            }

            if (lmValues.Count > 0)
            {
                double avgL = lmValues.Average();
                if (avgL > 0)
                {
                    logK.Add(Math.Log(k));
                    logL.Add(Math.Log(avgL));
                }
            }
        }

        if (logK.Count < 2)
            return 1.5; // Default dimension

        // Linear regression: FD = -slope
        double meanX = logK.Average();
        double meanY = logL.Average();

        double numerator = 0, denominator = 0;
        for (int i = 0; i < logK.Count; i++)
        {
            numerator += (logK[i] - meanX) * (logL[i] - meanY);
            denominator += (logK[i] - meanX) * (logK[i] - meanX);
        }

        if (denominator < 1e-10)
            return 1.5;

        double fd = -numerator / denominator;

        // Clamp to valid range [1, 2]
        return Math.Max(1, Math.Min(2, fd));
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FractalDimensionSelector has not been fitted.");

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
        throw new NotSupportedException("FractalDimensionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FractalDimensionSelector has not been fitted.");

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
