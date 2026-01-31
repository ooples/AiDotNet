using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wavelet;

/// <summary>
/// Wavelet Coefficient based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on the magnitude and correlation of their wavelet
/// coefficients with the target variable across multiple decomposition levels.
/// </para>
/// <para><b>For Beginners:</b> This decomposes features into different frequency
/// bands and checks which features have strong relationships with the target
/// at various scales, helping identify features with multi-scale predictive power.
/// </para>
/// </remarks>
public class WaveletCoefficientSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _decompositionLevel;

    private double[]? _waveletScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int DecompositionLevel => _decompositionLevel;
    public double[]? WaveletScores => _waveletScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WaveletCoefficientSelector(
        int nFeaturesToSelect = 10,
        int decompositionLevel = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decompositionLevel = decompositionLevel;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WaveletCoefficientSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Decompose target
        var targetCoeffs = DecomposeWavelet(y, n);

        _waveletScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var featureCoeffs = DecomposeWavelet(col, n);

            // Compute correlation at each level
            double totalScore = 0;
            int minLevels = Math.Min(featureCoeffs.Count, targetCoeffs.Count);
            for (int level = 0; level < minLevels; level++)
            {
                double corr = ComputeCorrelation(featureCoeffs[level], targetCoeffs[level]);
                totalScore += Math.Abs(corr);
            }

            _waveletScores[j] = minLevels > 0 ? totalScore / minLevels : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _waveletScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<double[]> DecomposeWavelet(double[] signal, int n)
    {
        var coefficients = new List<double[]>();
        var current = (double[])signal.Clone();

        for (int level = 0; level < _decompositionLevel && current.Length >= 2; level++)
        {
            int len = current.Length;
            int halfLen = len / 2;
            var approx = new double[halfLen];
            var detail = new double[halfLen];

            for (int i = 0; i < halfLen; i++)
            {
                approx[i] = (current[2 * i] + current[2 * i + 1]) / Math.Sqrt(2);
                detail[i] = (current[2 * i] - current[2 * i + 1]) / Math.Sqrt(2);
            }

            coefficients.Add(detail);
            current = approx;
        }

        coefficients.Add(current);
        return coefficients;
    }

    private double ComputeCorrelation(double[] x, double[] y)
    {
        int n = Math.Min(x.Length, y.Length);
        if (n < 2) return 0;

        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += x[i];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = x[i] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletCoefficientSelector has not been fitted.");

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
        throw new NotSupportedException("WaveletCoefficientSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletCoefficientSelector has not been fitted.");

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
