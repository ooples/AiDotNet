using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Fisher's Exact Test for categorical feature selection with small samples.
/// </summary>
/// <remarks>
/// <para>
/// Fisher's Exact Test computes the exact probability of observing the contingency
/// table assuming independence. Unlike chi-square, it's accurate for small samples.
/// </para>
/// <para><b>For Beginners:</b> When you have categorical features and binary targets,
/// this test determines if there's a significant relationship. Unlike chi-square,
/// it works well even with small sample sizes or sparse cells.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FisherExactTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _pValues;
    private double[]? _oddsRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? PValues => _pValues;
    public double[]? OddsRatios => _oddsRatios;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FisherExactTest(int nFeaturesToSelect = 10, int nBins = 2, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FisherExactTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Binary target (find unique classes)
        var classes = new HashSet<double>();
        for (int i = 0; i < n; i++)
            classes.Add(NumOps.ToDouble(target[i]));

        if (classes.Count != 2)
            throw new ArgumentException("FisherExactTest requires exactly 2 classes for binary target.");

        var classList = classes.OrderBy(x => x).ToList();
        double class0 = classList[0];

        _pValues = new double[p];
        _oddsRatios = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature into bins
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            double min = values.Min();
            double max = values.Max();
            double range = max - min;
            if (range < 1e-10) range = 1;

            // Create 2x2 contingency table (binary discretization)
            // For simplicity, split at median for 2 bins
            double median = values.OrderBy(x => x).Skip(n / 2).First();

            int a = 0, b = 0, c = 0, d = 0;
            for (int i = 0; i < n; i++)
            {
                bool lowFeature = values[i] <= median;
                bool class0Target = Math.Abs(NumOps.ToDouble(target[i]) - class0) < 1e-10;

                if (lowFeature && class0Target) a++;
                else if (lowFeature && !class0Target) b++;
                else if (!lowFeature && class0Target) c++;
                else d++;
            }

            // Fisher's exact test p-value for 2x2 table
            _pValues[j] = FisherExactP(a, b, c, d);

            // Odds ratio
            _oddsRatios[j] = (b * c) > 0 ? (double)(a * d) / (b * c) : double.PositiveInfinity;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double FisherExactP(int a, int b, int c, int d)
    {
        int n = a + b + c + d;
        int r1 = a + b;
        int r2 = c + d;
        int c1 = a + c;
        int c2 = b + d;

        // Hypergeometric probability
        double pObs = HypergeometricPMF(a, r1, c1, n);

        // Sum probabilities of all tables as or more extreme
        double pValue = 0;
        int minA = Math.Max(0, r1 - c2);
        int maxA = Math.Min(r1, c1);

        for (int aNew = minA; aNew <= maxA; aNew++)
        {
            double pNew = HypergeometricPMF(aNew, r1, c1, n);
            if (pNew <= pObs + 1e-10)
                pValue += pNew;
        }

        return Math.Min(pValue, 1.0);
    }

    private double HypergeometricPMF(int k, int n1, int K, int N)
    {
        // P(X=k) = C(K,k) * C(N-K, n1-k) / C(N, n1)
        return Math.Exp(LogBinomial(K, k) + LogBinomial(N - K, n1 - k) - LogBinomial(N, n1));
    }

    private double LogBinomial(int n, int k)
    {
        if (k < 0 || k > n) return double.NegativeInfinity;
        if (k == 0 || k == n) return 0;

        return LogFactorial(n) - LogFactorial(k) - LogFactorial(n - k);
    }

    private double LogFactorial(int n)
    {
        if (n <= 1) return 0;

        // Stirling's approximation for large n
        if (n > 20)
            return n * Math.Log(n) - n + 0.5 * Math.Log(2 * Math.PI * n);

        double result = 0;
        for (int i = 2; i <= n; i++)
            result += Math.Log(i);
        return result;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherExactTest has not been fitted.");

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
        throw new NotSupportedException("FisherExactTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherExactTest has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
