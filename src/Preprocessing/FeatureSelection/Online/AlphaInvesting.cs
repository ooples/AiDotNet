using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Online;

/// <summary>
/// Alpha-Investing for online feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// Alpha-Investing is an online algorithm for controlling the false discovery rate
/// when testing features sequentially. It maintains a "wealth" that increases when
/// null hypotheses are rejected and decreases when tests are performed.
/// </para>
/// <para><b>For Beginners:</b> Think of it like managing a budget for discoveries.
/// You start with some "discovery wealth." Each time you test a feature, you spend
/// some wealth. When you find a significant feature, you earn wealth back. This
/// ensures you don't make too many false discoveries while still finding real ones.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AlphaInvesting<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _initialWealth;
    private readonly double _omega; // Wealth earned when rejecting null
    private readonly double _eta; // Fraction of wealth to invest per test

    private double[]? _pValues;
    private double[]? _testAlphas;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InitialWealth => _initialWealth;
    public double Omega => _omega;
    public double Eta => _eta;
    public double[]? PValues => _pValues;
    public double[]? TestAlphas => _testAlphas;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AlphaInvesting(
        int nFeaturesToSelect = 10,
        double initialWealth = 0.5,
        double omega = 0.05,
        double eta = 0.25,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (initialWealth <= 0)
            throw new ArgumentException("Initial wealth must be positive.", nameof(initialWealth));
        if (omega <= 0 || omega > 1)
            throw new ArgumentException("Omega must be between 0 and 1.", nameof(omega));
        if (eta <= 0 || eta > 1)
            throw new ArgumentException("Eta must be between 0 and 1.", nameof(eta));

        _nFeaturesToSelect = nFeaturesToSelect;
        _initialWealth = initialWealth;
        _omega = omega;
        _eta = eta;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AlphaInvesting requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _pValues = new double[p];
        _testAlphas = new double[p];

        var selected = new List<int>();
        double wealth = _initialWealth;

        // Compute p-values for all features
        for (int j = 0; j < p; j++)
            _pValues[j] = ComputePValue(data, target, j, n);

        // Order features by p-value for more efficient testing
        var orderedFeatures = _pValues
            .Select((pval, idx) => (PValue: pval, Index: idx))
            .OrderBy(x => x.PValue)
            .ToList();

        // Alpha-investing procedure
        for (int t = 0; t < orderedFeatures.Count; t++)
        {
            if (selected.Count >= _nFeaturesToSelect)
                break;

            if (wealth <= 0)
                break;

            var (pValue, featureIdx) = orderedFeatures[t];

            // Compute alpha for this test
            double alpha = _eta * wealth / (1 + _eta * wealth);
            _testAlphas[featureIdx] = alpha;

            // Invest
            wealth -= alpha / (1 - alpha);

            // Test
            if (pValue <= alpha)
            {
                // Reject null hypothesis - feature is significant
                selected.Add(featureIdx);

                // Gain wealth from discovery
                wealth += _omega;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // If not enough selected, add top features by p-value
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var additional = orderedFeatures
                .Where(x => !selected.Contains(x.Index))
                .Take(_nFeaturesToSelect - _selectedIndices.Length)
                .Select(x => x.Index);

            _selectedIndices = selected.Concat(additional).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private double ComputePValue(Matrix<T> data, Vector<T> target, int featureIdx, int n)
    {
        // Compute correlation and convert to t-statistic
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, featureIdx]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, featureIdx]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        double r = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;

        // Convert to t-statistic
        double t = r * Math.Sqrt((n - 2) / (1 - r * r + 1e-10));

        // Two-tailed p-value using normal approximation
        return 2 * (1 - NormalCDF(Math.Abs(t)));
    }

    private double NormalCDF(double x)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AlphaInvesting has not been fitted.");

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
        throw new NotSupportedException("AlphaInvesting does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AlphaInvesting has not been fitted.");

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
