using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Information Gain and Gain Ratio feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Information Gain measures how much knowing a feature reduces uncertainty about the target.
/// Gain Ratio normalizes by the intrinsic value to handle features with many values.
/// </para>
/// <para>
/// Information Gain = H(Y) - H(Y|X)
/// Gain Ratio = Information Gain / Intrinsic Value
/// </para>
/// <para><b>For Beginners:</b> Information Gain measures how much a feature "tells you"
/// about the target variable. If knowing the feature value greatly reduces your
/// uncertainty about the outcome, it has high information gain.
///
/// Gain Ratio corrects for features with many categories (like IDs) that would
/// otherwise appear artificially important.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InformationGain<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _useGainRatio;

    private double[]? _scores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Scores => _scores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InformationGain(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool useGainRatio = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _useGainRatio = useGainRatio;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InformationGain requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var y = DiscretizeTarget(target, n);
        double hY = ComputeEntropy(y, n);

        _scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var x = DiscretizeFeature(data, j, n);
            double hYGivenX = ComputeConditionalEntropy(y, x, n);
            double ig = hY - hYGivenX;

            if (_useGainRatio)
            {
                double intrinsicValue = ComputeEntropy(x, n);
                _scores[j] = intrinsicValue > 1e-10 ? ig / intrinsicValue : 0;
            }
            else
            {
                _scores[j] = ig;
            }
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _scores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(target[i]);

        var unique = values.Distinct().OrderBy(x => x).ToList();
        var result = new int[n];

        for (int i = 0; i < n; i++)
            result[i] = unique.IndexOf(values[i]);

        return result;
    }

    private int[] DiscretizeFeature(Matrix<T> data, int j, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(data[i, j]);

        double min = values.Min();
        double max = values.Max();
        double range = max - min;
        if (range < 1e-10) range = 1;

        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            int bin = (int)((values[i] - min) / range * (_nBins - 1));
            result[i] = Math.Max(0, Math.Min(bin, _nBins - 1));
        }

        return result;
    }

    private double ComputeEntropy(int[] x, int n)
    {
        var counts = new Dictionary<int, int>();
        foreach (int v in x)
        {
            if (!counts.ContainsKey(v)) counts[v] = 0;
            counts[v]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / n;
            if (p > 0) entropy -= p * Math.Log(p);
        }

        return entropy;
    }

    private double ComputeConditionalEntropy(int[] y, int[] x, int n)
    {
        var xCounts = new Dictionary<int, int>();
        var jointCounts = new Dictionary<(int, int), int>();

        for (int i = 0; i < n; i++)
        {
            if (!xCounts.ContainsKey(x[i])) xCounts[x[i]] = 0;
            xCounts[x[i]]++;

            var key = (x[i], y[i]);
            if (!jointCounts.ContainsKey(key)) jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double hYGivenX = 0;
        foreach (var xVal in xCounts.Keys)
        {
            double pX = (double)xCounts[xVal] / n;
            double hYGivenXVal = 0;

            foreach (var joint in jointCounts.Where(kv => kv.Key.Item1 == xVal))
            {
                double pYGivenX = (double)joint.Value / xCounts[xVal];
                if (pYGivenX > 0) hYGivenXVal -= pYGivenX * Math.Log(pYGivenX);
            }

            hYGivenX += pX * hYGivenXVal;
        }

        return hYGivenX;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGain has not been fitted.");

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
        throw new NotSupportedException("InformationGain does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGain has not been fitted.");

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
