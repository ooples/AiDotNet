using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Gain;

/// <summary>
/// Information Gain Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on information gain ratio, which normalizes
/// information gain by the feature's intrinsic information to avoid
/// bias toward features with many distinct values.
/// </para>
/// <para><b>For Beginners:</b> Information gain measures how much knowing a
/// feature reduces uncertainty about the target. But features with many values
/// (like IDs) can have artificially high gain. Gain ratio fixes this by dividing
/// by the feature's own entropy, giving a fair comparison across features.
/// </para>
/// </remarks>
public class InformationGainRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _gainRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? GainRatios => _gainRatios;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InformationGainRatioSelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
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
            "InformationGainRatioSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _gainRatios = new double[p];

        // Entropy of target
        var yBins = Discretize(y, _nBins);
        double hY = ComputeEntropy(yBins, n);

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var xBins = Discretize(col, _nBins);

            // Intrinsic information (entropy of X)
            double intrinsicInfo = ComputeEntropy(xBins, n);

            // Conditional entropy H(Y|X)
            double hYGivenX = ComputeConditionalEntropy(xBins, yBins, n);

            // Information gain = H(Y) - H(Y|X)
            double gain = hY - hYGivenX;

            // Gain ratio = gain / intrinsic info
            _gainRatios[j] = intrinsicInfo > 1e-10 ? gain / intrinsicInfo : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _gainRatios[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] Discretize(double[] data, int nBins)
    {
        int n = data.Length;
        var result = new int[n];

        double min = data.Min();
        double max = data.Max();
        double range = max - min;

        if (range < 1e-10)
        {
            for (int i = 0; i < n; i++)
                result[i] = 0;
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                int bin = (int)((data[i] - min) / range * (nBins - 1));
                result[i] = Math.Min(bin, nBins - 1);
            }
        }

        return result;
    }

    private double ComputeEntropy(int[] bins, int n)
    {
        var counts = new Dictionary<int, int>();
        foreach (var b in bins)
        {
            if (!counts.ContainsKey(b))
                counts[b] = 0;
            counts[b]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / n;
            if (p > 0)
                entropy -= p * Math.Log(p) / Math.Log(2);
        }

        return entropy;
    }

    private double ComputeConditionalEntropy(int[] xBins, int[] yBins, int n)
    {
        var xCounts = new Dictionary<int, int>();
        var jointCounts = new Dictionary<(int, int), int>();

        for (int i = 0; i < n; i++)
        {
            if (!xCounts.ContainsKey(xBins[i]))
                xCounts[xBins[i]] = 0;
            xCounts[xBins[i]]++;

            var key = (xBins[i], yBins[i]);
            if (!jointCounts.ContainsKey(key))
                jointCounts[key] = 0;
            jointCounts[key]++;
        }

        double condEntropy = 0;
        foreach (var xVal in xCounts.Keys)
        {
            double pX = (double)xCounts[xVal] / n;

            // Entropy of Y given X = xVal
            double hYGivenXVal = 0;
            foreach (var joint in jointCounts.Where(kv => kv.Key.Item1 == xVal))
            {
                double pYGivenX = (double)joint.Value / xCounts[xVal];
                if (pYGivenX > 0)
                    hYGivenXVal -= pYGivenX * Math.Log(pYGivenX) / Math.Log(2);
            }

            condEntropy += pX * hYGivenXVal;
        }

        return condEntropy;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGainRatioSelector has not been fitted.");

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
        throw new NotSupportedException("InformationGainRatioSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGainRatioSelector has not been fitted.");

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
