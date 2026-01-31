using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Gain Ratio for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Gain Ratio normalizes Information Gain by the intrinsic value (entropy) of the feature
/// itself. This prevents bias toward features with many values and provides a more balanced
/// measure of feature relevance.
/// </para>
/// <para><b>For Beginners:</b> Information Gain can be unfairly high for features with many
/// unique values (like IDs). Gain Ratio fixes this by dividing by how spread out the
/// feature itself is. A feature with 100 values needs more "justification" to score high
/// than one with just 2 values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GainRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _gainRatios;
    private double[]? _informationGains;
    private double[]? _intrinsicValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? GainRatios => _gainRatios;
    public double[]? InformationGains => _informationGains;
    public double[]? IntrinsicValues => _intrinsicValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GainRatio(
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
            "GainRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute class probabilities and base entropy H(Y)
        var classCount = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int c = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCount.ContainsKey(c))
                classCount[c] = 0;
            classCount[c]++;
        }

        double baseEntropy = 0;
        foreach (var count in classCount.Values)
        {
            double p_c = (double)count / n;
            if (p_c > 0)
                baseEntropy -= p_c * Math.Log(p_c);
        }

        _informationGains = new double[p];
        _intrinsicValues = new double[p];
        _gainRatios = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get feature values and discretize into bins
            var values = new double[n];
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, values[i]);
                max = Math.Max(max, values[i]);
            }

            double binWidth = (max - min) / _nBins;
            if (binWidth < 1e-10) binWidth = 1;

            // Count joint frequencies
            var binClassCount = new Dictionary<int, Dictionary<int, int>>();
            var binCount = new Dictionary<int, int>();

            for (int i = 0; i < n; i++)
            {
                int bin = Math.Min((int)((values[i] - min) / binWidth), _nBins - 1);
                int c = (int)Math.Round(NumOps.ToDouble(target[i]));

                if (!binClassCount.ContainsKey(bin))
                    binClassCount[bin] = new Dictionary<int, int>();
                if (!binClassCount[bin].ContainsKey(c))
                    binClassCount[bin][c] = 0;
                binClassCount[bin][c]++;

                if (!binCount.ContainsKey(bin))
                    binCount[bin] = 0;
                binCount[bin]++;
            }

            // Compute conditional entropy H(Y|X)
            double conditionalEntropy = 0;
            foreach (var bin in binClassCount.Keys)
            {
                double binProb = (double)binCount[bin] / n;
                double binEntropy = 0;

                foreach (var count in binClassCount[bin].Values)
                {
                    double p_c_bin = (double)count / binCount[bin];
                    if (p_c_bin > 0)
                        binEntropy -= p_c_bin * Math.Log(p_c_bin);
                }

                conditionalEntropy += binProb * binEntropy;
            }

            // Information Gain = H(Y) - H(Y|X)
            _informationGains[j] = baseEntropy - conditionalEntropy;

            // Intrinsic Value (Split Information) = H(X)
            double intrinsicValue = 0;
            foreach (var count in binCount.Values)
            {
                double p_bin = (double)count / n;
                if (p_bin > 0)
                    intrinsicValue -= p_bin * Math.Log(p_bin);
            }
            _intrinsicValues[j] = intrinsicValue;

            // Gain Ratio = Information Gain / Intrinsic Value
            _gainRatios[j] = intrinsicValue > 1e-10 ? _informationGains[j] / intrinsicValue : 0;
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _gainRatios
            .Select((gr, idx) => (GR: gr, Index: idx))
            .OrderByDescending(x => x.GR)
            .Take(numToSelect)
            .Select(x => x.Index)
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
            throw new InvalidOperationException("GainRatio has not been fitted.");

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
        throw new NotSupportedException("GainRatio does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GainRatio has not been fitted.");

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
