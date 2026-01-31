using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gain Ratio for feature selection, normalizing Information Gain by feature entropy.
/// </summary>
/// <remarks>
/// <para>
/// Gain Ratio addresses Information Gain's bias toward features with many values
/// by dividing by the feature's intrinsic information (entropy of the feature itself).
/// </para>
/// <para><b>For Beginners:</b> Information Gain tends to favor features with many
/// unique values (like IDs). Gain Ratio fixes this by accounting for how complex
/// the feature is. A feature that tells you a lot AND is simple gets a higher score
/// than one that's complex but only slightly informative.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GainRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _gainRatios;
    private double[]? _informationGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GainRatios => _gainRatios;
    public double[]? InformationGains => _informationGains;
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

        // Get class distribution
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCounts.ContainsKey(label))
                classCounts[label] = 0;
            classCounts[label]++;
        }

        int nClasses = classCounts.Count;
        var classes = classCounts.Keys.OrderBy(x => x).ToArray();
        var classIndex = classes.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

        // Target entropy
        double targetEntropy = 0;
        foreach (var count in classCounts.Values)
        {
            double prob = (double)count / n;
            if (prob > 0)
                targetEntropy -= prob * Math.Log(prob);
        }

        _informationGains = new double[p];
        _gainRatios = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Bin the feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            var binCounts = new int[_nBins];
            var binClassCounts = new int[_nBins, nClasses];

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                int bin = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
                int classIdx = classIndex[(int)Math.Round(NumOps.ToDouble(target[i]))];

                binCounts[bin]++;
                binClassCounts[bin, classIdx]++;
            }

            // Conditional entropy H(Y|X)
            double conditionalEntropy = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] == 0) continue;

                double binProb = (double)binCounts[b] / n;
                double binEntropy = 0;

                for (int c = 0; c < nClasses; c++)
                {
                    if (binClassCounts[b, c] > 0)
                    {
                        double prob = (double)binClassCounts[b, c] / binCounts[b];
                        binEntropy -= prob * Math.Log(prob);
                    }
                }

                conditionalEntropy += binProb * binEntropy;
            }

            // Information gain
            double ig = Math.Max(0, targetEntropy - conditionalEntropy);
            _informationGains[j] = ig;

            // Intrinsic information (feature entropy)
            double intrinsicInfo = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] > 0)
                {
                    double prob = (double)binCounts[b] / n;
                    intrinsicInfo -= prob * Math.Log(prob);
                }
            }

            // Gain ratio = IG / IV
            _gainRatios[j] = intrinsicInfo > 1e-10 ? ig / intrinsicInfo : 0;
        }

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _gainRatios
            .Select((gr, idx) => (GR: gr, Index: idx))
            .OrderByDescending(x => x.GR)
            .Take(nToSelect)
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
