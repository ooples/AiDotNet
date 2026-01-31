using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gain Ratio Feature Selection (normalized Information Gain).
/// </summary>
/// <remarks>
/// <para>
/// Gain Ratio normalizes information gain by the intrinsic information of the
/// feature (split information). This corrects for the bias of information gain
/// towards features with many values.
/// </para>
/// <para><b>For Beginners:</b> Information gain tends to favor features with
/// many unique values (like IDs), even if they're not truly useful. Gain ratio
/// fixes this by dividing the information gain by how "spread out" the feature
/// values are. This gives fairer scores across features with different numbers
/// of unique values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GainRatioSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _gainRatioScores;
    private double[]? _informationGain;
    private double[]? _splitInfo;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? GainRatioScores => _gainRatioScores;
    public double[]? InformationGain => _informationGain;
    public double[]? SplitInfo => _splitInfo;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GainRatioSelection(
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
            "GainRatioSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _gainRatioScores = new double[p];
        _informationGain = new double[p];
        _splitInfo = new double[p];

        // Compute target entropy
        var targetCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            targetCounts[label] = targetCounts.GetValueOrDefault(label) + 1;
        }

        double targetEntropy = 0;
        foreach (int count in targetCounts.Values)
        {
            double prob = (double)count / n;
            targetEntropy -= prob * Math.Log(prob);
        }

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            var values = new double[n];
            double minVal = double.MaxValue;
            double maxVal = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, values[i]);
                maxVal = Math.Max(maxVal, values[i]);
            }

            var featureBins = new int[n];
            double range = maxVal - minVal;

            for (int i = 0; i < n; i++)
            {
                featureBins[i] = range > 1e-10
                    ? Math.Min((int)((values[i] - minVal) / range * (_nBins - 1)), _nBins - 1)
                    : 0;
            }

            // Compute conditional entropy H(Y|X)
            var binCounts = new int[_nBins];
            var binTargetCounts = new Dictionary<int, Dictionary<int, int>>();

            for (int i = 0; i < n; i++)
            {
                int bin = featureBins[i];
                int label = (int)Math.Round(NumOps.ToDouble(target[i]));

                binCounts[bin]++;
                if (!binTargetCounts.ContainsKey(bin))
                    binTargetCounts[bin] = new Dictionary<int, int>();
                binTargetCounts[bin][label] = binTargetCounts[bin].GetValueOrDefault(label) + 1;
            }

            double conditionalEntropy = 0;
            double splitInfo = 0;

            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] == 0) continue;

                double binProb = (double)binCounts[b] / n;
                splitInfo -= binProb * Math.Log(binProb);

                if (binTargetCounts.ContainsKey(b))
                {
                    double binEntropy = 0;
                    foreach (int count in binTargetCounts[b].Values)
                    {
                        double prob = (double)count / binCounts[b];
                        binEntropy -= prob * Math.Log(prob);
                    }
                    conditionalEntropy += binProb * binEntropy;
                }
            }

            _informationGain[j] = targetEntropy - conditionalEntropy;
            _splitInfo[j] = splitInfo;
            _gainRatioScores[j] = splitInfo > 1e-10 ? _informationGain[j] / splitInfo : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _gainRatioScores[j])
            .Take(numToSelect)
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
            throw new InvalidOperationException("GainRatioSelection has not been fitted.");

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
        throw new NotSupportedException("GainRatioSelection does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GainRatioSelection has not been fitted.");

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
