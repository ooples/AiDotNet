using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Information Gain (Mutual Information) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Information Gain measures the reduction in entropy of the target variable when
/// a feature is known. It's equivalent to mutual information and quantifies how much
/// information about the target a feature provides.
/// </para>
/// <para><b>For Beginners:</b> Information Gain asks: "How much does knowing this
/// feature reduce my uncertainty about the target?" If knowing the feature value
/// tells you a lot about what the target will be, it has high information gain.
/// It's measured in bits (or nats, depending on the logarithm base).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class InformationGain<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _informationGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? InformationGains => _informationGains;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InformationGain(
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
            "InformationGain requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute target entropy
        var targetCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!targetCounts.ContainsKey(label))
                targetCounts[label] = 0;
            targetCounts[label]++;
        }

        double targetEntropy = 0;
        foreach (int count in targetCounts.Values)
        {
            double p_y = (double)count / n;
            if (p_y > 0)
                targetEntropy -= p_y * Math.Log(p_y);
        }

        _informationGains = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }

            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            var bins = new int[n];
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                bins[i] = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
            }

            // Compute conditional entropy H(Y|X)
            var binCounts = new int[_nBins];
            var jointCounts = new Dictionary<(int, int), int>();

            for (int i = 0; i < n; i++)
            {
                int bin = bins[i];
                int label = (int)Math.Round(NumOps.ToDouble(target[i]));

                binCounts[bin]++;

                var key = (bin, label);
                if (!jointCounts.ContainsKey(key))
                    jointCounts[key] = 0;
                jointCounts[key]++;
            }

            double conditionalEntropy = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] == 0) continue;

                double p_x = (double)binCounts[b] / n;
                double h_y_given_x = 0;

                foreach (int label in targetCounts.Keys)
                {
                    var key = (b, label);
                    if (jointCounts.TryGetValue(key, out int jointCount) && jointCount > 0)
                    {
                        double p_y_given_x = (double)jointCount / binCounts[b];
                        h_y_given_x -= p_y_given_x * Math.Log(p_y_given_x);
                    }
                }

                conditionalEntropy += p_x * h_y_given_x;
            }

            // Information Gain = H(Y) - H(Y|X)
            _informationGains[j] = targetEntropy - conditionalEntropy;
        }

        // Select top features by Information Gain
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _informationGains
            .Select((ig, idx) => (IG: ig, Index: idx))
            .OrderByDescending(x => x.IG)
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
