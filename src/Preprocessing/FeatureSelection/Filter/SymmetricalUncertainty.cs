using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Symmetrical Uncertainty for feature selection based on normalized mutual information.
/// </summary>
/// <remarks>
/// <para>
/// Symmetrical Uncertainty normalizes mutual information to the range [0, 1] by considering
/// the entropy of both variables. It measures how much knowing one variable reduces
/// uncertainty about the other, symmetrically.
/// </para>
/// <para><b>For Beginners:</b> Mutual information tells you how much one variable reveals
/// about another, but the raw value depends on the variables' complexity. Symmetrical
/// Uncertainty fixes this by scaling the value between 0 and 1, where 0 means no
/// relationship and 1 means perfect predictability in both directions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SymmetricalUncertainty<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _minSU;

    private double[]? _suValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SUValues => _suValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SymmetricalUncertainty(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double minSU = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _minSU = minSU;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SymmetricalUncertainty requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute target entropy
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classCounts.ContainsKey(label))
                classCounts[label] = 0;
            classCounts[label]++;
        }

        double targetEntropy = 0;
        foreach (var count in classCounts.Values)
        {
            double prob = (double)count / n;
            if (prob > 0)
                targetEntropy -= prob * Math.Log(prob);
        }

        int nClasses = classCounts.Count;
        var classes = classCounts.Keys.OrderBy(x => x).ToArray();
        var classIndex = classes.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

        _suValues = new double[p];

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

            // Feature entropy H(X)
            double featureEntropy = 0;
            for (int b = 0; b < _nBins; b++)
            {
                if (binCounts[b] > 0)
                {
                    double prob = (double)binCounts[b] / n;
                    featureEntropy -= prob * Math.Log(prob);
                }
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

            // Symmetrical Uncertainty = 2 * IG / (H(X) + H(Y))
            double denominator = featureEntropy + targetEntropy;
            _suValues[j] = denominator > 1e-10 ? 2 * ig / denominator : 0;
        }

        // Select features above threshold or top by SU
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_suValues[j] >= _minSU)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _suValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _suValues
                .Select((su, idx) => (SU: su, Index: idx))
                .OrderByDescending(x => x.SU)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("SymmetricalUncertainty has not been fitted.");

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
        throw new NotSupportedException("SymmetricalUncertainty does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SymmetricalUncertainty has not been fitted.");

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
