using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Mutual Information for classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures the mutual information between each feature and the target class.
/// Captures any kind of dependency (linear or nonlinear) between features and target.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing a feature
/// tells you about the class. If knowing the feature value significantly reduces
/// uncertainty about the class, the mutual information is high. Unlike correlation,
/// it can capture any type of relationship, not just linear ones.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MutualInformationClassification<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _discreteFeatures;

    private double[]? _mutualInfoScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MutualInfoScores => _mutualInfoScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MutualInformationClassification(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool discreteFeatures = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _discreteFeatures = discreteFeatures;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MutualInformationClassification requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique classes
        var classes = new HashSet<int>();
        for (int i = 0; i < n; i++)
            classes.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        int nClasses = classes.Count;
        var classLabels = classes.OrderBy(x => x).ToArray();
        var classIndex = classLabels.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

        // Class probabilities
        var classProbs = new double[nClasses];
        for (int i = 0; i < n; i++)
        {
            int c = classIndex[(int)Math.Round(NumOps.ToDouble(target[i]))];
            classProbs[c] += 1.0 / n;
        }

        _mutualInfoScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            int nFeatureBins;
            int[] binAssignments = new int[n];

            if (_discreteFeatures)
            {
                // Treat feature values as discrete
                var uniqueValues = new HashSet<double>();
                for (int i = 0; i < n; i++)
                    uniqueValues.Add(NumOps.ToDouble(data[i, j]));

                var valueList = uniqueValues.OrderBy(x => x).ToList();
                var valueIndex = valueList.Select((v, idx) => (v, idx))
                    .ToDictionary(x => x.v, x => x.idx);

                nFeatureBins = uniqueValues.Count;
                for (int i = 0; i < n; i++)
                    binAssignments[i] = valueIndex[NumOps.ToDouble(data[i, j])];
            }
            else
            {
                // Bin continuous feature
                double minVal = double.MaxValue, maxVal = double.MinValue;
                for (int i = 0; i < n; i++)
                {
                    double val = NumOps.ToDouble(data[i, j]);
                    minVal = Math.Min(minVal, val);
                    maxVal = Math.Max(maxVal, val);
                }

                double range = maxVal - minVal;
                if (range < 1e-10) range = 1;

                nFeatureBins = _nBins;
                for (int i = 0; i < n; i++)
                {
                    double val = NumOps.ToDouble(data[i, j]);
                    binAssignments[i] = Math.Min((int)(((val - minVal) / range) * _nBins), _nBins - 1);
                }
            }

            // Compute joint and marginal probabilities
            var jointProb = new double[nFeatureBins, nClasses];
            var featureProb = new double[nFeatureBins];

            for (int i = 0; i < n; i++)
            {
                int bin = binAssignments[i];
                int c = classIndex[(int)Math.Round(NumOps.ToDouble(target[i]))];
                jointProb[bin, c] += 1.0 / n;
                featureProb[bin] += 1.0 / n;
            }

            // Compute mutual information
            double mi = 0;
            for (int b = 0; b < nFeatureBins; b++)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    if (jointProb[b, c] > 1e-10 && featureProb[b] > 1e-10 && classProbs[c] > 1e-10)
                    {
                        mi += jointProb[b, c] * Math.Log(jointProb[b, c] / (featureProb[b] * classProbs[c]));
                    }
                }
            }

            _mutualInfoScores[j] = Math.Max(0, mi);
        }

        // Select top features by MI
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _mutualInfoScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
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
            throw new InvalidOperationException("MutualInformationClassification has not been fitted.");

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
        throw new NotSupportedException("MutualInformationClassification does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInformationClassification has not been fitted.");

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
