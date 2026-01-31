using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// Mutual Information for classification-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Mutual Information Classification estimates the mutual information between
/// each feature and the discrete class label. MI measures how much knowing
/// the feature reduces uncertainty about the class.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing
/// a feature's value helps predict the class. Unlike correlation, MI can detect
/// any type of relationship, not just linear ones. A high MI score means the
/// feature contains useful information for classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MutualInfoClassification<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _miScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? MIScores => _miScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MutualInfoClassification(
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
            "MutualInfoClassification requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique class labels
        var classLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
            classLabels.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        int nClasses = classLabels.Count;
        var labelList = classLabels.OrderBy(x => x).ToList();

        // Count class frequencies
        var classCounts = new int[nClasses];
        for (int i = 0; i < n; i++)
        {
            int cIdx = labelList.IndexOf((int)Math.Round(NumOps.ToDouble(target[i])));
            classCounts[cIdx]++;
        }

        _miScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Discretize feature
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }

            double range = maxVal - minVal;

            // Build joint distribution
            var jointCounts = new int[_nBins, nClasses];
            var featureCounts = new int[_nBins];

            for (int i = 0; i < n; i++)
            {
                int fBin = range > 1e-10
                    ? Math.Min(_nBins - 1, (int)((NumOps.ToDouble(data[i, j]) - minVal) / range * (_nBins - 1)))
                    : 0;
                int cIdx = labelList.IndexOf((int)Math.Round(NumOps.ToDouble(target[i])));

                jointCounts[fBin, cIdx]++;
                featureCounts[fBin]++;
            }

            // Compute mutual information
            double mi = 0;
            for (int f = 0; f < _nBins; f++)
            {
                for (int c = 0; c < nClasses; c++)
                {
                    if (jointCounts[f, c] > 0)
                    {
                        double pJoint = (double)jointCounts[f, c] / n;
                        double pFeature = (double)featureCounts[f] / n;
                        double pClass = (double)classCounts[c] / n;

                        if (pFeature > 0 && pClass > 0)
                            mi += pJoint * Math.Log(pJoint / (pFeature * pClass) + 1e-10);
                    }
                }
            }

            _miScores[j] = Math.Max(0, mi);
        }

        // Select top features by MI
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _miScores[j])
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
            throw new InvalidOperationException("MutualInfoClassification has not been fitted.");

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
        throw new NotSupportedException("MutualInfoClassification does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInfoClassification has not been fitted.");

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
