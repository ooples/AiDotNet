using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label Mutual Information (ML-MI) for multi-label feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ML-MI extends mutual information to handle multi-label problems by computing
/// the mutual information between each feature and the entire label set. It can
/// consider label correlations and dependencies.
/// </para>
/// <para><b>For Beginners:</b> Mutual information measures how much knowing one
/// thing tells you about another. In multi-label problems, ML-MI measures how
/// much each feature tells you about the entire combination of labels an instance
/// might have.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MLMI<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _considerLabelCorrelations;

    private double[]? _mutualInformation;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool ConsiderLabelCorrelations => _considerLabelCorrelations;
    public double[]? MutualInformation => _mutualInformation;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MLMI(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool considerLabelCorrelations = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _considerLabelCorrelations = considerLabelCorrelations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MLMI requires label matrix. Use Fit(Matrix<T> data, Matrix<T> labels) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> labels)
    {
        if (data.Rows != labels.Rows)
            throw new ArgumentException("Number of samples must match between data and labels.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int numLabels = labels.Columns;

        _mutualInformation = new double[p];

        // Convert labels to binary
        var binaryLabels = new int[n, numLabels];
        for (int i = 0; i < n; i++)
            for (int l = 0; l < numLabels; l++)
                binaryLabels[i, l] = NumOps.ToDouble(labels[i, l]) > 0.5 ? 1 : 0;

        // Compute mutual information for each feature
        for (int j = 0; j < p; j++)
        {
            if (_considerLabelCorrelations)
            {
                // Compute MI with label powerset (subset of common combinations)
                _mutualInformation[j] = ComputeMIWithLabelSet(data, binaryLabels, j, n, numLabels);
            }
            else
            {
                // Sum of MI with individual labels
                double totalMI = 0;
                for (int l = 0; l < numLabels; l++)
                {
                    var singleLabel = new int[n];
                    for (int i = 0; i < n; i++)
                        singleLabel[i] = binaryLabels[i, l];

                    totalMI += ComputeMIWithSingleLabel(data, singleLabel, j, n);
                }
                _mutualInformation[j] = totalMI / numLabels;
            }
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _mutualInformation
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Convert single-label to multi-label format
        int n = data.Rows;
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
            uniqueLabels.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        var labelList = uniqueLabels.OrderBy(x => x).ToList();
        int numLabels = labelList.Count;

        var labelsArray = new T[n, numLabels];
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            int labelIdx = labelList.IndexOf(label);
            labelsArray[i, labelIdx] = NumOps.FromDouble(1.0);
        }

        Fit(data, new Matrix<T>(labelsArray));
    }

    private double ComputeMIWithLabelSet(Matrix<T> data, int[,] labels, int featureIdx, int n, int numLabels)
    {
        // Discretize feature
        var featureValues = new double[n];
        double minVal = double.MaxValue, maxVal = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            featureValues[i] = NumOps.ToDouble(data[i, featureIdx]);
            if (featureValues[i] < minVal) minVal = featureValues[i];
            if (featureValues[i] > maxVal) maxVal = featureValues[i];
        }

        var discretizedFeature = new int[n];
        double range = maxVal - minVal;
        if (range < 1e-10)
        {
            for (int i = 0; i < n; i++)
                discretizedFeature[i] = 0;
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                int bin = (int)((featureValues[i] - minVal) / range * (_nBins - 1));
                discretizedFeature[i] = Math.Min(_nBins - 1, Math.Max(0, bin));
            }
        }

        // Create label combination signature
        var labelSignatures = new Dictionary<string, int>();
        var sampleSignatures = new int[n];

        for (int i = 0; i < n; i++)
        {
            var sig = new System.Text.StringBuilder();
            for (int l = 0; l < numLabels; l++)
                sig.Append(labels[i, l]);

            string sigStr = sig.ToString();
            if (!labelSignatures.ContainsKey(sigStr))
                labelSignatures[sigStr] = labelSignatures.Count;

            sampleSignatures[i] = labelSignatures[sigStr];
        }

        // Compute joint and marginal distributions
        int numLabelCombos = labelSignatures.Count;
        var jointCounts = new int[_nBins, numLabelCombos];
        var featureCounts = new int[_nBins];
        var labelCounts = new int[numLabelCombos];

        for (int i = 0; i < n; i++)
        {
            jointCounts[discretizedFeature[i], sampleSignatures[i]]++;
            featureCounts[discretizedFeature[i]]++;
            labelCounts[sampleSignatures[i]]++;
        }

        // Compute MI
        double mi = 0;
        for (int f = 0; f < _nBins; f++)
        {
            for (int l = 0; l < numLabelCombos; l++)
            {
                if (jointCounts[f, l] > 0)
                {
                    double pJoint = (double)jointCounts[f, l] / n;
                    double pFeature = (double)featureCounts[f] / n;
                    double pLabel = (double)labelCounts[l] / n;

                    if (pFeature > 0 && pLabel > 0)
                        mi += pJoint * Math.Log(pJoint / (pFeature * pLabel) + 1e-10);
                }
            }
        }

        return mi;
    }

    private double ComputeMIWithSingleLabel(Matrix<T> data, int[] labels, int featureIdx, int n)
    {
        // Discretize feature
        var featureValues = new double[n];
        double minVal = double.MaxValue, maxVal = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            featureValues[i] = NumOps.ToDouble(data[i, featureIdx]);
            if (featureValues[i] < minVal) minVal = featureValues[i];
            if (featureValues[i] > maxVal) maxVal = featureValues[i];
        }

        var discretizedFeature = new int[n];
        double range = maxVal - minVal;
        if (range < 1e-10)
        {
            for (int i = 0; i < n; i++)
                discretizedFeature[i] = 0;
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                int bin = (int)((featureValues[i] - minVal) / range * (_nBins - 1));
                discretizedFeature[i] = Math.Min(_nBins - 1, Math.Max(0, bin));
            }
        }

        // Compute counts
        var jointCounts = new int[_nBins, 2];
        var featureCounts = new int[_nBins];
        var labelCounts = new int[2];

        for (int i = 0; i < n; i++)
        {
            jointCounts[discretizedFeature[i], labels[i]]++;
            featureCounts[discretizedFeature[i]]++;
            labelCounts[labels[i]]++;
        }

        // Compute MI
        double mi = 0;
        for (int f = 0; f < _nBins; f++)
        {
            for (int l = 0; l < 2; l++)
            {
                if (jointCounts[f, l] > 0)
                {
                    double pJoint = (double)jointCounts[f, l] / n;
                    double pFeature = (double)featureCounts[f] / n;
                    double pLabel = (double)labelCounts[l] / n;

                    if (pFeature > 0 && pLabel > 0)
                        mi += pJoint * Math.Log(pJoint / (pFeature * pLabel) + 1e-10);
                }
            }
        }

        return mi;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> labels)
    {
        Fit(data, labels);
        return Transform(data);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MLMI has not been fitted.");

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
        throw new NotSupportedException("MLMI does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MLMI has not been fitted.");

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
