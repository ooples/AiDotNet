using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label Mutual Information for multi-label classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Extends mutual information to handle multiple target labels simultaneously.
/// Aggregates MI scores across all labels to identify features relevant to
/// multiple prediction targets.
/// </para>
/// <para><b>For Beginners:</b> In multi-label classification, each sample can
/// have multiple labels at once. This method finds features that are informative
/// for predicting multiple labels, not just one.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiLabelMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly AggregationMethod _aggregation;

    public enum AggregationMethod { Sum, Average, Max, Min }

    private double[]? _miScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? MIScores => _miScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MultiLabelMutualInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        AggregationMethod aggregation = AggregationMethod.Sum,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _aggregation = aggregation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MultiLabelMutualInformation requires target values. Use Fit(Matrix<T> data, Matrix<T> targets) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> targets)
    {
        if (data.Rows != targets.Rows)
            throw new ArgumentException("Target rows must match data rows.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabels = targets.Columns;

        // Compute MI for each feature-label pair
        var miMatrix = new double[p, nLabels];

        for (int j = 0; j < p; j++)
        {
            var xDiscrete = DiscretizeFeature(data, j, n);

            for (int l = 0; l < nLabels; l++)
            {
                var yDiscrete = DiscretizeLabel(targets, l, n);
                miMatrix[j, l] = ComputeMI(xDiscrete, yDiscrete, n);
            }
        }

        // Aggregate across labels
        _miScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            var labelScores = new double[nLabels];
            for (int l = 0; l < nLabels; l++)
                labelScores[l] = miMatrix[j, l];

            _miScores[j] = _aggregation switch
            {
                AggregationMethod.Sum => labelScores.Sum(),
                AggregationMethod.Average => labelScores.Average(),
                AggregationMethod.Max => labelScores.Max(),
                AggregationMethod.Min => labelScores.Min(),
                _ => labelScores.Sum()
            };
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _miScores
            .Select((mi, idx) => (MI: mi, Index: idx))
            .OrderByDescending(x => x.MI)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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

    private int[] DiscretizeLabel(Matrix<T> targets, int l, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(targets[i, l]);

        // Binary labels
        var result = new int[n];
        for (int i = 0; i < n; i++)
            result[i] = values[i] > 0.5 ? 1 : 0;

        return result;
    }

    private double ComputeMI(int[] x, int[] y, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var xCounts = new Dictionary<int, int>();
        var yCounts = new Dictionary<int, int>();

        for (int i = 0; i < n; i++)
        {
            var key = (x[i], y[i]);
            if (!jointCounts.ContainsKey(key)) jointCounts[key] = 0;
            jointCounts[key]++;

            if (!xCounts.ContainsKey(x[i])) xCounts[x[i]] = 0;
            xCounts[x[i]]++;

            if (!yCounts.ContainsKey(y[i])) yCounts[y[i]] = 0;
            yCounts[y[i]]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            double pxy = (double)kvp.Value / n;
            double px = (double)xCounts[kvp.Key.Item1] / n;
            double py = (double)yCounts[kvp.Key.Item2] / n;

            if (pxy > 0 && px > 0 && py > 0)
                mi += pxy * Math.Log(pxy / (px * py));
        }

        return mi;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> targets)
    {
        Fit(data, targets);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelMutualInformation has not been fitted.");

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
        throw new NotSupportedException("MultiLabelMutualInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelMutualInformation has not been fitted.");

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
