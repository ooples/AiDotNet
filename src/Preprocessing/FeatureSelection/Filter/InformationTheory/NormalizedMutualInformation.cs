using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Normalized Mutual Information for scale-invariant dependency measurement.
/// </summary>
/// <remarks>
/// <para>
/// Normalized Mutual Information (NMI) divides mutual information by entropy
/// to get a value between 0 and 1, making comparisons across features more fair.
/// </para>
/// <para><b>For Beginners:</b> Regular mutual information can be higher for
/// features with more distinct values. NMI normalizes this so you can fairly
/// compare features with different numbers of unique values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class NormalizedMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly NormalizationMethod _normalization;

    public enum NormalizationMethod { Arithmetic, Geometric, Min, Max }

    private double[]? _nmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? NMIScores => _nmiScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public NormalizedMutualInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        NormalizationMethod normalization = NormalizationMethod.Arithmetic,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _normalization = normalization;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NormalizedMutualInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize target
        var yDiscrete = DiscretizeTarget(target, n);
        double hY = ComputeEntropy(yDiscrete, n);

        _nmiScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var xDiscrete = DiscretizeFeature(data, j, n);
            double hX = ComputeEntropy(xDiscrete, n);
            double hXY = ComputeJointEntropy(xDiscrete, yDiscrete, n);
            double mi = hX + hY - hXY;

            // Apply normalization
            double norm = _normalization switch
            {
                NormalizationMethod.Arithmetic => (hX + hY) / 2,
                NormalizationMethod.Geometric => Math.Sqrt(hX * hY),
                NormalizationMethod.Min => Math.Min(hX, hY),
                NormalizationMethod.Max => Math.Max(hX, hY),
                _ => (hX + hY) / 2
            };

            _nmiScores[j] = norm > 1e-10 ? mi / norm : 0;
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _nmiScores
            .Select((nmi, idx) => (NMI: nmi, Index: idx))
            .OrderByDescending(x => x.NMI)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var values = new double[n];
        for (int i = 0; i < n; i++)
            values[i] = NumOps.ToDouble(target[i]);

        var unique = values.Distinct().OrderBy(x => x).ToList();
        var result = new int[n];

        for (int i = 0; i < n; i++)
            result[i] = unique.IndexOf(values[i]);

        return result;
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

    private double ComputeEntropy(int[] x, int n)
    {
        var counts = new Dictionary<int, int>();
        foreach (int v in x)
        {
            if (!counts.ContainsKey(v)) counts[v] = 0;
            counts[v]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / n;
            if (p > 0) entropy -= p * Math.Log(p);
        }

        return entropy;
    }

    private double ComputeJointEntropy(int[] x, int[] y, int n)
    {
        var counts = new Dictionary<(int, int), int>();
        for (int i = 0; i < n; i++)
        {
            var key = (x[i], y[i]);
            if (!counts.ContainsKey(key)) counts[key] = 0;
            counts[key]++;
        }

        double entropy = 0;
        foreach (var count in counts.Values)
        {
            double p = (double)count / n;
            if (p > 0) entropy -= p * Math.Log(p);
        }

        return entropy;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedMutualInformation has not been fitted.");

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
        throw new NotSupportedException("NormalizedMutualInformation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedMutualInformation has not been fitted.");

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
