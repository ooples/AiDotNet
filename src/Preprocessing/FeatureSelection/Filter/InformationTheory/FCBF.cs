using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Fast Correlation-Based Filter (FCBF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// FCBF uses Symmetric Uncertainty to evaluate feature-target and feature-feature
/// correlations. It selects features that are highly correlated with the target
/// but not redundant with other selected features, using a fast backwards elimination
/// process.
/// </para>
/// <para><b>For Beginners:</b> Having two features that say the same thing is wasteful.
/// FCBF first picks features that are useful for prediction, then removes features
/// that duplicate what other selected features already tell you. It's like building
/// a team where each member brings unique skills.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FCBF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;
    private readonly int _nBins;

    private double[]? _suScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public int NBins => _nBins;
    public double[]? SUScores => _suScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FCBF(
        int nFeaturesToSelect = 10,
        double threshold = 0.0,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FCBF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Step 1: Compute SU for all features with target
        _suScores = new double[p];
        var bins = DiscretizeAll(data);

        for (int j = 0; j < p; j++)
            _suScores[j] = ComputeSU(bins[j], target, n);

        // Step 2: Filter features above threshold and sort by SU
        var candidates = Enumerable.Range(0, p)
            .Where(j => _suScores[j] >= _threshold)
            .OrderByDescending(j => _suScores[j])
            .ToList();

        if (candidates.Count == 0)
        {
            // If no features pass threshold, take top features by SU
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _suScores[j])
                .Take(Math.Min(_nFeaturesToSelect, p))
                .OrderBy(x => x)
                .ToArray();
            IsFitted = true;
            return;
        }

        // Step 3: Remove redundant features
        var selected = new List<int>();
        int i = 0;

        while (i < candidates.Count && selected.Count < _nFeaturesToSelect)
        {
            int fi = candidates[i];
            selected.Add(fi);

            // Remove features that are predominated by fi
            candidates = candidates
                .Where((fj, idx) => idx <= i || !IsPredominant(fi, fj, bins, n))
                .ToList();

            i++;
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private Dictionary<int, int[]> DiscretizeAll(Matrix<T> data)
    {
        int n = data.Rows;
        int p = data.Columns;
        var result = new Dictionary<int, int[]>();

        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }

            double binWidth = (max - min) / _nBins;
            if (binWidth < 1e-10) binWidth = 1;

            result[j] = new int[n];
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                result[j][i] = Math.Min((int)((val - min) / binWidth), _nBins - 1);
            }
        }

        return result;
    }

    private double ComputeSU(int[] bins, Vector<T> target, int n)
    {
        // Compute entropies and mutual information
        var binCount = new Dictionary<int, int>();
        var classCount = new Dictionary<int, int>();
        var jointCount = new Dictionary<(int, int), int>();

        for (int i = 0; i < n; i++)
        {
            int bin = bins[i];
            int c = (int)Math.Round(NumOps.ToDouble(target[i]));

            if (!binCount.ContainsKey(bin)) binCount[bin] = 0;
            binCount[bin]++;

            if (!classCount.ContainsKey(c)) classCount[c] = 0;
            classCount[c]++;

            var key = (bin, c);
            if (!jointCount.ContainsKey(key)) jointCount[key] = 0;
            jointCount[key]++;
        }

        double hX = 0;
        foreach (var count in binCount.Values)
        {
            double p = (double)count / n;
            if (p > 0) hX -= p * Math.Log(p);
        }

        double hY = 0;
        foreach (var count in classCount.Values)
        {
            double p = (double)count / n;
            if (p > 0) hY -= p * Math.Log(p);
        }

        double hXY = 0;
        foreach (var count in jointCount.Values)
        {
            double p = (double)count / n;
            if (p > 0) hXY -= p * Math.Log(p);
        }

        double mi = hX + hY - hXY;
        double denominator = hX + hY;

        return denominator > 1e-10 ? 2.0 * mi / denominator : 0;
    }

    private double ComputeSU(int[] bins1, int[] bins2, int n)
    {
        var count1 = new Dictionary<int, int>();
        var count2 = new Dictionary<int, int>();
        var jointCount = new Dictionary<(int, int), int>();

        for (int i = 0; i < n; i++)
        {
            int b1 = bins1[i];
            int b2 = bins2[i];

            if (!count1.ContainsKey(b1)) count1[b1] = 0;
            count1[b1]++;

            if (!count2.ContainsKey(b2)) count2[b2] = 0;
            count2[b2]++;

            var key = (b1, b2);
            if (!jointCount.ContainsKey(key)) jointCount[key] = 0;
            jointCount[key]++;
        }

        double h1 = 0;
        foreach (var count in count1.Values)
        {
            double p = (double)count / n;
            if (p > 0) h1 -= p * Math.Log(p);
        }

        double h2 = 0;
        foreach (var count in count2.Values)
        {
            double p = (double)count / n;
            if (p > 0) h2 -= p * Math.Log(p);
        }

        double hJoint = 0;
        foreach (var count in jointCount.Values)
        {
            double p = (double)count / n;
            if (p > 0) hJoint -= p * Math.Log(p);
        }

        double mi = h1 + h2 - hJoint;
        double denominator = h1 + h2;

        return denominator > 1e-10 ? 2.0 * mi / denominator : 0;
    }

    private bool IsPredominant(int fi, int fj, Dictionary<int, int[]> bins, int n)
    {
        if (fi == fj) return false;

        // fj is predominated by fi if SU(fi, fj) >= SU(fj, target)
        // and _suScores[fi] >= _suScores[fj]
        double su_fi_fj = ComputeSU(bins[fi], bins[fj], n);

        return su_fi_fj >= _suScores![fj];
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCBF has not been fitted.");

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
        throw new NotSupportedException("FCBF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCBF has not been fitted.");

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
