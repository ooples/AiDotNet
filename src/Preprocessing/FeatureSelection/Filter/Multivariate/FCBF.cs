using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Fast Correlation-Based Filter (FCBF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// FCBF uses symmetrical uncertainty to identify relevant features and remove
/// redundant ones. It's efficient for high-dimensional data with the concept
/// of predominant features that are not dominated by any other feature.
/// </para>
/// <para><b>For Beginners:</b> FCBF finds features that are strongly related to
/// the target and removes any feature that's "dominated" by another (meaning
/// the other feature is just as good at predicting the target AND is more
/// correlated with this feature). This efficiently eliminates redundancy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FCBF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    private double[]? _symmetricalUncertainties;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double[]? SymmetricalUncertainties => _symmetricalUncertainties;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FCBF(
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _threshold = threshold;
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

        // Discretize continuous features for MI computation
        int nBins = 10;
        var discretizedData = new int[n, p];
        var targetDiscrete = new int[n];

        // Discretize target
        var uniqueClasses = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            targetDiscrete[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            uniqueClasses.Add(targetDiscrete[i]);
        }
        int nClasses = uniqueClasses.Count;

        // Discretize features
        for (int j = 0; j < p; j++)
        {
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, val);
                maxVal = Math.Max(maxVal, val);
            }
            double range = maxVal - minVal;
            if (range < 1e-10) range = 1;

            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                discretizedData[i, j] = Math.Min((int)(((val - minVal) / range) * nBins), nBins - 1);
            }
        }

        // Compute symmetrical uncertainty between each feature and target
        _symmetricalUncertainties = new double[p];
        for (int j = 0; j < p; j++)
        {
            var featureValues = new int[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = discretizedData[i, j];

            _symmetricalUncertainties[j] = ComputeSymmetricalUncertainty(featureValues, targetDiscrete, nBins, nClasses);
        }

        // Get relevant features (SU > threshold)
        var relevant = new List<int>();
        for (int j = 0; j < p; j++)
            if (_symmetricalUncertainties[j] > _threshold)
                relevant.Add(j);

        // Sort by SU descending
        relevant = relevant.OrderByDescending(j => _symmetricalUncertainties[j]).ToList();

        // Remove redundant features
        var selectedList = new List<int>();
        var processed = new HashSet<int>();

        foreach (int fp in relevant)
        {
            if (processed.Contains(fp))
                continue;

            selectedList.Add(fp);
            processed.Add(fp);

            // Check remaining features for dominance
            foreach (int fq in relevant.Where(f => !processed.Contains(f)))
            {
                var fp_values = new int[n];
                var fq_values = new int[n];
                for (int i = 0; i < n; i++)
                {
                    fp_values[i] = discretizedData[i, fp];
                    fq_values[i] = discretizedData[i, fq];
                }

                double su_pq = ComputeSymmetricalUncertainty(fp_values, fq_values, nBins, nBins);

                // If SU(fp, fq) >= SU(fq, target), fq is dominated
                if (su_pq >= _symmetricalUncertainties[fq])
                    processed.Add(fq);
            }
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        if (_selectedIndices.Length == 0)
        {
            // Fall back to top by SU
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _symmetricalUncertainties[j])
                .Take(Math.Min(10, p))
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double ComputeSymmetricalUncertainty(int[] x, int[] y, int nBinsX, int nBinsY)
    {
        int n = x.Length;

        // Compute entropies
        var pX = new double[nBinsX];
        var pY = new double[nBinsY];
        var pXY = new double[nBinsX, nBinsY];

        for (int i = 0; i < n; i++)
        {
            pX[x[i]] += 1.0 / n;
            pY[y[i]] += 1.0 / n;
            pXY[x[i], y[i]] += 1.0 / n;
        }

        double hX = 0, hY = 0, hXY = 0;
        for (int i = 0; i < nBinsX; i++)
            if (pX[i] > 1e-10) hX -= pX[i] * Math.Log(pX[i]);

        for (int i = 0; i < nBinsY; i++)
            if (pY[i] > 1e-10) hY -= pY[i] * Math.Log(pY[i]);

        for (int i = 0; i < nBinsX; i++)
            for (int j = 0; j < nBinsY; j++)
                if (pXY[i, j] > 1e-10) hXY -= pXY[i, j] * Math.Log(pXY[i, j]);

        // Mutual information
        double mi = hX + hY - hXY;

        // Symmetrical uncertainty
        double denom = hX + hY;
        return denom > 1e-10 ? 2 * mi / denom : 0;
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
