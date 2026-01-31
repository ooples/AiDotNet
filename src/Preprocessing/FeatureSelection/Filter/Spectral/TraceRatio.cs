using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Trace Ratio criterion for multi-class feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Trace Ratio maximizes the ratio of between-class scatter to within-class
/// scatter. Unlike Fisher Score which works per feature, Trace Ratio considers
/// feature interactions through scatter matrices.
/// </para>
/// <para><b>For Beginners:</b> This extends Fisher Score to consider how features
/// work together. Instead of scoring each feature independently, it evaluates
/// how a set of features collectively separates classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TraceRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _traceRatioValue;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public double TraceRatioValue => _traceRatioValue;
    public override bool SupportsInverseTransform => false;

    public TraceRatio(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TraceRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classGroups = new Dictionary<double, List<int>>();
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            if (!classGroups.ContainsKey(y))
                classGroups[y] = new List<int>();
            classGroups[y].Add(i);
        }

        // Compute global mean
        var globalMean = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                globalMean[j] += NumOps.ToDouble(data[i, j]);
            globalMean[j] /= n;
        }

        // Compute between-class and within-class scatter (diagonal approximation)
        var Sb = new double[p];  // Diagonal of between-class scatter
        var Sw = new double[p];  // Diagonal of within-class scatter

        foreach (var kvp in classGroups)
        {
            var classIndices = kvp.Value;
            int nk = classIndices.Count;

            // Class mean
            var classMean = new double[p];
            foreach (int i in classIndices)
                for (int j = 0; j < p; j++)
                    classMean[j] += NumOps.ToDouble(data[i, j]);
            for (int j = 0; j < p; j++)
                classMean[j] /= nk;

            // Between-class scatter
            for (int j = 0; j < p; j++)
                Sb[j] += nk * Math.Pow(classMean[j] - globalMean[j], 2);

            // Within-class scatter
            foreach (int i in classIndices)
            {
                for (int j = 0; j < p; j++)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean[j];
                    Sw[j] += diff * diff;
                }
            }
        }

        // Compute feature scores (Sb / Sw per feature)
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = Sw[j] > 1e-10 ? Sb[j] / Sw[j] : 0;

        // Greedy selection to maximize trace ratio
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));

        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestRatio = double.NegativeInfinity;

            foreach (int f in remaining)
            {
                // Compute incremental trace ratio
                double sbSum = 0, swSum = 0;
                foreach (int s in selected)
                {
                    sbSum += Sb[s];
                    swSum += Sw[s];
                }
                sbSum += Sb[f];
                swSum += Sw[f];

                double ratio = swSum > 1e-10 ? sbSum / swSum : 0;
                if (ratio > bestRatio)
                {
                    bestRatio = ratio;
                    bestFeature = f;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _traceRatioValue = bestRatio;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("TraceRatio has not been fitted.");

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
        throw new NotSupportedException("TraceRatio does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TraceRatio has not been fitted.");

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
