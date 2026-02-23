using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Information;

/// <summary>
/// Interaction Information-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their interaction information with the target,
/// capturing synergistic and redundant relationships between feature pairs.
/// </para>
/// <para><b>For Beginners:</b> Interaction information measures whether two
/// features together tell us more (synergy) or less (redundancy) about the
/// target than we'd expect from their individual contributions. This helps
/// find features that work well together.
/// </para>
/// </remarks>
public class InteractionInformationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? InteractionScores => _interactionScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public InteractionInformationSelector(
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
            "InteractionInformationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Discretize all features and target
        var discretized = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++) col[i] = X[i, j];
            var bins = Discretize(col, n, _nBins);
            for (int i = 0; i < n; i++) discretized[i, j] = bins[i];
        }
        var yBins = Discretize(y, n, _nBins);

        // Compute individual mutual information
        var mi = new double[p];
        for (int j = 0; j < p; j++)
        {
            var xBins = new int[n];
            for (int i = 0; i < n; i++) xBins[i] = discretized[i, j];
            mi[j] = ComputeMutualInformation(xBins, yBins, n, _nBins);
        }

        // Compute interaction information scores
        _interactionScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            _interactionScores[j] = mi[j];

            // Add average interaction information with other features
            double totalInteraction = 0;
            int count = 0;
            for (int k = 0; k < p; k++)
            {
                if (j == k) continue;

                var xj = new int[n];
                var xk = new int[n];
                for (int i = 0; i < n; i++)
                {
                    xj[i] = discretized[i, j];
                    xk[i] = discretized[i, k];
                }

                // I(X;Y;Z) = I(X;Y) - I(X;Y|Z) = I(X;Y) + I(Z;Y) - I(X,Z;Y)
                double miJK = ComputeJointMI(xj, xk, yBins, n, _nBins);
                double interaction = mi[j] + mi[k] - miJK;
                totalInteraction += Math.Abs(interaction);
                count++;
            }

            if (count > 0)
                _interactionScores[j] += totalInteraction / count;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _interactionScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] Discretize(double[] values, int n, int nBins)
    {
        double min = values.Min();
        double max = values.Max();
        double range = max - min + 1e-10;

        var bins = new int[n];
        for (int i = 0; i < n; i++)
            bins[i] = Math.Min((int)((values[i] - min) / range * nBins), nBins - 1);

        return bins;
    }

    private double ComputeMutualInformation(int[] x, int[] y, int n, int nBins)
    {
        var jointCounts = new int[nBins, nBins];
        var xCounts = new int[nBins];
        var yCounts = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[x[i], y[i]]++;
            xCounts[x[i]]++;
            yCounts[y[i]]++;
        }

        double mi = 0;
        for (int xi = 0; xi < nBins; xi++)
        {
            for (int yi = 0; yi < nBins; yi++)
            {
                if (jointCounts[xi, yi] == 0) continue;
                double pxy = (double)jointCounts[xi, yi] / n;
                double px = (double)xCounts[xi] / n;
                double py = (double)yCounts[yi] / n;
                mi += pxy * Math.Log(pxy / (px * py)) / Math.Log(2);
            }
        }

        return mi;
    }

    private double ComputeJointMI(int[] x1, int[] x2, int[] y, int n, int nBins)
    {
        // MI between (X1,X2) and Y
        var jointCounts = new Dictionary<(int, int, int), int>();
        var x12Counts = new Dictionary<(int, int), int>();
        var yCounts = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            var key = (x1[i], x2[i], y[i]);
            jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;

            var x12Key = (x1[i], x2[i]);
            x12Counts[x12Key] = x12Counts.GetValueOrDefault(x12Key) + 1;

            yCounts[y[i]]++;
        }

        double mi = 0;
        foreach (var kvp in jointCounts)
        {
            var (xi1, xi2, yi) = kvp.Key;
            double pxxy = (double)kvp.Value / n;
            double pxx = (double)x12Counts[(xi1, xi2)] / n;
            double py = (double)yCounts[yi] / n;
            mi += pxxy * Math.Log(pxxy / (pxx * py)) / Math.Log(2);
        }

        return mi;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionInformationSelector has not been fitted.");

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
        throw new NotSupportedException("InteractionInformationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionInformationSelector has not been fitted.");

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
