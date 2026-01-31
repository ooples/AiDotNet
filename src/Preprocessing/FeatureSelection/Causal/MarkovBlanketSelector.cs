using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// Markov Blanket Feature Selection for finding causal features.
/// </summary>
/// <remarks>
/// <para>
/// Identifies the Markov blanket of the target variable - the minimal set of
/// features that makes the target conditionally independent of all other
/// features. These features are the most relevant for prediction.
/// </para>
/// <para><b>For Beginners:</b> Imagine the target variable has a "blanket" of
/// features around it that shields it from all other features. If you know
/// the blanket features, knowing any other feature doesn't help predict the
/// target. This method finds that protective blanket of features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MarkovBlanketSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxFeatures;
    private readonly double _threshold;
    private readonly int _nBins;

    private double[]? _relevanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public double[]? RelevanceScores => _relevanceScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MarkovBlanketSelector(
        int maxFeatures = 20,
        double threshold = 0.01,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));

        _maxFeatures = maxFeatures;
        _threshold = threshold;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MarkovBlanketSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert and discretize data
        var X = DiscretizeData(data, n, p);
        var y = DiscretizeTarget(target, n);

        // Initialize relevance scores
        _relevanceScores = new double[p];

        // Compute mutual information with target for each feature
        for (int j = 0; j < p; j++)
            _relevanceScores[j] = ComputeMutualInformation(X, y, j, n);

        // Grow-Shrink algorithm for Markov blanket
        var blanket = new HashSet<int>();
        var candidates = Enumerable.Range(0, p)
            .OrderByDescending(j => _relevanceScores[j])
            .ToList();

        // Growing phase: add features that increase conditional MI
        foreach (int candidate in candidates)
        {
            if (blanket.Count >= _maxFeatures) break;

            double conditionalMI = ComputeConditionalMI(X, y, candidate, blanket, n);
            if (conditionalMI > _threshold)
                blanket.Add(candidate);
        }

        // Shrinking phase: remove features that become redundant
        var blanketList = blanket.ToList();
        for (int i = blanketList.Count - 1; i >= 0; i--)
        {
            int feature = blanketList[i];
            var others = blanket.Where(f => f != feature).ToHashSet();

            double conditionalMI = ComputeConditionalMI(X, y, feature, others, n);
            if (conditionalMI <= _threshold)
                blanket.Remove(feature);
        }

        _selectedIndices = blanket.OrderBy(x => x).ToArray();

        // If blanket is empty, select top features by MI
        if (_selectedIndices.Length == 0)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _relevanceScores[j])
                .Take(Math.Min(5, p))
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private int[,] DiscretizeData(Matrix<T> data, int n, int p)
    {
        var result = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }

            double range = max - min;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                result[i, j] = range > 1e-10
                    ? Math.Min((int)((val - min) / range * (_nBins - 1)), _nBins - 1)
                    : 0;
            }
        }
        return result;
    }

    private int[] DiscretizeTarget(Vector<T> target, int n)
    {
        var result = new int[n];
        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            min = Math.Min(min, val);
            max = Math.Max(max, val);
        }

        double range = max - min;
        for (int i = 0; i < n; i++)
        {
            double val = NumOps.ToDouble(target[i]);
            result[i] = range > 1e-10
                ? Math.Min((int)((val - min) / range * (_nBins - 1)), _nBins - 1)
                : 0;
        }
        return result;
    }

    private double ComputeMutualInformation(int[,] X, int[] y, int feature, int n)
    {
        var jointCounts = new Dictionary<(int, int), int>();
        var xCounts = new int[_nBins];
        var yCounts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int xVal = X[i, feature];
            int yVal = y[i];
            xCounts[xVal]++;
            yCounts[yVal]++;
            var key = (xVal, yVal);
            jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
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

    private double ComputeConditionalMI(int[,] X, int[] y, int feature, HashSet<int> conditionSet, int n)
    {
        if (conditionSet.Count == 0)
            return ComputeMutualInformation(X, y, feature, n);

        // Simplified: compute MI conditioned on each conditioning variable and average
        double totalMI = 0;
        int count = 0;

        foreach (int condFeature in conditionSet)
        {
            // Group by conditioning variable value
            var groups = Enumerable.Range(0, n)
                .GroupBy(i => X[i, condFeature])
                .Where(g => g.Count() > 5)
                .ToList();

            foreach (var group in groups)
            {
                var indices = group.ToList();
                int groupN = indices.Count;

                var jointCounts = new Dictionary<(int, int), int>();
                var xCounts = new int[_nBins];
                var yCounts = new int[_nBins];

                foreach (int i in indices)
                {
                    int xVal = X[i, feature];
                    int yVal = y[i];
                    xCounts[xVal]++;
                    yCounts[yVal]++;
                    var key = (xVal, yVal);
                    jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
                }

                double groupMI = 0;
                foreach (var kvp in jointCounts)
                {
                    double pxy = (double)kvp.Value / groupN;
                    double px = (double)xCounts[kvp.Key.Item1] / groupN;
                    double py = (double)yCounts[kvp.Key.Item2] / groupN;
                    if (pxy > 0 && px > 0 && py > 0)
                        groupMI += pxy * Math.Log(pxy / (px * py));
                }

                totalMI += groupMI * ((double)groupN / n);
                count++;
            }
        }

        return count > 0 ? totalMI : ComputeMutualInformation(X, y, feature, n);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MarkovBlanketSelector has not been fitted.");

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
        throw new NotSupportedException("MarkovBlanketSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MarkovBlanketSelector has not been fitted.");

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
