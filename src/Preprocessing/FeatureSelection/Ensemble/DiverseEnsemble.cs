using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Diverse Ensemble Feature Selection using multiple diverse methods.
/// </summary>
/// <remarks>
/// <para>
/// Combines multiple feature selection methods with intentionally different
/// characteristics (filter, wrapper, embedded) to get robust rankings.
/// Features consistently ranked high across diverse methods are selected.
/// </para>
/// <para><b>For Beginners:</b> Different feature selection methods look at
/// features from different angles - some focus on correlation, others on
/// prediction power, others on regularization. By combining diverse perspectives,
/// we identify features that are truly important from multiple viewpoints.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DiverseEnsemble<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;

    private double[]? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AggregatedScores => _aggregatedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DiverseEnsemble(
        int nFeaturesToSelect = 10,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DiverseEnsemble requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var allScores = new List<double[]>();

        // Method 1: Correlation-based
        allScores.Add(ComputeCorrelationScores(data, target, n, p));

        // Method 2: Mutual Information-based
        allScores.Add(ComputeMutualInformationScores(data, target, n, p));

        // Method 3: F-statistic (ANOVA)
        allScores.Add(ComputeFScores(data, target, n, p));

        // Method 4: Variance-based
        allScores.Add(ComputeVarianceScores(data, n, p));

        // Method 5: Distance correlation
        allScores.Add(ComputeDistanceCorrelationScores(data, target, n, p, random));

        // Aggregate scores (normalize and average)
        _aggregatedScores = new double[p];

        foreach (var scores in allScores)
        {
            // Normalize to [0, 1]
            double minScore = scores.Min();
            double maxScore = scores.Max();
            double range = maxScore - minScore;

            for (int j = 0; j < p; j++)
            {
                double normalizedScore = range > 1e-10 ? (scores[j] - minScore) / range : 0;
                _aggregatedScores[j] += normalizedScore;
            }
        }

        // Average across methods
        for (int j = 0; j < p; j++)
            _aggregatedScores[j] /= allScores.Count;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _aggregatedScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeCorrelationScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private double[] ComputeMutualInformationScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        int nBins = 10;

        for (int j = 0; j < p; j++)
        {
            var values = new double[n];
            double minVal = double.MaxValue, maxVal = double.MinValue;

            for (int i = 0; i < n; i++)
            {
                values[i] = NumOps.ToDouble(data[i, j]);
                minVal = Math.Min(minVal, values[i]);
                maxVal = Math.Max(maxVal, values[i]);
            }

            double range = maxVal - minVal;
            var xBins = new int[n];
            for (int i = 0; i < n; i++)
                xBins[i] = range > 1e-10 ? Math.Min((int)((values[i] - minVal) / range * (nBins - 1)), nBins - 1) : 0;

            // Discretize target
            var yVals = new double[n];
            double yMin = double.MaxValue, yMax = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                yVals[i] = NumOps.ToDouble(target[i]);
                yMin = Math.Min(yMin, yVals[i]);
                yMax = Math.Max(yMax, yVals[i]);
            }

            double yRange = yMax - yMin;
            var yBins = new int[n];
            for (int i = 0; i < n; i++)
                yBins[i] = yRange > 1e-10 ? Math.Min((int)((yVals[i] - yMin) / yRange * (nBins - 1)), nBins - 1) : 0;

            // Compute mutual information
            var jointCounts = new Dictionary<(int, int), int>();
            var xCounts = new int[nBins];
            var yCounts = new int[nBins];

            for (int i = 0; i < n; i++)
            {
                var key = (xBins[i], yBins[i]);
                jointCounts[key] = jointCounts.GetValueOrDefault(key) + 1;
                xCounts[xBins[i]]++;
                yCounts[yBins[i]]++;
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

            scores[j] = mi;
        }

        return scores;
    }

    private double[] ComputeFScores(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        // Group by target class
        var classes = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classes.ContainsKey(label))
                classes[label] = new List<int>();
            classes[label].Add(i);
        }

        if (classes.Count < 2)
            return scores;

        for (int j = 0; j < p; j++)
        {
            double grandMean = 0;
            for (int i = 0; i < n; i++)
                grandMean += NumOps.ToDouble(data[i, j]);
            grandMean /= n;

            double ssb = 0, ssw = 0;

            foreach (var kvp in classes)
            {
                double classMean = kvp.Value.Average(i => NumOps.ToDouble(data[i, j]));
                ssb += kvp.Value.Count * Math.Pow(classMean - grandMean, 2);

                foreach (int i in kvp.Value)
                    ssw += Math.Pow(NumOps.ToDouble(data[i, j]) - classMean, 2);
            }

            int dfb = classes.Count - 1;
            int dfw = n - classes.Count;

            scores[j] = (ssw > 1e-10 && dfw > 0) ? (ssb / dfb) / (ssw / dfw) : 0;
        }

        return scores;
    }

    private double[] ComputeVarianceScores(Matrix<T> data, int n, int p)
    {
        var scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += NumOps.ToDouble(data[i, j]);
            mean /= n;

            double variance = 0;
            for (int i = 0; i < n; i++)
                variance += Math.Pow(NumOps.ToDouble(data[i, j]) - mean, 2);
            variance /= n;

            scores[j] = variance;
        }

        return scores;
    }

    private double[] ComputeDistanceCorrelationScores(Matrix<T> data, Vector<T> target, int n, int p, Random random)
    {
        var scores = new double[p];

        // Use a sample for computational efficiency
        int sampleSize = Math.Min(100, n);
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(sampleSize).ToArray();

        for (int j = 0; j < p; j++)
        {
            // Compute distance matrices
            var dx = new double[sampleSize, sampleSize];
            var dy = new double[sampleSize, sampleSize];

            for (int i1 = 0; i1 < sampleSize; i1++)
            {
                for (int i2 = 0; i2 < sampleSize; i2++)
                {
                    dx[i1, i2] = Math.Abs(NumOps.ToDouble(data[sampleIndices[i1], j]) - NumOps.ToDouble(data[sampleIndices[i2], j]));
                    dy[i1, i2] = Math.Abs(NumOps.ToDouble(target[sampleIndices[i1]]) - NumOps.ToDouble(target[sampleIndices[i2]]));
                }
            }

            // Double-center
            DoubleCenterMatrix(dx, sampleSize);
            DoubleCenterMatrix(dy, sampleSize);

            // Compute distance covariance
            double dcov = 0;
            for (int i1 = 0; i1 < sampleSize; i1++)
                for (int i2 = 0; i2 < sampleSize; i2++)
                    dcov += dx[i1, i2] * dy[i1, i2];
            dcov /= sampleSize * sampleSize;

            scores[j] = Math.Sqrt(Math.Max(0, dcov));
        }

        return scores;
    }

    private void DoubleCenterMatrix(double[,] matrix, int n)
    {
        var rowMeans = new double[n];
        var colMeans = new double[n];
        double grandMean = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                rowMeans[i] += matrix[i, j];
                colMeans[j] += matrix[i, j];
            }
            rowMeans[i] /= n;
        }

        for (int j = 0; j < n; j++)
        {
            colMeans[j] /= n;
            grandMean += colMeans[j];
        }
        grandMean /= n;

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i, j] = matrix[i, j] - rowMeans[i] - colMeans[j] + grandMean;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DiverseEnsemble has not been fitted.");

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
        throw new NotSupportedException("DiverseEnsemble does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DiverseEnsemble has not been fitted.");

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
