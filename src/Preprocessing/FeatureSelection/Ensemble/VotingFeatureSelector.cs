using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Voting-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines multiple feature selection methods using voting. Each method casts
/// votes for its top features, and features with the most votes are selected.
/// </para>
/// <para><b>For Beginners:</b> Instead of trusting just one method to pick
/// features, this approach asks multiple methods to vote. Features that are
/// chosen by many different methods are more likely to be truly important,
/// giving you more confident selections.
/// </para>
/// </remarks>
public class VotingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _votesPerMethod;

    private int[]? _voteCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int[]? VoteCounts => _voteCounts;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VotingFeatureSelector(
        int nFeaturesToSelect = 10,
        int votesPerMethod = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _votesPerMethod = votesPerMethod;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "VotingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _voteCounts = new int[p];
        int votesPerMethod = Math.Min(_votesPerMethod, p);

        // Method 1: Correlation
        var corrScores = ComputeCorrelation(X, y, n, p);
        foreach (int j in TopK(corrScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 2: Variance
        var varScores = ComputeVariance(X, n, p);
        foreach (int j in TopK(varScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 3: Fisher score
        var fisherScores = ComputeFisherScore(X, y, n, p);
        foreach (int j in TopK(fisherScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 4: Chi-squared approximation
        var chiScores = ComputeChiSquared(X, y, n, p);
        foreach (int j in TopK(chiScores, votesPerMethod))
            _voteCounts[j]++;

        // Method 5: Gini importance
        var giniScores = ComputeGiniImportance(X, y, n, p);
        foreach (int j in TopK(giniScores, votesPerMethod))
            _voteCounts[j]++;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _voteCounts[j])
            .ThenByDescending(j => corrScores[j]) // Tiebreaker
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[] TopK(double[] scores, int k)
    {
        return Enumerable.Range(0, scores.Length)
            .OrderByDescending(j => scores[j])
            .Take(k)
            .ToArray();
    }

    private double[] ComputeCorrelation(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }
            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }
        return scores;
    }

    private double[] ComputeVariance(double[,] X, int n, int p)
    {
        var scores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += X[i, j];
            mean /= n;
            for (int i = 0; i < n; i++)
                scores[j] += (X[i, j] - mean) * (X[i, j] - mean);
            scores[j] /= (n - 1);
        }
        return scores;
    }

    private double[] ComputeFisherScore(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        var classes = y.Distinct().ToList();
        int k = classes.Count;

        for (int j = 0; j < p; j++)
        {
            double overallMean = 0;
            for (int i = 0; i < n; i++) overallMean += X[i, j];
            overallMean /= n;

            double betweenVar = 0, withinVar = 0;
            foreach (var c in classes)
            {
                var indices = Enumerable.Range(0, n).Where(i => Math.Abs(y[i] - c) < 1e-10).ToList();
                if (indices.Count == 0) continue;

                double classMean = indices.Sum(i => X[i, j]) / indices.Count;
                betweenVar += indices.Count * (classMean - overallMean) * (classMean - overallMean);

                foreach (int i in indices)
                    withinVar += (X[i, j] - classMean) * (X[i, j] - classMean);
            }

            scores[j] = withinVar > 1e-10 ? betweenVar / withinVar : 0;
        }
        return scores;
    }

    private double[] ComputeChiSquared(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        int nBins = Math.Min(10, (int)Math.Sqrt(n) + 1);
        var classes = y.Distinct().ToList();

        for (int j = 0; j < p; j++)
        {
            double xMin = double.MaxValue, xMax = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                xMin = Math.Min(xMin, X[i, j]);
                xMax = Math.Max(xMax, X[i, j]);
            }
            double binWidth = (xMax - xMin) / nBins + 1e-10;

            var observed = new int[nBins, classes.Count];
            var rowSums = new int[nBins];
            var colSums = new int[classes.Count];

            for (int i = 0; i < n; i++)
            {
                int bin = Math.Min((int)((X[i, j] - xMin) / binWidth), nBins - 1);
                int classIdx = classes.IndexOf(y[i]);
                observed[bin, classIdx]++;
                rowSums[bin]++;
                colSums[classIdx]++;
            }

            double chi2 = 0;
            for (int b = 0; b < nBins; b++)
            {
                for (int c = 0; c < classes.Count; c++)
                {
                    double expected = (double)rowSums[b] * colSums[c] / n;
                    if (expected > 0)
                        chi2 += (observed[b, c] - expected) * (observed[b, c] - expected) / expected;
                }
            }
            scores[j] = chi2;
        }
        return scores;
    }

    private double[] ComputeGiniImportance(double[,] X, double[] y, int n, int p)
    {
        var scores = new double[p];
        var classes = y.Distinct().ToList();

        double giniParent = ComputeGini(y, classes, n);

        for (int j = 0; j < p; j++)
        {
            double threshold = 0;
            for (int i = 0; i < n; i++) threshold += X[i, j];
            threshold /= n;

            var leftY = new List<double>();
            var rightY = new List<double>();
            for (int i = 0; i < n; i++)
            {
                if (X[i, j] <= threshold)
                    leftY.Add(y[i]);
                else
                    rightY.Add(y[i]);
            }

            if (leftY.Count > 0 && rightY.Count > 0)
            {
                double giniLeft = ComputeGini(leftY.ToArray(), classes, leftY.Count);
                double giniRight = ComputeGini(rightY.ToArray(), classes, rightY.Count);
                double weightedGini = (double)leftY.Count / n * giniLeft + (double)rightY.Count / n * giniRight;
                scores[j] = giniParent - weightedGini;
            }
        }
        return scores;
    }

    private double ComputeGini(double[] y, List<double> classes, int n)
    {
        double gini = 1.0;
        foreach (var c in classes)
        {
            double p = (double)y.Count(yi => Math.Abs(yi - c) < 1e-10) / n;
            gini -= p * p;
        }
        return gini;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VotingFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("VotingFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("VotingFeatureSelector has not been fitted.");

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
