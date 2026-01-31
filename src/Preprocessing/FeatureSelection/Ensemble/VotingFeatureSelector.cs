using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Voting-based Ensemble Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Voting Feature Selection combines multiple feature selection methods and selects
/// features based on how many methods agree on their importance. Features selected
/// by more methods are considered more reliable.
/// </para>
/// <para><b>For Beginners:</b> Different feature selection methods might disagree on
/// which features are important. Voting combines their opinions - if many methods
/// agree that a feature is useful, it's probably genuinely important. It's like
/// getting a second, third, and fourth opinion.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class VotingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPerMethod;
    private readonly double _minVoteRatio;

    private int[]? _voteCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NPerMethod => _nPerMethod;
    public double MinVoteRatio => _minVoteRatio;
    public int[]? VoteCounts => _voteCounts;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public VotingFeatureSelector(
        int nFeaturesToSelect = 10,
        int nPerMethod = 20,
        double minVoteRatio = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPerMethod < nFeaturesToSelect)
            throw new ArgumentException("Features per method must be at least nFeaturesToSelect.", nameof(nPerMethod));
        if (minVoteRatio < 0 || minVoteRatio > 1)
            throw new ArgumentException("Minimum vote ratio must be between 0 and 1.", nameof(minVoteRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPerMethod = nPerMethod;
        _minVoteRatio = minVoteRatio;
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

        _voteCounts = new int[p];

        // Method 1: Correlation-based
        var correlationScores = ComputeCorrelation(data, target, n, p);
        AddVotes(correlationScores, p);

        // Method 2: Fisher score
        var fisherScores = ComputeFisherScore(data, target, n, p);
        AddVotes(fisherScores, p);

        // Method 3: Variance-based
        var varianceScores = ComputeVariance(data, n, p);
        AddVotes(varianceScores, p);

        // Method 4: Chi-square-like (for binary)
        var chiSquareScores = ComputeChiSquareApprox(data, target, n, p);
        AddVotes(chiSquareScores, p);

        // Select features with sufficient votes
        int nMethods = 4;
        int minVotes = (int)Math.Ceiling(_minVoteRatio * nMethods);

        var candidates = Enumerable.Range(0, p)
            .Where(j => _voteCounts[j] >= minVotes)
            .OrderByDescending(j => _voteCounts[j])
            .ToList();

        // If not enough candidates meet threshold, take top by votes
        if (candidates.Count < _nFeaturesToSelect)
        {
            candidates = Enumerable.Range(0, p)
                .OrderByDescending(j => _voteCounts[j])
                .ToList();
        }

        _selectedIndices = candidates
            .Take(_nFeaturesToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void AddVotes(double[] scores, int p)
    {
        var topIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => scores[j])
            .Take(_nPerMethod)
            .ToHashSet();

        foreach (int j in topIndices)
            _voteCounts![j]++;
    }

    private double[] ComputeCorrelation(Matrix<T> data, Vector<T> target, int n, int p)
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

    private double[] ComputeFisherScore(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        var class0 = new List<int>();
        var class1 = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count < 2 || class1.Count < 2)
            return scores;

        for (int j = 0; j < p; j++)
        {
            double mean0 = class0.Sum(i => NumOps.ToDouble(data[i, j])) / class0.Count;
            double mean1 = class1.Sum(i => NumOps.ToDouble(data[i, j])) / class1.Count;

            double var0 = class0.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean0, 2)) / class0.Count;
            double var1 = class1.Sum(i => Math.Pow(NumOps.ToDouble(data[i, j]) - mean1, 2)) / class1.Count;

            double denom = var0 + var1;
            scores[j] = denom > 1e-10 ? Math.Pow(mean0 - mean1, 2) / denom : 0;
        }

        return scores;
    }

    private double[] ComputeVariance(Matrix<T> data, int n, int p)
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
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean;
                variance += diff * diff;
            }
            scores[j] = variance / n;
        }

        return scores;
    }

    private double[] ComputeChiSquareApprox(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Use median split for continuous features
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            Array.Sort(values);
            double median = values[n / 2];

            // Create 2x2 contingency table
            int a = 0, b = 0, c = 0, d = 0;
            for (int i = 0; i < n; i++)
            {
                bool high = NumOps.ToDouble(data[i, j]) > median;
                bool positive = NumOps.ToDouble(target[i]) >= 0.5;

                if (high && positive) a++;
                else if (high && !positive) b++;
                else if (!high && positive) c++;
                else d++;
            }

            // Chi-square statistic
            double total = a + b + c + d;
            double expected = (a + b) * (a + c) / total;
            if (expected > 0)
            {
                scores[j] = Math.Pow(a - expected, 2) / expected;
            }
        }

        return scores;
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
