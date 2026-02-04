using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Randomized Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Randomized Search evaluates random subsets of features to find good
/// combinations. It's faster than exhaustive search and can handle larger
/// feature spaces while still exploring diverse solutions.
/// </para>
/// <para><b>For Beginners:</b> Instead of trying every combination (too slow) or
/// being systematic (might miss good solutions), this method randomly samples
/// feature combinations. It's like randomly picking lottery numbers - given enough
/// tries, you'll likely find something good.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomizedSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double[]? FeatureScores => _featureScores;
    public double BestScore => _bestScore;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RandomizedSearch(
        int nFeaturesToSelect = 10,
        int nIterations = 1000,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RandomizedSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _featureScores = ComputeFeatureScores(data, target, n, p);

        _bestScore = double.MinValue;
        _selectedIndices = null;

        // Track seen combinations to avoid duplicates
        var seenCombinations = new HashSet<string>();

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate random subset
            var subset = GenerateRandomSubset(p, random);

            // Check if already seen
            string key = string.Join(",", subset.OrderBy(x => x));
            if (seenCombinations.Contains(key))
                continue;
            seenCombinations.Add(key);

            // Evaluate subset
            double score = EvaluateSubset(subset);

            if (score > _bestScore)
            {
                _bestScore = score;
                _selectedIndices = subset.OrderBy(x => x).ToArray();
            }
        }

        // Fallback if no good solution found
        if (_selectedIndices is null)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _featureScores[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private int[] GenerateRandomSubset(int p, Random random)
    {
        var allIndices = Enumerable.Range(0, p).ToList();
        var subset = new int[_nFeaturesToSelect];

        for (int i = 0; i < _nFeaturesToSelect; i++)
        {
            int idx = random.Next(allIndices.Count);
            subset[i] = allIndices[idx];
            allIndices.RemoveAt(idx);
        }

        return subset;
    }

    private double EvaluateSubset(int[] subset)
    {
        double score = 0;
        foreach (int j in subset)
            score += _featureScores![j];

        return score / subset.Length;
    }

    private double[] ComputeFeatureScores(Matrix<T> data, Vector<T> target, int n, int p)
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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedSearch has not been fitted.");

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
        throw new NotSupportedException("RandomizedSearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RandomizedSearch has not been fitted.");

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
