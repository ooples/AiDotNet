using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Bandit;

/// <summary>
/// UCB1 (Upper Confidence Bound) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using the UCB1 algorithm from multi-armed bandit theory,
/// balancing exploitation of known good features with exploration of uncertain ones.
/// </para>
/// <para><b>For Beginners:</b> UCB1 treats feature selection like a slot machine problem.
/// Each feature is an "arm" to pull. We want features that are good (high reward)
/// but also want to try less-tested features that might be better. UCB1 picks features
/// with high estimated value plus an uncertainty bonus. As we "pull" features more,
/// the uncertainty decreases, and we converge to truly good features.
/// </para>
/// </remarks>
public class UCB1Selector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _explorationWeight;
    private readonly int? _randomState;

    private double[]? _ucbScores;
    private double[]? _estimatedRewards;
    private int[]? _pullCounts;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public double ExplorationWeight => _explorationWeight;
    public double[]? UCBScores => _ucbScores;
    public double[]? EstimatedRewards => _estimatedRewards;
    public int[]? PullCounts => _pullCounts;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public UCB1Selector(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        double explorationWeight = 2.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));
        if (explorationWeight < 0)
            throw new ArgumentException("Exploration weight must be non-negative.", nameof(explorationWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _explorationWeight = explorationWeight;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "UCB1Selector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize UCB1 state
        _estimatedRewards = new double[p];
        _pullCounts = new int[p];
        var totalRewards = new double[p];

        // Precompute feature-target correlations as potential rewards
        double targetMean = y.Average();
        double targetVar = y.Sum(v => (v - targetMean) * (v - targetMean));
        var baseCorrelations = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double mean = col.Average();
            double cov = 0, varX = 0;
            for (int i = 0; i < n; i++)
            {
                cov += (col[i] - mean) * (y[i] - targetMean);
                varX += (col[i] - mean) * (col[i] - mean);
            }

            double denom = Math.Sqrt(varX * targetVar);
            baseCorrelations[j] = denom > 1e-10 ? Math.Abs(cov / denom) : 0;
        }

        // UCB1 iterations
        int totalPulls = 0;

        // Initial round: pull each arm once
        for (int j = 0; j < p; j++)
        {
            // Simulate pulling arm j: get noisy reward based on correlation
            double noise = (rand.NextDouble() - 0.5) * 0.2;
            double reward = Math.Max(0, Math.Min(1, baseCorrelations[j] + noise));

            totalRewards[j] = reward;
            _pullCounts[j] = 1;
            _estimatedRewards[j] = reward;
            totalPulls++;
        }

        // UCB1 selection loop
        for (int iter = p; iter < _nIterations; iter++)
        {
            // Compute UCB for each arm
            int bestArm = 0;
            double bestUCB = double.MinValue;

            for (int j = 0; j < p; j++)
            {
                double exploitation = _estimatedRewards[j];
                double exploration = Math.Sqrt(_explorationWeight * Math.Log(totalPulls + 1) / _pullCounts[j]);
                double ucb = exploitation + exploration;

                if (ucb > bestUCB)
                {
                    bestUCB = ucb;
                    bestArm = j;
                }
            }

            // Pull the best arm
            double noise = (rand.NextDouble() - 0.5) * 0.2;
            double reward = Math.Max(0, Math.Min(1, baseCorrelations[bestArm] + noise));

            totalRewards[bestArm] += reward;
            _pullCounts[bestArm]++;
            _estimatedRewards[bestArm] = totalRewards[bestArm] / _pullCounts[bestArm];
            totalPulls++;
        }

        // Compute final UCB scores
        _ucbScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            double exploitation = _estimatedRewards[j];
            double exploration = Math.Sqrt(_explorationWeight * Math.Log(totalPulls + 1) / _pullCounts[j]);
            _ucbScores[j] = exploitation + exploration;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _estimatedRewards[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

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
            throw new InvalidOperationException("UCB1Selector has not been fitted.");

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
        throw new NotSupportedException("UCB1Selector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("UCB1Selector has not been fitted.");

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
