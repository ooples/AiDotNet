using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Ant Colony Optimization for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Ant Colony Optimization (ACO) is a swarm intelligence algorithm inspired by how ants
/// find optimal paths using pheromone trails. In feature selection, ants construct feature
/// subsets, and pheromone is deposited on good feature combinations.
/// </para>
/// <para><b>For Beginners:</b> Imagine ants searching for food. When an ant finds a good path,
/// it leaves a scent (pheromone) that attracts other ants. Over time, better paths get
/// stronger scents. Here, "paths" are feature subsets, and "food quality" is model performance.
/// Features that are part of good subsets accumulate more pheromone and are more likely to be selected.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AntColonyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _alpha; // Pheromone importance
    private readonly double _beta; // Heuristic importance
    private readonly double _evaporationRate;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private double[]? _pheromones;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public double[]? Pheromones => _pheromones;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AntColonyFS(
        int nFeaturesToSelect = 10,
        int nAnts = 30,
        int nIterations = 50,
        double alpha = 1.0,
        double beta = 2.0,
        double evaporationRate = 0.1,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _alpha = alpha;
        _beta = beta;
        _evaporationRate = evaporationRate;
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonyFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var scorer = _scorer ?? DefaultScorer;

        // Compute initial heuristic (correlation-based)
        var heuristic = ComputeHeuristic(data, target, n, p);

        // Initialize pheromones
        _pheromones = new double[p];
        for (int j = 0; j < p; j++)
            _pheromones[j] = 1.0;

        int[] bestSolution = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
        double bestFitness = double.NegativeInfinity;

        // ACO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            var antSolutions = new List<int[]>();
            var antFitnesses = new double[_nAnts];

            // Each ant constructs a solution
            for (int ant = 0; ant < _nAnts; ant++)
            {
                var solution = ConstructSolution(p, heuristic, random);
                antSolutions.Add(solution);
                antFitnesses[ant] = scorer(data, target, solution);

                if (antFitnesses[ant] > bestFitness)
                {
                    bestFitness = antFitnesses[ant];
                    bestSolution = (int[])solution.Clone();
                }
            }

            // Update pheromones
            // Evaporation
            for (int j = 0; j < p; j++)
                _pheromones[j] *= (1 - _evaporationRate);

            // Deposit
            for (int ant = 0; ant < _nAnts; ant++)
            {
                double deposit = antFitnesses[ant] > 0 ? antFitnesses[ant] : 0.01;
                foreach (int j in antSolutions[ant])
                    _pheromones[j] += deposit;
            }

            // Elitist strategy: extra deposit for best solution
            foreach (int j in bestSolution)
                _pheromones[j] += bestFitness > 0 ? bestFitness * 2 : 0.02;
        }

        _featureScores = (double[])_pheromones.Clone();
        _selectedIndices = bestSolution.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private int[] ConstructSolution(int p, double[] heuristic, Random random)
    {
        var selected = new HashSet<int>();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        while (selected.Count < numToSelect)
        {
            // Compute selection probabilities
            var probabilities = new double[p];
            double sum = 0;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j))
                    continue;

                double tau = Math.Pow(_pheromones![j], _alpha);
                double eta = Math.Pow(heuristic[j], _beta);
                probabilities[j] = tau * eta;
                sum += probabilities[j];
            }

            if (sum < 1e-10)
            {
                // Random selection if all probabilities are zero
                var remaining = Enumerable.Range(0, p).Where(j => !selected.Contains(j)).ToList();
                if (remaining.Count > 0)
                    selected.Add(remaining[random.Next(remaining.Count)]);
                else
                    break;
                continue;
            }

            // Roulette wheel selection
            double threshold = random.NextDouble() * sum;
            double cumulative = 0;

            for (int j = 0; j < p; j++)
            {
                if (selected.Contains(j))
                    continue;

                cumulative += probabilities[j];
                if (cumulative >= threshold)
                {
                    selected.Add(j);
                    break;
                }
            }
        }

        return selected.ToArray();
    }

    private double[] ComputeHeuristic(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var heuristic = new double[p];

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

            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            heuristic[j] = corr + 0.01; // Add small value to avoid zero
        }

        return heuristic;
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

        // Simple correlation-based score
        int n = data.Rows;
        double totalCorr = 0;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        foreach (int j in featureIndices)
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

            double corr = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            totalCorr += Math.Abs(corr);
        }

        return totalCorr / featureIndices.Length;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyFS has not been fitted.");

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
        throw new NotSupportedException("AntColonyFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyFS has not been fitted.");

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
