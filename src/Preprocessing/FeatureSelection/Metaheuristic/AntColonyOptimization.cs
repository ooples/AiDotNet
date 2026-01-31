using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Ant Colony Optimization (ACO) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ACO is inspired by how ants find shortest paths using pheromone trails.
/// Each ant constructs a feature subset, and pheromone levels are updated
/// based on the quality of the solutions found.
/// </para>
/// <para><b>For Beginners:</b> Imagine ants leaving scent trails as they explore.
/// Better paths get more scent, attracting more ants. Similarly, good feature
/// combinations get stronger "pheromone" signals, guiding the search toward
/// optimal feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AntColonyOptimization<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _alpha; // Pheromone importance
    private readonly double _beta;  // Heuristic importance
    private readonly double _rho;   // Evaporation rate
    private readonly int? _randomState;

    private double[]? _pheromones;
    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NAnts => _nAnts;
    public int NIterations => _nIterations;
    public double[]? Pheromones => _pheromones;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AntColonyOptimization(
        int nFeaturesToSelect = 10,
        int nAnts = 20,
        int nIterations = 50,
        double alpha = 1.0,
        double beta = 2.0,
        double rho = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nAnts < 1)
            throw new ArgumentException("Number of ants must be at least 1.", nameof(nAnts));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _alpha = alpha;
        _beta = beta;
        _rho = rho;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonyOptimization requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize pheromone trails
        _pheromones = new double[p];
        for (int j = 0; j < p; j++)
            _pheromones[j] = 1.0;

        // Compute heuristic information (correlation-based)
        _featureScores = ComputeHeuristicInfo(data, target);

        int[]? bestSubset = null;
        double bestFitness = double.MinValue;

        for (int iter = 0; iter < _nIterations; iter++)
        {
            var antSubsets = new List<int[]>();
            var antFitness = new double[_nAnts];

            // Each ant constructs a solution
            for (int ant = 0; ant < _nAnts; ant++)
            {
                var subset = ConstructSolution(p, random);
                antSubsets.Add(subset);
                antFitness[ant] = EvaluateFitness(data, target, subset);

                if (antFitness[ant] > bestFitness)
                {
                    bestFitness = antFitness[ant];
                    bestSubset = (int[])subset.Clone();
                }
            }

            // Update pheromones
            UpdatePheromones(antSubsets, antFitness, p);
        }

        // Use best found subset
        if (bestSubset is not null)
        {
            _selectedIndices = bestSubset.OrderBy(x => x).ToArray();
        }
        else
        {
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
        }

        IsFitted = true;
    }

    private double[] ComputeHeuristicInfo(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
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

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0.01;
        }

        return scores;
    }

    private int[] ConstructSolution(int p, Random random)
    {
        var selected = new List<int>();
        var available = Enumerable.Range(0, p).ToList();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        while (selected.Count < numToSelect && available.Count > 0)
        {
            // Compute selection probabilities
            var probabilities = new double[available.Count];
            double total = 0;

            for (int i = 0; i < available.Count; i++)
            {
                int j = available[i];
                double tau = Math.Pow(_pheromones![j], _alpha);
                double eta = Math.Pow(_featureScores![j], _beta);
                probabilities[i] = tau * eta;
                total += probabilities[i];
            }

            // Normalize and select
            if (total > 0)
            {
                double r = random.NextDouble() * total;
                double cumulative = 0;
                int selectedIdx = available.Count - 1;

                for (int i = 0; i < available.Count; i++)
                {
                    cumulative += probabilities[i];
                    if (r <= cumulative)
                    {
                        selectedIdx = i;
                        break;
                    }
                }

                selected.Add(available[selectedIdx]);
                available.RemoveAt(selectedIdx);
            }
            else
            {
                int idx = random.Next(available.Count);
                selected.Add(available[idx]);
                available.RemoveAt(idx);
            }
        }

        return selected.ToArray();
    }

    private double EvaluateFitness(Matrix<T> data, Vector<T> target, int[] subset)
    {
        if (subset.Length == 0) return 0;

        // Use sum of correlations as fitness
        double fitness = 0;
        foreach (int j in subset)
            fitness += _featureScores![j];

        return fitness / subset.Length;
    }

    private void UpdatePheromones(List<int[]> antSubsets, double[] antFitness, int p)
    {
        // Evaporation
        for (int j = 0; j < p; j++)
            _pheromones![j] *= (1 - _rho);

        // Deposit
        for (int ant = 0; ant < antSubsets.Count; ant++)
        {
            double deposit = antFitness[ant];
            foreach (int j in antSubsets[ant])
                _pheromones![j] += deposit;
        }

        // Ensure minimum pheromone level
        for (int j = 0; j < p; j++)
            if (_pheromones![j] < 0.01) _pheromones[j] = 0.01;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyOptimization has not been fitted.");

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
        throw new NotSupportedException("AntColonyOptimization does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyOptimization has not been fitted.");

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
