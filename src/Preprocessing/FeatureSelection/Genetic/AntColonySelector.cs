using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Ant Colony Optimization based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses ant colony optimization where artificial ants construct solutions by
/// probabilistically selecting features based on pheromone trails.
/// </para>
/// <para><b>For Beginners:</b> Like ants finding food, virtual ants explore
/// feature combinations. Good combinations get "pheromone" deposited, making
/// those features more likely to be selected in future. Over time, pheromones
/// guide ants to good feature subsets.
/// </para>
/// </remarks>
public class AntColonySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _evaporationRate;
    private readonly double _alpha;
    private readonly double _beta;
    private readonly int? _randomState;

    private double[]? _pheromones;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NAnts => _nAnts;
    public int NIterations => _nIterations;
    public double[]? Pheromones => _pheromones;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AntColonySelector(
        int nFeaturesToSelect = 10,
        int nAnts = 20,
        int nIterations = 100,
        double evaporationRate = 0.1,
        double alpha = 1.0,
        double beta = 2.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _evaporationRate = evaporationRate;
        _alpha = alpha;
        _beta = beta;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Compute heuristic information (feature quality)
        var heuristic = new double[p];
        for (int j = 0; j < p; j++)
        {
            double corr = ComputeCorrelation(X, y, j, n);
            heuristic[j] = Math.Abs(corr) + 0.01;
        }

        // Initialize pheromones
        _pheromones = new double[p];
        for (int j = 0; j < p; j++)
            _pheromones[j] = 1.0;

        var bestSolution = new bool[p];
        double bestFitness = double.MinValue;

        // ACO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            var solutions = new bool[_nAnts][];
            var fitness = new double[_nAnts];

            // Each ant constructs a solution
            for (int ant = 0; ant < _nAnts; ant++)
            {
                solutions[ant] = ConstructSolution(rand, _pheromones, heuristic, numToSelect, p);
                fitness[ant] = EvaluateFitness(X, y, solutions[ant], n, p);

                if (fitness[ant] > bestFitness)
                {
                    bestFitness = fitness[ant];
                    Array.Copy(solutions[ant], bestSolution, p);
                }
            }

            // Evaporate pheromones
            for (int j = 0; j < p; j++)
                _pheromones[j] *= (1 - _evaporationRate);

            // Deposit pheromones
            for (int ant = 0; ant < _nAnts; ant++)
            {
                double deposit = fitness[ant];
                for (int j = 0; j < p; j++)
                {
                    if (solutions[ant][j])
                        _pheromones[j] += deposit;
                }
            }

            // Reinforce best solution
            for (int j = 0; j < p; j++)
            {
                if (bestSolution[j])
                    _pheromones[j] += bestFitness;
            }
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => bestSolution[j])
            .OrderBy(x => x)
            .ToArray();

        // Normalize pheromones
        double maxPheromone = _pheromones.Max();
        if (maxPheromone > 0)
        {
            for (int j = 0; j < p; j++)
                _pheromones[j] /= maxPheromone;
        }

        IsFitted = true;
    }

    private bool[] ConstructSolution(Random rand, double[] pheromones, double[] heuristic, int numToSelect, int p)
    {
        var solution = new bool[p];
        var available = new HashSet<int>(Enumerable.Range(0, p));

        for (int k = 0; k < numToSelect && available.Count > 0; k++)
        {
            // Compute probabilities
            double totalProb = 0;
            foreach (int j in available)
            {
                double prob = Math.Pow(pheromones[j], _alpha) * Math.Pow(heuristic[j], _beta);
                totalProb += prob;
            }

            // Roulette wheel selection
            double r = rand.NextDouble() * totalProb;
            double cumulative = 0;
            int selected = available.First();

            foreach (int j in available)
            {
                double prob = Math.Pow(pheromones[j], _alpha) * Math.Pow(heuristic[j], _beta);
                cumulative += prob;
                if (cumulative >= r)
                {
                    selected = j;
                    break;
                }
            }

            solution[selected] = true;
            available.Remove(selected);
        }

        return solution;
    }

    private double EvaluateFitness(double[,] X, double[] y, bool[] mask, int n, int p)
    {
        var selectedFeatures = Enumerable.Range(0, p).Where(j => mask[j]).ToList();
        if (selectedFeatures.Count == 0) return 0;

        double totalCorr = 0;
        foreach (int j in selectedFeatures)
        {
            double corr = ComputeCorrelation(X, y, j, n);
            totalCorr += Math.Abs(corr);
        }

        return totalCorr / selectedFeatures.Count;
    }

    private double ComputeCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += X[i, j];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonySelector has not been fitted.");

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
        throw new NotSupportedException("AntColonySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonySelector has not been fitted.");

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
