using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Genetic Algorithm based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a genetic algorithm to evolve feature subsets, optimizing for
/// predictive performance through selection, crossover, and mutation.
/// </para>
/// <para><b>For Beginners:</b> Like natural evolution, this creates a population
/// of different feature combinations, keeps the best ones, combines them to make
/// new combinations, and occasionally mutates them. Over generations, it finds
/// good feature subsets.
/// </para>
/// </remarks>
public class GeneticAlgorithmSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationRate;
    private readonly double _crossoverRate;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NGenerations => _nGenerations;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GeneticAlgorithmSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationRate = 0.1,
        double crossoverRate = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _mutationRate = mutationRate;
        _crossoverRate = crossoverRate;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GeneticAlgorithmSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population
        var population = new bool[_populationSize][];
        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = new bool[p];
            var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToList();
            foreach (int idx in indices)
                population[i][idx] = true;
        }

        _featureImportances = new double[p];

        // Evolution
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            // Evaluate fitness
            var fitness = new double[_populationSize];
            for (int i = 0; i < _populationSize; i++)
                fitness[i] = EvaluateFitness(X, y, population[i], n, p);

            // Track feature importance
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    if (population[i][j])
                        _featureImportances[j] += fitness[i];
                }
            }

            // Selection (tournament)
            var newPopulation = new bool[_populationSize][];
            for (int i = 0; i < _populationSize; i++)
            {
                int idx1 = rand.Next(_populationSize);
                int idx2 = rand.Next(_populationSize);
                newPopulation[i] = (bool[])population[fitness[idx1] > fitness[idx2] ? idx1 : idx2].Clone();
            }

            // Crossover
            for (int i = 0; i < _populationSize - 1; i += 2)
            {
                if (rand.NextDouble() < _crossoverRate)
                {
                    int crossPoint = rand.Next(p);
                    for (int j = crossPoint; j < p; j++)
                        (newPopulation[i][j], newPopulation[i + 1][j]) = (newPopulation[i + 1][j], newPopulation[i][j]);
                }
            }

            // Mutation
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    if (rand.NextDouble() < _mutationRate)
                        newPopulation[i][j] = !newPopulation[i][j];
                }
            }

            population = newPopulation;
        }

        // Normalize feature importances
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureImportances[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateFitness(double[,] X, double[] y, bool[] mask, int n, int p)
    {
        var selectedFeatures = Enumerable.Range(0, p).Where(j => mask[j]).ToList();
        if (selectedFeatures.Count == 0) return 0;

        // Use correlation-based fitness
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
            throw new InvalidOperationException("GeneticAlgorithmSelector has not been fitted.");

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
        throw new NotSupportedException("GeneticAlgorithmSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GeneticAlgorithmSelector has not been fitted.");

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
