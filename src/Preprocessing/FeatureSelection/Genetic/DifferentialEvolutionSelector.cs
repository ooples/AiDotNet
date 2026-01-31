using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Differential Evolution based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses differential evolution algorithm to optimize feature subsets by combining
/// differences between population members to explore the solution space.
/// </para>
/// <para><b>For Beginners:</b> Differential evolution creates new solutions by taking
/// the difference between existing solutions and adding it to another. This helps
/// explore feature combinations more efficiently than random mutation.
/// </para>
/// </remarks>
public class DifferentialEvolutionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationFactor;
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

    public DifferentialEvolutionSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationFactor = 0.8,
        double crossoverRate = 0.7,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _mutationFactor = mutationFactor;
        _crossoverRate = crossoverRate;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialEvolutionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var population = new double[_populationSize, p];
        var fitness = new double[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
                population[i, j] = rand.NextDouble();
            fitness[i] = EvaluateSolution(X, y, population, i, numToSelect, n, p);
        }

        _featureImportances = new double[p];

        // DE iterations
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Select three random distinct individuals
                int a, b, c;
                do { a = rand.Next(_populationSize); } while (a == i);
                do { b = rand.Next(_populationSize); } while (b == i || b == a);
                do { c = rand.Next(_populationSize); } while (c == i || c == a || c == b);

                // Mutation and crossover
                var trial = new double[p];
                int jRand = rand.Next(p);

                for (int j = 0; j < p; j++)
                {
                    if (rand.NextDouble() < _crossoverRate || j == jRand)
                    {
                        trial[j] = population[a, j] + _mutationFactor * (population[b, j] - population[c, j]);
                        trial[j] = Math.Max(0, Math.Min(1, trial[j]));
                    }
                    else
                    {
                        trial[j] = population[i, j];
                    }
                }

                // Selection
                double trialFitness = EvaluateTrialFitness(X, y, trial, numToSelect, n, p);
                if (trialFitness > fitness[i])
                {
                    for (int j = 0; j < p; j++)
                        population[i, j] = trial[j];
                    fitness[i] = trialFitness;
                }

                // Track feature importance
                var mask = GetMask(population, i, numToSelect, p);
                for (int j = 0; j < p; j++)
                {
                    if (mask[j])
                        _featureImportances[j] += fitness[i];
                }
            }
        }

        // Find best solution
        int bestIdx = 0;
        for (int i = 1; i < _populationSize; i++)
        {
            if (fitness[i] > fitness[bestIdx])
                bestIdx = i;
        }

        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => population[bestIdx, j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // Normalize feature importances
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        IsFitted = true;
    }

    private double EvaluateSolution(double[,] X, double[] y, double[,] population, int idx, int numToSelect, int n, int p)
    {
        var mask = GetMask(population, idx, numToSelect, p);
        return EvaluateFitness(X, y, mask, n, p);
    }

    private double EvaluateTrialFitness(double[,] X, double[] y, double[] trial, int numToSelect, int n, int p)
    {
        var sortedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => trial[j])
            .Take(numToSelect)
            .ToHashSet();
        var mask = new bool[p];
        foreach (int idx in sortedIndices)
            mask[idx] = true;
        return EvaluateFitness(X, y, mask, n, p);
    }

    private bool[] GetMask(double[,] population, int idx, int numToSelect, int p)
    {
        var values = new double[p];
        for (int j = 0; j < p; j++)
            values[j] = population[idx, j];

        var sortedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => values[j])
            .Take(numToSelect)
            .ToHashSet();

        var mask = new bool[p];
        foreach (int i in sortedIndices)
            mask[i] = true;
        return mask;
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
            throw new InvalidOperationException("DifferentialEvolutionSelector has not been fitted.");

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
        throw new NotSupportedException("DifferentialEvolutionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEvolutionSelector has not been fitted.");

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
