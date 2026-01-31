using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Differential Evolution for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Differential Evolution (DE) is an evolutionary optimization algorithm that uses
/// mutation based on the difference between population members. For feature selection,
/// real-valued vectors are converted to binary feature masks using thresholding.
/// </para>
/// <para><b>For Beginners:</b> DE evolves a population of candidate solutions. To create
/// new solutions, it takes differences between existing ones and adds them to others.
/// This creates mutations that explore the search space efficiently. For feature selection,
/// these continuous values are converted to yes/no decisions about each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DifferentialEvolutionFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationFactor; // F
    private readonly double _crossoverRate; // CR
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MutationFactor => _mutationFactor;
    public double CrossoverRate => _crossoverRate;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DifferentialEvolutionFS(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationFactor = 0.8,
        double crossoverRate = 0.7,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
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
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialEvolutionFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scorer = _scorer ?? DefaultScorer;

        // Initialize population with random real-valued vectors in [0, 1]
        var population = new double[_populationSize, p];
        var fitness = new double[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
                population[i, j] = random.NextDouble();
            fitness[i] = EvaluateIndividual(data, target, population, i, p, scorer);
        }

        int bestIdx = 0;
        double bestFitness = fitness[0];
        for (int i = 1; i < _populationSize; i++)
        {
            if (fitness[i] > bestFitness)
            {
                bestFitness = fitness[i];
                bestIdx = i;
            }
        }

        // DE evolution
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Select three random individuals different from i
                var candidates = Enumerable.Range(0, _populationSize)
                    .Where(x => x != i)
                    .OrderBy(_ => random.Next())
                    .Take(3)
                    .ToArray();

                int r1 = candidates[0];
                int r2 = candidates[1];
                int r3 = candidates[2];

                // Mutation: v = r1 + F * (r2 - r3)
                var mutant = new double[p];
                for (int j = 0; j < p; j++)
                {
                    mutant[j] = population[r1, j] + _mutationFactor * (population[r2, j] - population[r3, j]);
                    mutant[j] = Math.Max(0, Math.Min(1, mutant[j])); // Clip to [0, 1]
                }

                // Crossover
                var trial = new double[p];
                int jRand = random.Next(p); // Ensure at least one dimension from mutant
                for (int j = 0; j < p; j++)
                {
                    if (random.NextDouble() < _crossoverRate || j == jRand)
                        trial[j] = mutant[j];
                    else
                        trial[j] = population[i, j];
                }

                // Selection
                double trialFitness = EvaluateVector(data, target, trial, p, scorer);
                if (trialFitness >= fitness[i])
                {
                    for (int j = 0; j < p; j++)
                        population[i, j] = trial[j];
                    fitness[i] = trialFitness;

                    if (trialFitness > bestFitness)
                    {
                        bestFitness = trialFitness;
                        bestIdx = i;
                    }
                }
            }
        }

        // Extract best solution
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = population[bestIdx, j];

        // Convert to binary selection
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateIndividual(Matrix<T> data, Vector<T> target, double[,] population,
        int individualIdx, int p, Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        var vector = new double[p];
        for (int j = 0; j < p; j++)
            vector[j] = population[individualIdx, j];
        return EvaluateVector(data, target, vector, p, scorer);
    }

    private double EvaluateVector(Matrix<T> data, Vector<T> target, double[] vector,
        int p, Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        // Select features with value > 0.5 (or top k if fewer)
        var selected = vector
            .Select((v, idx) => (Value: v, Index: idx))
            .Where(x => x.Value > 0.5)
            .Select(x => x.Index)
            .ToArray();

        // If too few, add the highest scoring ones
        if (selected.Length == 0)
        {
            selected = vector
                .Select((v, idx) => (Value: v, Index: idx))
                .OrderByDescending(x => x.Value)
                .Take(Math.Min(_nFeaturesToSelect, p))
                .Select(x => x.Index)
                .ToArray();
        }

        return scorer(data, target, selected);
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

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
            throw new InvalidOperationException("DifferentialEvolutionFS has not been fitted.");

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
        throw new NotSupportedException("DifferentialEvolutionFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEvolutionFS has not been fitted.");

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
