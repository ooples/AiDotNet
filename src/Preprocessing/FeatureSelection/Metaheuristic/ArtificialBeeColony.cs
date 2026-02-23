using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Artificial Bee Colony for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Artificial Bee Colony is inspired by the foraging behavior of honey bees.
/// Employed bees exploit known food sources, onlookers choose sources based on
/// quality, and scouts search for new sources when old ones are exhausted.
/// </para>
/// <para><b>For Beginners:</b> Bees work together to find the best flowers.
/// Some bees (employed) keep working on known flowers, others (onlookers) watch
/// and join the bees with the best finds, and scout bees explore randomly for
/// new flowers. This balance of exploitation and exploration helps find optimal
/// feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ArtificialBeeColony<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _colonySize;
    private readonly int _maxIterations;
    private readonly int _limit;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int ColonySize => _colonySize;
    public int MaxIterations => _maxIterations;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ArtificialBeeColony(
        int nFeaturesToSelect = 10,
        int colonySize = 30,
        int maxIterations = 100,
        int limit = 20,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (colonySize < 2)
            throw new ArgumentException("Colony size must be at least 2.", nameof(colonySize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _colonySize = colonySize;
        _maxIterations = maxIterations;
        _limit = limit;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ArtificialBeeColony requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nFood = _colonySize / 2;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize food sources
        var foodSources = new double[nFood, p];
        for (int i = 0; i < nFood; i++)
            for (int j = 0; j < p; j++)
                foodSources[i, j] = random.NextDouble();

        var fitness = new double[nFood];
        var trials = new int[nFood];
        for (int i = 0; i < nFood; i++)
            fitness[i] = EvaluateSolution(data, target, foodSources, i, p, n);

        var bestSolution = new double[p];
        double bestFitness = double.MinValue;

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Employed bees phase
            for (int i = 0; i < nFood; i++)
            {
                var newSolution = new double[p];
                int partner = random.Next(nFood);
                while (partner == i)
                    partner = random.Next(nFood);

                for (int j = 0; j < p; j++)
                {
                    double phi = 2 * random.NextDouble() - 1;
                    newSolution[j] = foodSources[i, j] + phi * (foodSources[i, j] - foodSources[partner, j]);
                    newSolution[j] = Math.Max(0, Math.Min(1, newSolution[j]));
                }

                var tempSources = (double[,])foodSources.Clone();
                for (int j = 0; j < p; j++)
                    tempSources[i, j] = newSolution[j];

                double newFitness = EvaluateSolution(data, target, tempSources, i, p, n);

                if (newFitness > fitness[i])
                {
                    for (int j = 0; j < p; j++)
                        foodSources[i, j] = newSolution[j];
                    fitness[i] = newFitness;
                    trials[i] = 0;
                }
                else
                {
                    trials[i]++;
                }
            }

            // Compute probabilities
            double fitnessSum = fitness.Sum();
            var probabilities = fitness.Select(f => f / (fitnessSum + 1e-10)).ToArray();

            // Onlooker bees phase
            for (int i = 0; i < nFood; i++)
            {
                // Select food source based on probability
                double r = random.NextDouble();
                double cumSum = 0;
                int selected = 0;
                for (int k = 0; k < nFood; k++)
                {
                    cumSum += probabilities[k];
                    if (r <= cumSum)
                    {
                        selected = k;
                        break;
                    }
                }

                var newSolution = new double[p];
                int partner = random.Next(nFood);
                while (partner == selected)
                    partner = random.Next(nFood);

                for (int j = 0; j < p; j++)
                {
                    double phi = 2 * random.NextDouble() - 1;
                    newSolution[j] = foodSources[selected, j] + phi * (foodSources[selected, j] - foodSources[partner, j]);
                    newSolution[j] = Math.Max(0, Math.Min(1, newSolution[j]));
                }

                var tempSources = (double[,])foodSources.Clone();
                for (int j = 0; j < p; j++)
                    tempSources[selected, j] = newSolution[j];

                double newFitness = EvaluateSolution(data, target, tempSources, selected, p, n);

                if (newFitness > fitness[selected])
                {
                    for (int j = 0; j < p; j++)
                        foodSources[selected, j] = newSolution[j];
                    fitness[selected] = newFitness;
                    trials[selected] = 0;
                }
                else
                {
                    trials[selected]++;
                }
            }

            // Scout bees phase
            for (int i = 0; i < nFood; i++)
            {
                if (trials[i] > _limit)
                {
                    for (int j = 0; j < p; j++)
                        foodSources[i, j] = random.NextDouble();
                    fitness[i] = EvaluateSolution(data, target, foodSources, i, p, n);
                    trials[i] = 0;
                }
            }

            // Track best
            for (int i = 0; i < nFood; i++)
            {
                if (fitness[i] > bestFitness)
                {
                    bestFitness = fitness[i];
                    for (int j = 0; j < p; j++)
                        bestSolution[j] = foodSources[i, j];
                }
            }
        }

        _featureScores = bestSolution;

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateSolution(Matrix<T> data, Vector<T> target, double[,] positions, int idx, int p, int n)
    {
        var selectedFeatures = Enumerable.Range(0, p)
            .Where(j => positions[idx, j] > 0.5)
            .ToArray();

        if (selectedFeatures.Length == 0)
            return 0;

        double totalScore = 0;
        foreach (int f in selectedFeatures)
            totalScore += ComputeCorrelation(data, target, f, n);

        int targetCount = Math.Min(_nFeaturesToSelect, p);
        double countPenalty = Math.Abs(selectedFeatures.Length - targetCount);

        return totalScore / selectedFeatures.Length - 0.1 * countPenalty;
    }

    private double ComputeCorrelation(Matrix<T> data, Vector<T> target, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, j]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ArtificialBeeColony has not been fitted.");

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
        throw new NotSupportedException("ArtificialBeeColony does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ArtificialBeeColony has not been fitted.");

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
