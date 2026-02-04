using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// NSGA-II Multi-objective Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize
/// multiple objectives simultaneously: maximizing prediction accuracy while
/// minimizing the number of selected features.
/// </para>
/// <para><b>For Beginners:</b> Sometimes we want both good accuracy AND few
/// features. NSGA-II finds multiple solutions that trade off between these goals.
/// Some solutions have high accuracy but many features; others have fewer
/// features but slightly lower accuracy. You can then choose the best trade-off.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class NSGAIISelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxFeatures;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _crossoverRate;
    private readonly double _mutationRate;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public int PopulationSize => _populationSize;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public NSGAIISelector(
        int maxFeatures = 20,
        int populationSize = 50,
        int nGenerations = 100,
        double crossoverRate = 0.9,
        double mutationRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));

        _maxFeatures = maxFeatures;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _crossoverRate = crossoverRate;
        _mutationRate = mutationRate;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NSGAIISelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();

        // Convert to arrays
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize population
        var population = new bool[_populationSize][];
        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = new bool[p];
            int numSelected = rand.Next(1, Math.Min(p, _maxFeatures) + 1);
            var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numSelected).ToList();
            foreach (int idx in indices)
                population[i][idx] = true;
        }

        // NSGA-II main loop
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            // Evaluate objectives
            var objectives = new (double accuracy, double nFeatures)[_populationSize];
            for (int i = 0; i < _populationSize; i++)
                objectives[i] = EvaluateObjectives(X, y, population[i], n, p);

            // Non-dominated sorting
            var fronts = NonDominatedSort(objectives);

            // Calculate crowding distance
            var crowdingDistances = CalculateCrowdingDistance(objectives, fronts);

            // Create offspring
            var offspring = new bool[_populationSize][];
            for (int i = 0; i < _populationSize; i += 2)
            {
                // Tournament selection
                int parent1 = TournamentSelect(fronts, crowdingDistances, rand);
                int parent2 = TournamentSelect(fronts, crowdingDistances, rand);

                // Crossover
                var (child1, child2) = Crossover(population[parent1], population[parent2], p, rand);

                // Mutation
                Mutate(child1, p, rand);
                Mutate(child2, p, rand);

                offspring[i] = child1;
                if (i + 1 < _populationSize)
                    offspring[i + 1] = child2;
            }

            // Combine parent and offspring
            var combined = population.Concat(offspring).ToArray();
            var combinedObjectives = new (double accuracy, double nFeatures)[combined.Length];
            for (int i = 0; i < combined.Length; i++)
                combinedObjectives[i] = EvaluateObjectives(X, y, combined[i], n, p);

            // Select next generation using NSGA-II selection
            var nextGen = SelectNextGeneration(combined, combinedObjectives, _populationSize);
            population = nextGen;
        }

        // Select best solution (knee point or best accuracy with acceptable features)
        var finalObjectives = new (double accuracy, double nFeatures)[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            finalObjectives[i] = EvaluateObjectives(X, y, population[i], n, p);

        int bestIdx = FindKneePoint(finalObjectives);
        var bestSolution = population[bestIdx];

        // Track feature importance by selection frequency
        _featureImportances = new double[p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                if (population[i][j])
                    _featureImportances[j] += finalObjectives[i].accuracy;

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => bestSolution[j])
            .OrderBy(x => x)
            .ToArray();

        // Limit to max features
        if (_selectedIndices.Length > _maxFeatures)
        {
            _selectedIndices = _selectedIndices
                .OrderByDescending(j => _featureImportances[j])
                .Take(_maxFeatures)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private (double accuracy, double nFeatures) EvaluateObjectives(double[,] X, double[] y, bool[] selected, int n, int p)
    {
        var indices = Enumerable.Range(0, p).Where(j => selected[j]).ToList();
        if (indices.Count == 0)
            return (0, 0);

        double accuracy = ComputeCorrelationScore(X, y, indices, n);
        return (accuracy, indices.Count);
    }

    private double ComputeCorrelationScore(double[,] X, double[] y, List<int> features, int n)
    {
        double yMean = y.Average();
        double totalScore = 0;

        foreach (int j in features)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = X[i, j] - xMean;
                double yDiff = y[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            totalScore += corr;
        }

        return totalScore / features.Count;
    }

    private List<List<int>> NonDominatedSort((double accuracy, double nFeatures)[] objectives)
    {
        int n = objectives.Length;
        var dominated = new List<int>[n];
        var dominationCount = new int[n];
        var fronts = new List<List<int>> { new List<int>() };

        for (int i = 0; i < n; i++)
        {
            dominated[i] = new List<int>();
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                if (Dominates(objectives[i], objectives[j]))
                    dominated[i].Add(j);
                else if (Dominates(objectives[j], objectives[i]))
                    dominationCount[i]++;
            }
            if (dominationCount[i] == 0)
                fronts[0].Add(i);
        }

        int frontIdx = 0;
        while (fronts[frontIdx].Count > 0)
        {
            var nextFront = new List<int>();
            foreach (int i in fronts[frontIdx])
            {
                foreach (int j in dominated[i])
                {
                    dominationCount[j]--;
                    if (dominationCount[j] == 0)
                        nextFront.Add(j);
                }
            }
            fronts.Add(nextFront);
            frontIdx++;
        }

        return fronts;
    }

    private bool Dominates((double accuracy, double nFeatures) a, (double accuracy, double nFeatures) b)
    {
        return a.accuracy >= b.accuracy && a.nFeatures <= b.nFeatures &&
               (a.accuracy > b.accuracy || a.nFeatures < b.nFeatures);
    }

    private double[] CalculateCrowdingDistance((double accuracy, double nFeatures)[] objectives, List<List<int>> fronts)
    {
        var distances = new double[objectives.Length];
        foreach (var front in fronts)
        {
            if (front.Count == 0) continue;

            // Sort by accuracy
            var sorted = front.OrderBy(i => objectives[i].accuracy).ToList();
            distances[sorted[0]] = double.MaxValue;
            distances[sorted[sorted.Count - 1]] = double.MaxValue;

            double accRange = objectives[sorted[sorted.Count - 1]].accuracy - objectives[sorted[0]].accuracy;
            for (int i = 1; i < sorted.Count - 1; i++)
            {
                if (accRange > 1e-10)
                    distances[sorted[i]] += (objectives[sorted[i + 1]].accuracy - objectives[sorted[i - 1]].accuracy) / accRange;
            }

            // Sort by nFeatures
            sorted = front.OrderBy(i => objectives[i].nFeatures).ToList();
            double nFeatRange = objectives[sorted[sorted.Count - 1]].nFeatures - objectives[sorted[0]].nFeatures;
            for (int i = 1; i < sorted.Count - 1; i++)
            {
                if (nFeatRange > 1e-10)
                    distances[sorted[i]] += (objectives[sorted[i + 1]].nFeatures - objectives[sorted[i - 1]].nFeatures) / nFeatRange;
            }
        }
        return distances;
    }

    private int TournamentSelect(List<List<int>> fronts, double[] crowdingDistances, Random rand)
    {
        int a = rand.Next(fronts.Sum(f => f.Count));
        int b = rand.Next(fronts.Sum(f => f.Count));

        int frontA = GetFrontIndex(a, fronts);
        int frontB = GetFrontIndex(b, fronts);

        if (frontA < frontB) return a;
        if (frontB < frontA) return b;
        return crowdingDistances[a] > crowdingDistances[b] ? a : b;
    }

    private int GetFrontIndex(int individual, List<List<int>> fronts)
    {
        for (int i = 0; i < fronts.Count; i++)
            if (fronts[i].Contains(individual))
                return i;
        return fronts.Count;
    }

    private (bool[], bool[]) Crossover(bool[] parent1, bool[] parent2, int p, Random rand)
    {
        var child1 = new bool[p];
        var child2 = new bool[p];

        if (rand.NextDouble() < _crossoverRate)
        {
            int point = rand.Next(1, p);
            for (int i = 0; i < p; i++)
            {
                child1[i] = i < point ? parent1[i] : parent2[i];
                child2[i] = i < point ? parent2[i] : parent1[i];
            }
        }
        else
        {
            Array.Copy(parent1, child1, p);
            Array.Copy(parent2, child2, p);
        }

        return (child1, child2);
    }

    private void Mutate(bool[] individual, int p, Random rand)
    {
        for (int i = 0; i < p; i++)
            if (rand.NextDouble() < _mutationRate)
                individual[i] = !individual[i];

        // Ensure at least one feature
        if (!individual.Any(x => x))
            individual[rand.Next(p)] = true;
    }

    private bool[][] SelectNextGeneration(bool[][] combined, (double accuracy, double nFeatures)[] objectives, int size)
    {
        var fronts = NonDominatedSort(objectives);
        var crowding = CalculateCrowdingDistance(objectives, fronts);

        var selected = new List<bool[]>();
        int frontIdx = 0;

        while (selected.Count + fronts[frontIdx].Count <= size && frontIdx < fronts.Count)
        {
            foreach (int i in fronts[frontIdx])
                selected.Add(combined[i]);
            frontIdx++;
        }

        if (selected.Count < size && frontIdx < fronts.Count)
        {
            var remaining = fronts[frontIdx]
                .OrderByDescending(i => crowding[i])
                .Take(size - selected.Count);
            foreach (int i in remaining)
                selected.Add(combined[i]);
        }

        return selected.ToArray();
    }

    private int FindKneePoint((double accuracy, double nFeatures)[] objectives)
    {
        // Find solution with best trade-off (closest to ideal)
        double maxAcc = objectives.Max(o => o.accuracy);
        double minFeat = objectives.Min(o => o.nFeatures);

        int best = 0;
        double bestDist = double.MaxValue;
        for (int i = 0; i < objectives.Length; i++)
        {
            double normAcc = maxAcc > 0 ? objectives[i].accuracy / maxAcc : 0;
            double normFeat = minFeat > 0 ? minFeat / objectives[i].nFeatures : 0;
            double dist = Math.Sqrt((1 - normAcc) * (1 - normAcc) + (1 - normFeat) * (1 - normFeat));
            if (dist < bestDist)
            {
                bestDist = dist;
                best = i;
            }
        }
        return best;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NSGAIISelector has not been fitted.");

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
        throw new NotSupportedException("NSGAIISelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NSGAIISelector has not been fitted.");

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
