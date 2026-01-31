using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// SPEA-II Multi-objective Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Strength Pareto Evolutionary Algorithm II (SPEA-II) which maintains
/// an archive of non-dominated solutions and uses fine-grained fitness assignment.
/// </para>
/// <para><b>For Beginners:</b> SPEA-II keeps track of the best solutions found
/// so far in an "archive." It measures how good a solution is by counting how
/// many other solutions it dominates (is better than in all objectives).
/// </para>
/// </remarks>
public class SPEAIISelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _archiveSize;
    private readonly int _nGenerations;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SPEAIISelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int archiveSize = 25,
        int nGenerations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _archiveSize = archiveSize;
        _nGenerations = nGenerations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SPEAIISelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize population
        var population = new List<bool[]>();
        for (int i = 0; i < _populationSize; i++)
        {
            var individual = new bool[p];
            int numToSelect = rand.Next(1, Math.Min(p, _nFeaturesToSelect) + 1);
            var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect);
            foreach (int idx in indices) individual[idx] = true;
            population.Add(individual);
        }

        var archive = new List<bool[]>();

        // SPEA-II main loop
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            var combined = population.Concat(archive).ToList();
            var objectives = combined.Select(ind => EvaluateObjectives(X, y, ind, n, p)).ToList();

            // Compute fitness
            var fitness = ComputeFitness(objectives);

            // Environmental selection
            archive = EnvironmentalSelection(combined, fitness, objectives);

            // Mating selection and variation
            population = new List<bool[]>();
            while (population.Count < _populationSize)
            {
                int p1 = TournamentSelect(archive, fitness, rand);
                int p2 = TournamentSelect(archive, fitness, rand);
                var child = Crossover(archive[p1], archive[p2], p, rand);
                Mutate(child, p, rand);
                population.Add(child);
            }
        }

        // Track feature importance from archive
        _featureImportances = new double[p];
        var archiveObjectives = archive.Select(ind => EvaluateObjectives(X, y, ind, n, p)).ToList();
        for (int i = 0; i < archive.Count; i++)
            for (int j = 0; j < p; j++)
                if (archive[i][j])
                    _featureImportances[j] += archiveObjectives[i].acc;

        // Select best solution from archive
        int bestIdx = 0;
        double bestScore = double.MinValue;
        for (int i = 0; i < archive.Count; i++)
        {
            var obj = archiveObjectives[i];
            double score = obj.acc - 0.1 * obj.nFeat / p;
            if (score > bestScore) { bestScore = score; bestIdx = i; }
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => archive[bestIdx][j])
            .Take(_nFeaturesToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private (double acc, double nFeat) EvaluateObjectives(double[,] X, double[] y, bool[] selected, int n, int p)
    {
        var indices = Enumerable.Range(0, p).Where(j => selected[j]).ToList();
        if (indices.Count == 0) return (0, p);

        double yMean = y.Average();
        double totalScore = 0;
        foreach (int j in indices)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
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
            totalScore += (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return (totalScore / indices.Count, indices.Count);
    }

    private double[] ComputeFitness(List<(double acc, double nFeat)> objectives)
    {
        int n = objectives.Count;
        var strength = new int[n];
        var rawFitness = new double[n];

        // Compute strength
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j && Dominates(objectives[i], objectives[j]))
                    strength[i]++;

        // Compute raw fitness
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j && Dominates(objectives[j], objectives[i]))
                    rawFitness[i] += strength[j];

        // Add density
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                double d = Math.Sqrt(Math.Pow(objectives[i].acc - objectives[j].acc, 2) +
                                    Math.Pow(objectives[i].nFeat - objectives[j].nFeat, 2));
                distances[i, j] = d;
                distances[j, i] = d;
            }

        var fitness = new double[n];
        int k = (int)Math.Sqrt(n);
        for (int i = 0; i < n; i++)
        {
            var sortedDist = Enumerable.Range(0, n).Where(j => j != i)
                .Select(j => distances[i, j]).OrderBy(d => d).ToList();
            double density = 1.0 / (sortedDist[Math.Min(k, sortedDist.Count - 1)] + 2);
            fitness[i] = rawFitness[i] + density;
        }

        return fitness;
    }

    private bool Dominates((double acc, double nFeat) a, (double acc, double nFeat) b)
    {
        return a.acc >= b.acc && a.nFeat <= b.nFeat && (a.acc > b.acc || a.nFeat < b.nFeat);
    }

    private List<bool[]> EnvironmentalSelection(List<bool[]> combined, double[] fitness, List<(double acc, double nFeat)> objectives)
    {
        var sorted = Enumerable.Range(0, combined.Count).OrderBy(i => fitness[i]).ToList();
        var selected = new List<bool[]>();
        foreach (int i in sorted.Take(_archiveSize))
            selected.Add((bool[])combined[i].Clone());
        return selected;
    }

    private int TournamentSelect(List<bool[]> archive, double[] fitness, Random rand)
    {
        int a = rand.Next(archive.Count);
        int b = rand.Next(archive.Count);
        return fitness[a] < fitness[b] ? a : b;
    }

    private bool[] Crossover(bool[] p1, bool[] p2, int p, Random rand)
    {
        var child = new bool[p];
        for (int i = 0; i < p; i++)
            child[i] = rand.NextDouble() < 0.5 ? p1[i] : p2[i];
        if (!child.Any(x => x)) child[rand.Next(p)] = true;
        return child;
    }

    private void Mutate(bool[] individual, int p, Random rand)
    {
        for (int i = 0; i < p; i++)
            if (rand.NextDouble() < 0.05)
                individual[i] = !individual[i];
        if (!individual.Any(x => x)) individual[rand.Next(p)] = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEAIISelector has not been fitted.");

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
        throw new NotSupportedException("SPEAIISelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEAIISelector has not been fitted.");

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
