using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// MOEA/D Multi-objective Feature Selection using decomposition.
/// </summary>
/// <remarks>
/// <para>
/// Uses the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
/// to optimize accuracy and feature count simultaneously by decomposing the
/// multi-objective problem into scalar subproblems.
/// </para>
/// <para><b>For Beginners:</b> MOEA/D breaks down the hard problem of optimizing
/// multiple goals at once into many simpler single-goal problems. Each simple
/// problem focuses on a different trade-off between accuracy and feature count.
/// </para>
/// </remarks>
public class MOEADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly int _neighborhoodSize;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MOEADSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        int neighborhoodSize = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _neighborhoodSize = neighborhoodSize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MOEADSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Generate weight vectors
        var weights = new double[_populationSize, 2];
        for (int i = 0; i < _populationSize; i++)
        {
            weights[i, 0] = (double)i / (_populationSize - 1);
            weights[i, 1] = 1.0 - weights[i, 0];
        }

        // Compute neighborhoods
        var neighborhoods = ComputeNeighborhoods(weights);

        // Initialize population
        var population = new bool[_populationSize][];
        var objectives = new (double acc, double nFeat)[_populationSize];
        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = new bool[p];
            int numToSelect = rand.Next(1, Math.Min(p, _nFeaturesToSelect) + 1);
            var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect);
            foreach (int idx in indices) population[i][idx] = true;
            objectives[i] = EvaluateObjectives(X, y, population[i], n, p);
        }

        // Reference point
        double zAcc = objectives.Max(o => o.acc);
        double zFeat = objectives.Min(o => o.nFeat);

        // MOEA/D main loop
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Select parents from neighborhood
                int p1 = neighborhoods[i, rand.Next(_neighborhoodSize)];
                int p2 = neighborhoods[i, rand.Next(_neighborhoodSize)];

                // Crossover and mutation
                var child = Crossover(population[p1], population[p2], p, rand);
                Mutate(child, p, rand);

                var childObj = EvaluateObjectives(X, y, child, n, p);

                // Update reference point
                zAcc = Math.Max(zAcc, childObj.acc);
                zFeat = Math.Min(zFeat, childObj.nFeat);

                // Update neighbors
                foreach (int j in Enumerable.Range(0, _neighborhoodSize).Select(k => neighborhoods[i, k]))
                {
                    double childScalar = ComputeTchebycheff(childObj, weights, j, zAcc, zFeat);
                    double currentScalar = ComputeTchebycheff(objectives[j], weights, j, zAcc, zFeat);

                    if (childScalar < currentScalar)
                    {
                        population[j] = (bool[])child.Clone();
                        objectives[j] = childObj;
                    }
                }
            }
        }

        // Track feature importance
        _featureImportances = new double[p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                if (population[i][j])
                    _featureImportances[j] += objectives[i].acc;

        // Select best solution
        int bestIdx = FindBestSolution(objectives, weights, zAcc, zFeat);
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => population[bestIdx][j])
            .Take(_nFeaturesToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private int[,] ComputeNeighborhoods(double[,] weights)
    {
        var neighborhoods = new int[_populationSize, _neighborhoodSize];
        for (int i = 0; i < _populationSize; i++)
        {
            var distances = Enumerable.Range(0, _populationSize)
                .Select(j => (j, Math.Sqrt(Math.Pow(weights[i, 0] - weights[j, 0], 2) + Math.Pow(weights[i, 1] - weights[j, 1], 2))))
                .OrderBy(x => x.Item2)
                .Take(_neighborhoodSize)
                .ToList();

            for (int k = 0; k < _neighborhoodSize; k++)
                neighborhoods[i, k] = distances[k].j;
        }
        return neighborhoods;
    }

    private (double acc, double nFeat) EvaluateObjectives(double[,] X, double[] y, bool[] selected, int n, int p)
    {
        var indices = Enumerable.Range(0, p).Where(j => selected[j]).ToList();
        if (indices.Count == 0) return (0, p);

        double acc = ComputeCorrelationScore(X, y, indices, n);
        return (acc, indices.Count);
    }

    private double ComputeCorrelationScore(double[,] X, double[] y, List<int> features, int n)
    {
        double yMean = y.Average();
        double totalScore = 0;

        foreach (int j in features)
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

        return totalScore / features.Count;
    }

    private double ComputeTchebycheff((double acc, double nFeat) obj, double[,] weights, int idx, double zAcc, double zFeat)
    {
        double d1 = weights[idx, 0] * Math.Abs(zAcc - obj.acc);
        double d2 = weights[idx, 1] * Math.Abs(obj.nFeat - zFeat);
        return Math.Max(d1, d2);
    }

    private bool[] Crossover(bool[] p1, bool[] p2, int p, Random rand)
    {
        var child = new bool[p];
        int point = rand.Next(1, p);
        for (int i = 0; i < p; i++)
            child[i] = i < point ? p1[i] : p2[i];
        if (!child.Any(x => x)) child[rand.Next(p)] = true;
        return child;
    }

    private void Mutate(bool[] individual, int p, Random rand)
    {
        for (int i = 0; i < p; i++)
            if (rand.NextDouble() < 0.1)
                individual[i] = !individual[i];
        if (!individual.Any(x => x)) individual[rand.Next(p)] = true;
    }

    private int FindBestSolution((double acc, double nFeat)[] objectives, double[,] weights, double zAcc, double zFeat)
    {
        int best = 0;
        double bestScore = double.MaxValue;
        for (int i = 0; i < _populationSize; i++)
        {
            double score = ComputeTchebycheff(objectives[i], weights, _populationSize / 2, zAcc, zFeat);
            if (score < bestScore) { bestScore = score; best = i; }
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
            throw new InvalidOperationException("MOEADSelector has not been fitted.");

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
        throw new NotSupportedException("MOEADSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MOEADSelector has not been fitted.");

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
