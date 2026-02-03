using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Dragonfly Algorithm (DA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Dragonfly Algorithm is inspired by the static and dynamic swarming behaviors
/// of dragonflies. It uses five main principles: separation, alignment, cohesion,
/// attraction towards food, and distraction from enemies.
/// </para>
/// <para><b>For Beginners:</b> This algorithm mimics how dragonflies swarm. They
/// avoid collisions (separation), fly in the same direction (alignment), stay close
/// to the group (cohesion), move toward food (good solutions), and flee from enemies
/// (bad solutions). This balance of exploration and exploitation finds good features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DragonflyAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NIterations => _nIterations;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DragonflyAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 5)
            throw new ArgumentException("Population size must be at least 5.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DragonflyAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _featureScores = ComputeFeatureScores(data, target);

        // Initialize dragonfly positions and velocities
        var positions = new double[_populationSize, p];
        var velocities = new double[_populationSize, p];

        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
            {
                positions[i, j] = random.NextDouble();
                velocities[i, j] = (random.NextDouble() - 0.5) * 0.1;
            }
        }

        var fitness = new double[_populationSize];
        int foodIdx = 0, enemyIdx = 0;
        double foodFitness = double.MinValue, enemyFitness = double.MaxValue;

        // Evaluate initial population
        for (int i = 0; i < _populationSize; i++)
        {
            fitness[i] = EvaluateFitness(positions, i, p);
            if (fitness[i] > foodFitness)
            {
                foodFitness = fitness[i];
                foodIdx = i;
            }
            if (fitness[i] < enemyFitness)
            {
                enemyFitness = fitness[i];
                enemyIdx = i;
            }
        }

        // Main loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            double w = 0.9 - iter * (0.9 - 0.4) / _nIterations; // Inertia weight

            for (int i = 0; i < _populationSize; i++)
            {
                // Calculate swarm behaviors
                var separation = CalculateSeparation(positions, i, p);
                var alignment = CalculateAlignment(velocities, i, p);
                var cohesion = CalculateCohesion(positions, i, p);

                // Calculate attraction to food and distraction from enemy
                var foodAttraction = new double[p];
                var enemyDistraction = new double[p];

                for (int j = 0; j < p; j++)
                {
                    foodAttraction[j] = positions[foodIdx, j] - positions[i, j];
                    enemyDistraction[j] = positions[enemyIdx, j] + positions[i, j];
                }

                // Update velocity and position
                double s = 2 * random.NextDouble() * (1 - (double)iter / _nIterations);
                double a = 2 * random.NextDouble() * (1 - (double)iter / _nIterations);
                double c = 2 * random.NextDouble() * (1 - (double)iter / _nIterations);
                double f = 2 * random.NextDouble();
                double e = 0.1 * random.NextDouble();

                for (int j = 0; j < p; j++)
                {
                    velocities[i, j] = w * velocities[i, j] +
                        s * separation[j] +
                        a * alignment[j] +
                        c * cohesion[j] +
                        f * foodAttraction[j] -
                        e * enemyDistraction[j];

                    // Clamp velocity
                    velocities[i, j] = Math.Max(-1, Math.Min(1, velocities[i, j]));

                    // Update position
                    positions[i, j] += velocities[i, j];
                    positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j]));
                }

                // Evaluate fitness
                fitness[i] = EvaluateFitness(positions, i, p);

                // Update food and enemy
                if (fitness[i] > foodFitness)
                {
                    foodFitness = fitness[i];
                    foodIdx = i;
                }
                if (fitness[i] < enemyFitness)
                {
                    enemyFitness = fitness[i];
                    enemyIdx = i;
                }
            }
        }

        // Extract selected features from best position
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => positions[foodIdx, j] * _featureScores[j])
            .Take(_nFeaturesToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] CalculateSeparation(double[,] positions, int idx, int p)
    {
        var separation = new double[p];
        int count = 0;

        for (int i = 0; i < _populationSize; i++)
        {
            if (i == idx) continue;

            double dist = 0;
            for (int j = 0; j < p; j++)
                dist += Math.Pow(positions[i, j] - positions[idx, j], 2);
            dist = Math.Sqrt(dist);

            if (dist < 0.5) // Neighborhood radius
            {
                for (int j = 0; j < p; j++)
                    separation[j] -= positions[i, j] - positions[idx, j];
                count++;
            }
        }

        if (count > 0)
        {
            for (int j = 0; j < p; j++)
                separation[j] /= count;
        }

        return separation;
    }

    private double[] CalculateAlignment(double[,] velocities, int idx, int p)
    {
        var alignment = new double[p];
        int count = 0;

        for (int i = 0; i < _populationSize; i++)
        {
            if (i == idx) continue;
            for (int j = 0; j < p; j++)
                alignment[j] += velocities[i, j];
            count++;
        }

        if (count > 0)
        {
            for (int j = 0; j < p; j++)
                alignment[j] /= count;
        }

        return alignment;
    }

    private double[] CalculateCohesion(double[,] positions, int idx, int p)
    {
        var cohesion = new double[p];
        int count = 0;

        for (int i = 0; i < _populationSize; i++)
        {
            if (i == idx) continue;
            for (int j = 0; j < p; j++)
                cohesion[j] += positions[i, j];
            count++;
        }

        if (count > 0)
        {
            for (int j = 0; j < p; j++)
                cohesion[j] = cohesion[j] / count - positions[idx, j];
        }

        return cohesion;
    }

    private double[] ComputeFeatureScores(Matrix<T> data, Vector<T> target)
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

            scores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        return scores;
    }

    private double EvaluateFitness(double[,] positions, int idx, int p)
    {
        double fitness = 0;
        double selected = 0;

        for (int j = 0; j < p; j++)
        {
            if (positions[idx, j] > 0.5)
            {
                fitness += _featureScores![j];
                selected++;
            }
        }

        // Penalize for deviating from target number of features
        double penalty = Math.Abs(selected - _nFeaturesToSelect) * 0.1;

        return selected > 0 ? fitness / selected - penalty : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DragonflyAlgorithm has not been fitted.");

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
        throw new NotSupportedException("DragonflyAlgorithm does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DragonflyAlgorithm has not been fitted.");

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
