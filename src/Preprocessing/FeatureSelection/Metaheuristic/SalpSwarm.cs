using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Salp Swarm Algorithm (SSA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Salp Swarm Algorithm is inspired by the swarming behavior of salps in oceans.
/// Salps form a chain where the leader follows the food source and followers follow
/// the salp ahead of them.
/// </para>
/// <para><b>For Beginners:</b> Salps are sea creatures that form chains. The first
/// salp (leader) moves toward the best food, while others follow in line. This
/// creates a balance between exploring new areas and exploiting known good spots,
/// which helps find the best feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SalpSwarm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public SalpSwarm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SalpSwarm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize salp positions
        var positions = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
                positions[i, j] = random.NextDouble();
        }

        // Evaluate fitness and find food source (best solution)
        var fitness = new double[_populationSize];
        var foodPosition = new double[p];
        double foodFitness = double.MinValue;

        for (int i = 0; i < _populationSize; i++)
        {
            fitness[i] = EvaluateFitness(positions, i, p);
            if (fitness[i] > foodFitness)
            {
                foodFitness = fitness[i];
                for (int j = 0; j < p; j++)
                    foodPosition[j] = positions[i, j];
            }
        }

        // Main loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // c1 decreases linearly from 2 to 0
            double c1 = 2 * Math.Exp(-Math.Pow(4.0 * iter / _nIterations, 2));

            for (int i = 0; i < _populationSize; i++)
            {
                if (i < _populationSize / 2)
                {
                    // Leaders - follow the food
                    for (int j = 0; j < p; j++)
                    {
                        double c2 = random.NextDouble();
                        double c3 = random.NextDouble();

                        if (c3 < 0.5)
                        {
                            positions[i, j] = foodPosition[j] + c1 * (1.0 - 0) * c2 + 0;
                        }
                        else
                        {
                            positions[i, j] = foodPosition[j] - c1 * (1.0 - 0) * c2 + 0;
                        }

                        // Clamp to [0, 1]
                        positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j]));
                    }
                }
                else
                {
                    // Followers - follow the salp ahead
                    for (int j = 0; j < p; j++)
                    {
                        positions[i, j] = (positions[i, j] + positions[i - 1, j]) / 2;
                        positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j]));
                    }
                }

                // Evaluate fitness
                fitness[i] = EvaluateFitness(positions, i, p);

                // Update food source
                if (fitness[i] > foodFitness)
                {
                    foodFitness = fitness[i];
                    for (int j = 0; j < p; j++)
                        foodPosition[j] = positions[i, j];
                }
            }
        }

        // Extract selected features from food position
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => foodPosition[j] * _featureScores[j])
            .Take(_nFeaturesToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
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
            throw new InvalidOperationException("SalpSwarm has not been fitted.");

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
        throw new NotSupportedException("SalpSwarm does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SalpSwarm has not been fitted.");

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
