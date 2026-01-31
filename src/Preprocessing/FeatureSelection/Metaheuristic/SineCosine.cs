using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Sine Cosine Algorithm (SCA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Sine Cosine Algorithm uses sine and cosine functions to create a mathematical
/// model for optimization. Agents oscillate around the best solution, with the
/// oscillation range decreasing over iterations.
/// </para>
/// <para><b>For Beginners:</b> Imagine solutions swinging back and forth around the
/// best known answer, like a pendulum. Early on, the swings are big (exploring widely),
/// but they get smaller over time (focusing on the best area). The sine and cosine
/// functions create this natural oscillating behavior.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SineCosine<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly double _a; // Amplitude parameter
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NIterations => _nIterations;
    public double Amplitude => _a;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SineCosine(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        double a = 2.0,
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
        _a = a;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SineCosine requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize positions
        var positions = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
                positions[i, j] = random.NextDouble();
        }

        // Evaluate fitness and find best solution
        var fitness = new double[_populationSize];
        var bestPosition = new double[p];
        double bestFitness = double.MinValue;

        for (int i = 0; i < _populationSize; i++)
        {
            fitness[i] = EvaluateFitness(positions, i, p);
            if (fitness[i] > bestFitness)
            {
                bestFitness = fitness[i];
                for (int j = 0; j < p; j++)
                    bestPosition[j] = positions[i, j];
            }
        }

        // Main loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // r1 decreases linearly from a to 0
            double r1 = _a - iter * _a / _nIterations;

            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double r2 = 2 * Math.PI * random.NextDouble();
                    double r3 = 2 * random.NextDouble();
                    double r4 = random.NextDouble();

                    if (r4 < 0.5)
                    {
                        // Sine equation
                        positions[i, j] = positions[i, j] + r1 * Math.Sin(r2) * Math.Abs(r3 * bestPosition[j] - positions[i, j]);
                    }
                    else
                    {
                        // Cosine equation
                        positions[i, j] = positions[i, j] + r1 * Math.Cos(r2) * Math.Abs(r3 * bestPosition[j] - positions[i, j]);
                    }

                    // Clamp to [0, 1]
                    positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j]));
                }

                // Evaluate fitness
                fitness[i] = EvaluateFitness(positions, i, p);

                // Update best solution
                if (fitness[i] > bestFitness)
                {
                    bestFitness = fitness[i];
                    for (int j = 0; j < p; j++)
                        bestPosition[j] = positions[i, j];
                }
            }
        }

        // Extract selected features from best position
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => bestPosition[j] * _featureScores[j])
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
            throw new InvalidOperationException("SineCosine has not been fitted.");

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
        throw new NotSupportedException("SineCosine does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SineCosine has not been fitted.");

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
