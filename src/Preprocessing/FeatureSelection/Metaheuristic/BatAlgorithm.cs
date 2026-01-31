using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Bat Algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Bat Algorithm mimics the echolocation behavior of bats. Bats adjust their
/// pulse frequency and loudness as they approach prey, which translates to
/// exploration-exploitation balance in optimization.
/// </para>
/// <para><b>For Beginners:</b> Bats use sound pulses to find food. When far from
/// prey, they use loud, low-frequency sounds. As they get closer, they use
/// quieter, higher-frequency sounds. This algorithm mimics that behavior -
/// exploring broadly at first, then focusing on promising areas.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BatAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _frequencyMin;
    private readonly double _frequencyMax;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BatAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double frequencyMin = 0.0,
        double frequencyMax = 2.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _frequencyMin = frequencyMin;
        _frequencyMax = frequencyMax;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BatAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize bats
        var positions = new double[_populationSize, p];
        var velocities = new double[_populationSize, p];
        var frequencies = new double[_populationSize];
        var loudness = new double[_populationSize];
        var pulseRates = new double[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            for (int j = 0; j < p; j++)
            {
                positions[i, j] = random.NextDouble();
                velocities[i, j] = 0;
            }
            loudness[i] = random.NextDouble() * 2;
            pulseRates[i] = random.NextDouble();
        }

        var fitness = new double[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            fitness[i] = EvaluateSolution(data, target, positions, i, p, n);

        // Find best bat
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

        var bestPos = new double[p];
        for (int j = 0; j < p; j++)
            bestPos[j] = positions[bestIdx, j];

        // Main loop
        double alpha = 0.9;
        double gamma = 0.9;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                frequencies[i] = _frequencyMin + (_frequencyMax - _frequencyMin) * random.NextDouble();

                for (int j = 0; j < p; j++)
                {
                    velocities[i, j] += (positions[i, j] - bestPos[j]) * frequencies[i];
                    double newPos = positions[i, j] + velocities[i, j];

                    // Local search
                    if (random.NextDouble() > pulseRates[i])
                    {
                        double avgLoudness = 0;
                        for (int k = 0; k < _populationSize; k++)
                            avgLoudness += loudness[k];
                        avgLoudness /= _populationSize;

                        newPos = bestPos[j] + 0.01 * avgLoudness * (random.NextDouble() * 2 - 1);
                    }

                    positions[i, j] = Math.Max(0, Math.Min(1, newPos));
                }

                double newFitness = EvaluateSolution(data, target, positions, i, p, n);

                // Accept if better and loud enough
                if (random.NextDouble() < loudness[i] && newFitness > fitness[i])
                {
                    fitness[i] = newFitness;
                    loudness[i] *= alpha;
                    pulseRates[i] = pulseRates[i] * (1 - Math.Exp(-gamma * iter));
                }

                if (fitness[i] > bestFitness)
                {
                    bestFitness = fitness[i];
                    for (int j = 0; j < p; j++)
                        bestPos[j] = positions[i, j];
                }
            }
        }

        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = bestPos[j];

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
            throw new InvalidOperationException("BatAlgorithm has not been fitted.");

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
        throw new NotSupportedException("BatAlgorithm does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BatAlgorithm has not been fitted.");

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
