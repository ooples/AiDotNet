using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Bat Algorithm for feature selection optimization.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Bat Algorithm metaheuristic inspired by the echolocation behavior of bats.
/// Bats emit pulses and adjust their frequency, loudness, and pulse rate to hunt prey.
/// </para>
/// <para><b>For Beginners:</b> Bats use echolocation to find food. In this algorithm,
/// each bat represents a feature subset. Bats fly toward better solutions (prey)
/// by adjusting their "frequency" (search behavior) and "loudness" (exploration vs
/// exploitation balance).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BatAlgorithmFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _fMin;  // Minimum frequency
    private readonly double _fMax;  // Maximum frequency
    private readonly double _loudness;  // Initial loudness
    private readonly double _pulseRate;  // Initial pulse rate
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _fitnessFunction;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BatAlgorithmFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double fMin = 0.0,
        double fMax = 2.0,
        double loudness = 0.5,
        double pulseRate = 0.5,
        Func<Matrix<T>, Vector<T>, int[], double>? fitnessFunction = null,
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
        _maxIterations = maxIterations;
        _fMin = fMin;
        _fMax = fMax;
        _loudness = loudness;
        _pulseRate = pulseRate;
        _fitnessFunction = fitnessFunction;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BatAlgorithmFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize bats
        var positions = new double[_populationSize][];
        var velocities = new double[_populationSize][];
        var fitness = new double[_populationSize];
        var loudness = new double[_populationSize];
        var pulseRate = new double[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            positions[i] = new double[p];
            velocities[i] = new double[p];
            for (int j = 0; j < p; j++)
            {
                positions[i][j] = random.NextDouble();
                velocities[i][j] = 0;
            }
            fitness[i] = EvaluateSolution(positions[i], data, target);
            loudness[i] = _loudness;
            pulseRate[i] = _pulseRate;
        }

        // Find best bat
        int bestIdx = 0;
        var bestPosition = (double[])positions[0].Clone();
        double bestFitness = fitness[0];
        for (int i = 1; i < _populationSize; i++)
        {
            if (fitness[i] > bestFitness)
            {
                bestIdx = i;
                bestFitness = fitness[i];
                bestPosition = (double[])positions[i].Clone();
            }
        }

        // Optimization loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Generate new frequency
                double freq = _fMin + (_fMax - _fMin) * random.NextDouble();

                // Update velocity and position
                for (int j = 0; j < p; j++)
                {
                    velocities[i][j] += (positions[i][j] - bestPosition[j]) * freq;
                    double newPos = positions[i][j] + velocities[i][j];

                    // Local search
                    if (random.NextDouble() > pulseRate[i])
                        newPos = bestPosition[j] + 0.01 * (2 * random.NextDouble() - 1) * loudness[i];

                    positions[i][j] = Math.Max(0, Math.Min(1, newPos));
                }

                // Evaluate new solution
                double newFitness = EvaluateSolution(positions[i], data, target);

                // Accept if improved and random condition
                if (newFitness > fitness[i] && random.NextDouble() < loudness[i])
                {
                    fitness[i] = newFitness;
                    loudness[i] *= 0.9;  // Decrease loudness
                    pulseRate[i] = _pulseRate * (1 - Math.Exp(-0.9 * iter));  // Increase pulse rate
                }

                // Update best
                if (fitness[i] > bestFitness)
                {
                    bestFitness = fitness[i];
                    bestPosition = (double[])positions[i].Clone();
                }
            }
        }

        // Calculate feature importances from best solution
        _featureImportances = bestPosition;

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureImportances
            .Select((v, idx) => (Value: v, Index: idx))
            .OrderByDescending(x => x.Value)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluateSolution(double[] position, Matrix<T> data, Vector<T> target)
    {
        var selectedFeatures = new List<int>();
        for (int i = 0; i < position.Length; i++)
            if (position[i] > 0.5)
                selectedFeatures.Add(i);

        if (selectedFeatures.Count == 0)
            return 0;

        if (_fitnessFunction is not null)
            return _fitnessFunction(data, target, selectedFeatures.ToArray());

        return DefaultFitness(data, target, selectedFeatures.ToArray());
    }

    private double DefaultFitness(Matrix<T> data, Vector<T> target, int[] features)
    {
        int n = data.Rows;
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double totalCorr = 0;
        foreach (int j in features)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                totalCorr += Math.Abs(ssXY / Math.Sqrt(ssXX * ssYY));
        }

        return totalCorr / features.Length - 0.01 * features.Length;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BatAlgorithmFS has not been fitted.");

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
        throw new NotSupportedException("BatAlgorithmFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BatAlgorithmFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
