using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Cuckoo Search for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Cuckoo Search is inspired by the brood parasitism of cuckoo birds. Cuckoos
/// lay eggs in other birds' nests; if discovered, the host bird may abandon
/// the nest. This translates to solution replacement and exploration.
/// </para>
/// <para><b>For Beginners:</b> Cuckoos lay eggs in other birds' nests. If the
/// host bird finds the foreign egg, it throws it out and rebuilds. In optimization,
/// poor solutions are "discovered" and replaced with new random ones, while good
/// solutions survive and improve through Lévy flights (big random jumps).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CuckooSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _discoveryRate;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public double DiscoveryRate => _discoveryRate;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CuckooSearch(
        int nFeaturesToSelect = 10,
        int populationSize = 25,
        int maxIterations = 50,
        double discoveryRate = 0.25,
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
        if (discoveryRate < 0 || discoveryRate > 1)
            throw new ArgumentException("Discovery rate must be between 0 and 1.", nameof(discoveryRate));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _discoveryRate = discoveryRate;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CuckooSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize nests
        var nests = new double[_populationSize, p];
        for (int i = 0; i < _populationSize; i++)
            for (int j = 0; j < p; j++)
                nests[i, j] = random.NextDouble();

        var fitness = new double[_populationSize];
        for (int i = 0; i < _populationSize; i++)
            fitness[i] = EvaluateSolution(data, target, nests, i, p, n);

        // Find best nest
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

        var bestNest = new double[p];
        for (int j = 0; j < p; j++)
            bestNest[j] = nests[bestIdx, j];

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Get cuckoo by Lévy flights
            int cuckooIdx = random.Next(_populationSize);
            var newSolution = new double[p];

            for (int j = 0; j < p; j++)
            {
                double levy = LevyFlight(random);
                newSolution[j] = nests[cuckooIdx, j] + 0.01 * levy * (nests[cuckooIdx, j] - bestNest[j]);
                newSolution[j] = Math.Max(0, Math.Min(1, newSolution[j]));
            }

            // Evaluate new solution
            var tempNests = (double[,])nests.Clone();
            for (int j = 0; j < p; j++)
                tempNests[cuckooIdx, j] = newSolution[j];

            double newFitness = EvaluateSolution(data, target, tempNests, cuckooIdx, p, n);

            // Replace a random nest if better
            int randomNest = random.Next(_populationSize);
            if (newFitness > fitness[randomNest])
            {
                for (int j = 0; j < p; j++)
                    nests[randomNest, j] = newSolution[j];
                fitness[randomNest] = newFitness;

                if (newFitness > bestFitness)
                {
                    bestFitness = newFitness;
                    for (int j = 0; j < p; j++)
                        bestNest[j] = newSolution[j];
                }
            }

            // Abandon worst nests (discovery)
            int nAbandon = (int)(_populationSize * _discoveryRate);
            var worstIndices = Enumerable.Range(0, _populationSize)
                .OrderBy(i => fitness[i])
                .Take(nAbandon)
                .ToList();

            foreach (int idx in worstIndices)
            {
                for (int j = 0; j < p; j++)
                    nests[idx, j] = random.NextDouble();
                fitness[idx] = EvaluateSolution(data, target, nests, idx, p, n);

                if (fitness[idx] > bestFitness)
                {
                    bestFitness = fitness[idx];
                    for (int j = 0; j < p; j++)
                        bestNest[j] = nests[idx, j];
                }
            }
        }

        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = bestNest[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double LevyFlight(Random random)
    {
        double beta = 1.5;
        double sigma = Math.Pow(
            Math.Exp(LogGamma(1 + beta)) * Math.Sin(Math.PI * beta / 2) /
            (Math.Exp(LogGamma((1 + beta) / 2)) * beta * Math.Pow(2, (beta - 1) / 2)),
            1 / beta);

        double u = GaussianRandom(random) * sigma;
        double v = GaussianRandom(random);

        return u / Math.Pow(Math.Abs(v), 1 / beta);
    }

    private double GaussianRandom(Random random)
    {
        double u1 = random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1 + 1e-10)) * Math.Cos(2 * Math.PI * u2);
    }

    private double LogGamma(double x)
    {
        double[] coef = { 76.18009172947146, -86.50532032941677,
                          24.01409824083091, -1.231739572450155,
                          0.1208650973866179e-2, -0.5395239384953e-5 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += coef[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
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
            throw new InvalidOperationException("CuckooSearch has not been fitted.");

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
        throw new NotSupportedException("CuckooSearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CuckooSearch has not been fitted.");

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
