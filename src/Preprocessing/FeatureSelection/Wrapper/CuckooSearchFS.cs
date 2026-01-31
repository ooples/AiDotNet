using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Cuckoo Search algorithm for feature selection optimization.
/// </summary>
/// <remarks>
/// <para>
/// Inspired by the brood parasitism of cuckoo birds. Uses Levy flights
/// for exploration and replaces worst nests to evolve better solutions.
/// </para>
/// <para><b>For Beginners:</b> Like cuckoos laying eggs in other birds' nests,
/// this algorithm places new solutions and keeps the best ones. Levy flights
/// (long occasional jumps) help escape local optima.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CuckooSearchFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly double _discoveryProbability;
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, bool[], double>? _fitnessFunc;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private double _bestFitness;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public double BestFitness => _bestFitness;
    public override bool SupportsInverseTransform => false;

    public CuckooSearchFS(
        int nFeaturesToSelect = 10,
        int populationSize = 25,
        int nIterations = 100,
        double discoveryProbability = 0.25,
        Func<Matrix<T>, Vector<T>, bool[], double>? fitnessFunc = null,
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
        _discoveryProbability = discoveryProbability;
        _fitnessFunc = fitnessFunc;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CuckooSearchFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize nests (population)
        var nests = new List<(bool[] Solution, double Fitness)>();
        for (int i = 0; i < _populationSize; i++)
        {
            var solution = new bool[p];
            int nSelected = random.Next(1, Math.Min(_nFeaturesToSelect + 3, p));
            var indices = Enumerable.Range(0, p).OrderBy(_ => random.Next()).Take(nSelected).ToList();
            foreach (int idx in indices)
                solution[idx] = true;

            double fitness = EvaluateFitness(data, target, solution);
            nests.Add((solution, fitness));
        }

        _featureImportances = new double[p];
        _bestFitness = nests.Max(n => n.Fitness);
        var bestSolution = nests.OrderByDescending(n => n.Fitness).First().Solution;

        // Main Cuckoo Search loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate new solutions via Levy flights
            for (int i = 0; i < _populationSize; i++)
            {
                var newSolution = LevyFlight(nests[i].Solution, bestSolution, p, random);

                // Ensure at least one feature
                if (!newSolution.Any(b => b))
                    newSolution[random.Next(p)] = true;

                double newFitness = EvaluateFitness(data, target, newSolution);

                // Update feature importances
                for (int j = 0; j < p; j++)
                    if (newSolution[j])
                        _featureImportances[j] += newFitness;

                // Replace if better
                if (newFitness > nests[i].Fitness)
                {
                    nests[i] = (newSolution, newFitness);
                    if (newFitness > _bestFitness)
                    {
                        _bestFitness = newFitness;
                        bestSolution = (bool[])newSolution.Clone();
                    }
                }
            }

            // Abandon worst nests (discovery)
            var sorted = nests.OrderBy(n => n.Fitness).ToList();
            int nToAbandon = (int)(_populationSize * _discoveryProbability);

            for (int i = 0; i < nToAbandon; i++)
            {
                var newSolution = new bool[p];
                int nSelected = random.Next(1, Math.Min(_nFeaturesToSelect + 3, p));
                var indices = Enumerable.Range(0, p).OrderBy(_ => random.Next()).Take(nSelected).ToList();
                foreach (int idx in indices)
                    newSolution[idx] = true;

                double newFitness = EvaluateFitness(data, target, newSolution);
                sorted[i] = (newSolution, newFitness);
            }

            nests = sorted;
        }

        // Select from best solution
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
            if (bestSolution[j])
                selectedList.Add(j);

        if (selectedList.Count > _nFeaturesToSelect)
        {
            selectedList = selectedList
                .OrderByDescending(j => _featureImportances[j])
                .Take(_nFeaturesToSelect)
                .ToList();
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private bool[] LevyFlight(bool[] current, bool[] best, int p, Random random)
    {
        var result = new bool[p];

        for (int j = 0; j < p; j++)
        {
            // Levy flight: occasional large jumps
            double levy = LevyDistribution(random);
            double step = levy * (best[j] ? 1 : -1) * 0.1;

            // Probabilistic update
            double prob = 1.0 / (1.0 + Math.Exp(-step));
            if (random.NextDouble() < prob)
                result[j] = current[j];
            else
                result[j] = !current[j];

            // Bias toward best solution
            if (random.NextDouble() < 0.3)
                result[j] = best[j];
        }

        return result;
    }

    private double LevyDistribution(Random random)
    {
        // Mantegna's algorithm for Levy stable distribution
        double sigma = Math.Pow(
            (SpecialFunctions.Gamma(1.5) * Math.Sin(Math.PI * 0.5 / 2)) /
            (SpecialFunctions.Gamma(0.75) * 0.5 * Math.Pow(2, 0.25)),
            2);

        double u = GaussianRandom(random) * sigma;
        double v = Math.Abs(GaussianRandom(random));

        return u / Math.Pow(v, 0.5);
    }

    private double GaussianRandom(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private static class SpecialFunctions
    {
        public static double Gamma(double x)
        {
            // Lanczos approximation
            double[] g = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                          -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };

            double y = x;
            double tmp = x + 5.5;
            tmp -= (x + 0.5) * Math.Log(tmp);
            double ser = 1.000000000190015;
            for (int j = 0; j < 6; j++)
                ser += g[j] / ++y;

            return Math.Exp(-tmp + Math.Log(2.5066282746310005 * ser / x));
        }
    }

    private double EvaluateFitness(Matrix<T> data, Vector<T> target, bool[] mask)
    {
        if (_fitnessFunc is not null)
            return _fitnessFunc(data, target, mask);

        int n = data.Rows;
        int nSelected = mask.Count(b => b);
        if (nSelected == 0) return double.NegativeInfinity;

        double totalCorr = 0;
        for (int j = 0; j < mask.Length; j++)
        {
            if (!mask[j]) continue;

            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += NumOps.ToDouble(data[i, j]);
                yMean += NumOps.ToDouble(target[i]);
            }
            xMean /= n;
            yMean /= n;

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

        double penalty = nSelected > _nFeaturesToSelect ? 0.1 * (nSelected - _nFeaturesToSelect) : 0;
        return totalCorr / nSelected - penalty;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CuckooSearchFS has not been fitted.");

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
        throw new NotSupportedException("CuckooSearchFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CuckooSearchFS has not been fitted.");

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
