using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Simulated Annealing for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Simulated Annealing (SA) is inspired by the annealing process in metallurgy.
/// It explores the search space by accepting worse solutions with decreasing probability
/// as the "temperature" cools, helping escape local optima.
/// </para>
/// <para><b>For Beginners:</b> Like heating metal and slowly cooling it to remove defects,
/// SA starts "hot" (accepting almost any change) and gradually cools (becoming pickier).
/// Early on, it accepts worse solutions to explore widely. Later, it focuses on refining
/// the best solutions found. This helps avoid getting stuck in suboptimal solutions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SimulatedAnnealingFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly double _initialTemperature;
    private readonly double _coolingRate;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InitialTemperature => _initialTemperature;
    public double CoolingRate => _coolingRate;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SimulatedAnnealingFS(
        int nFeaturesToSelect = 10,
        int nIterations = 1000,
        double initialTemperature = 1.0,
        double coolingRate = 0.995,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (initialTemperature <= 0)
            throw new ArgumentException("Initial temperature must be positive.", nameof(initialTemperature));
        if (coolingRate <= 0 || coolingRate >= 1)
            throw new ArgumentException("Cooling rate must be between 0 and 1.", nameof(coolingRate));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _initialTemperature = initialTemperature;
        _coolingRate = coolingRate;
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SimulatedAnnealingFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var scorer = _scorer ?? DefaultScorer;
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize with random solution
        var current = new bool[p];
        var indices = Enumerable.Range(0, p)
            .OrderBy(_ => random.Next())
            .Take(numToSelect)
            .ToList();
        foreach (int idx in indices)
            current[idx] = true;

        double currentFitness = EvaluateSolution(data, target, current, scorer);

        // Track best solution
        var best = (bool[])current.Clone();
        double bestFitness = currentFitness;

        // Track feature selection frequency for scores
        _featureScores = new double[p];

        double temperature = _initialTemperature;

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate neighbor by flipping one or two bits
            var neighbor = (bool[])current.Clone();

            int numFlips = random.NextDouble() < 0.7 ? 1 : 2;
            for (int f = 0; f < numFlips; f++)
            {
                int flipIdx = random.Next(p);
                neighbor[flipIdx] = !neighbor[flipIdx];
            }

            // Ensure at least one feature is selected
            if (!neighbor.Any(x => x))
            {
                neighbor[random.Next(p)] = true;
            }

            double neighborFitness = EvaluateSolution(data, target, neighbor, scorer);

            // Decide whether to accept neighbor
            double delta = neighborFitness - currentFitness;
            bool accept = delta > 0 || random.NextDouble() < Math.Exp(delta / temperature);

            if (accept)
            {
                current = neighbor;
                currentFitness = neighborFitness;

                if (currentFitness > bestFitness)
                {
                    best = (bool[])current.Clone();
                    bestFitness = currentFitness;
                }
            }

            // Update feature scores based on current solution
            for (int j = 0; j < p; j++)
            {
                if (current[j])
                    _featureScores[j] += currentFitness;
            }

            // Cool down
            temperature *= _coolingRate;
        }

        // Extract selected features from best solution
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (best[j])
                selected.Add(j);
        }

        // If too many features, take top by score
        if (selected.Count > numToSelect)
        {
            _selectedIndices = selected
                .OrderByDescending(j => _featureScores[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = [.. selected.OrderBy(x => x)];
        }

        IsFitted = true;
    }

    private double EvaluateSolution(Matrix<T> data, Vector<T> target, bool[] solution,
        Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        var selected = new List<int>();
        for (int j = 0; j < solution.Length; j++)
        {
            if (solution[j])
                selected.Add(j);
        }

        if (selected.Count == 0)
            return double.NegativeInfinity;

        return scorer(data, target, [.. selected]);
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return double.NegativeInfinity;

        int n = data.Rows;
        double totalCorr = 0;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        foreach (int j in featureIndices)
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

            double corr = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            totalCorr += Math.Abs(corr);
        }

        return totalCorr / featureIndices.Length;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SimulatedAnnealingFS has not been fitted.");

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
        throw new NotSupportedException("SimulatedAnnealingFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SimulatedAnnealingFS has not been fitted.");

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
