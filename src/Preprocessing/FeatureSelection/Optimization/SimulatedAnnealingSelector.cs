using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Optimization;

/// <summary>
/// Simulated Annealing based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses simulated annealing optimization to find optimal feature subsets by
/// accepting both improvements and occasionally worse solutions to escape local optima.
/// </para>
/// <para><b>For Beginners:</b> Like cooling metal, this starts with high "temperature"
/// where it accepts almost any change, then gradually cools down and becomes pickier.
/// This helps find good feature combinations by exploring broadly first, then refining.
/// </para>
/// </remarks>
public class SimulatedAnnealingSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _initialTemperature;
    private readonly double _coolingRate;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InitialTemperature => _initialTemperature;
    public double CoolingRate => _coolingRate;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SimulatedAnnealingSelector(
        int nFeaturesToSelect = 10,
        double initialTemperature = 100.0,
        double coolingRate = 0.95,
        int nIterations = 1000,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _initialTemperature = initialTemperature;
        _coolingRate = coolingRate;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SimulatedAnnealingSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize current solution
        var current = new bool[p];
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToList();
        foreach (int idx in indices)
            current[idx] = true;

        double currentEnergy = -EvaluateFitness(X, y, current, n, p);
        var best = (bool[])current.Clone();
        double bestEnergy = currentEnergy;

        _featureImportances = new double[p];
        double temperature = _initialTemperature;

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Generate neighbor by swapping a feature
            var neighbor = (bool[])current.Clone();
            int selectedCount = 0;
            for (int j = 0; j < p; j++) if (neighbor[j]) selectedCount++;

            if (rand.NextDouble() < 0.5 && selectedCount > 1)
            {
                // Remove a random selected feature
                var selected = Enumerable.Range(0, p).Where(j => neighbor[j]).ToList();
                int toRemove = selected[rand.Next(selected.Count)];
                neighbor[toRemove] = false;

                // Add a random unselected feature
                var unselected = Enumerable.Range(0, p).Where(j => !neighbor[j]).ToList();
                if (unselected.Count > 0)
                {
                    int toAdd = unselected[rand.Next(unselected.Count)];
                    neighbor[toAdd] = true;
                }
            }
            else
            {
                // Flip a random feature
                int toFlip = rand.Next(p);
                neighbor[toFlip] = !neighbor[toFlip];
            }

            double neighborEnergy = -EvaluateFitness(X, y, neighbor, n, p);
            double delta = neighborEnergy - currentEnergy;

            // Accept or reject
            if (delta < 0 || rand.NextDouble() < Math.Exp(-delta / temperature))
            {
                current = neighbor;
                currentEnergy = neighborEnergy;

                if (currentEnergy < bestEnergy)
                {
                    best = (bool[])current.Clone();
                    bestEnergy = currentEnergy;
                }
            }

            // Track feature importance
            for (int j = 0; j < p; j++)
            {
                if (current[j])
                    _featureImportances[j] += -currentEnergy;
            }

            // Cool down
            temperature *= _coolingRate;
        }

        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => best[j])
            .OrderBy(x => x)
            .ToArray();

        // Normalize feature importances
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        IsFitted = true;
    }

    private double EvaluateFitness(double[,] X, double[] y, bool[] mask, int n, int p)
    {
        var selectedFeatures = Enumerable.Range(0, p).Where(j => mask[j]).ToList();
        if (selectedFeatures.Count == 0) return 0;

        double totalCorr = 0;
        foreach (int j in selectedFeatures)
        {
            double corr = ComputeCorrelation(X, y, j, n);
            totalCorr += Math.Abs(corr);
        }

        return totalCorr / selectedFeatures.Count;
    }

    private double ComputeCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += X[i, j];
            yMean += y[i];
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SimulatedAnnealingSelector has not been fitted.");

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
        throw new NotSupportedException("SimulatedAnnealingSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SimulatedAnnealingSelector has not been fitted.");

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
