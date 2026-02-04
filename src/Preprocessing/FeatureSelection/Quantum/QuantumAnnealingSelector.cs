using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Quantum;

/// <summary>
/// Quantum Annealing-Inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Simulates quantum annealing for feature selection, using quantum tunneling
/// effects to escape local optima and find globally optimal feature subsets.
/// </para>
/// <para><b>For Beginners:</b> Quantum annealing is like simulated annealing
/// (slowly cooling a system) but with quantum effects that allow "tunneling"
/// through barriers to find better solutions. This method simulates this
/// process to search for the best feature combination.
/// </para>
/// </remarks>
public class QuantumAnnealingSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSteps;
    private readonly double _initialGamma;
    private readonly double _finalGamma;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public QuantumAnnealingSelector(
        int nFeaturesToSelect = 10,
        int nSteps = 1000,
        double initialGamma = 1.0,
        double finalGamma = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSteps = nSteps;
        _initialGamma = initialGamma;
        _finalGamma = finalGamma;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "QuantumAnnealingSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Precompute feature correlations with target
        var correlations = new double[p];
        double yMean = y.Average();
        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }
            correlations[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        // Initialize current solution
        var current = new bool[p];
        var indices = Enumerable.Range(0, p)
            .OrderByDescending(j => correlations[j])
            .Take(_nFeaturesToSelect)
            .ToList();
        foreach (int j in indices) current[j] = true;

        double currentEnergy = ComputeEnergy(correlations, current, p);
        var best = (bool[])current.Clone();
        double bestEnergy = currentEnergy;

        var selectionCounts = new double[p];

        for (int step = 0; step < _nSteps; step++)
        {
            // Anneal gamma (quantum tunneling strength)
            double progress = (double)step / _nSteps;
            double gamma = _initialGamma * Math.Pow(_finalGamma / _initialGamma, progress);

            // Temperature decreases linearly
            double temp = (1 - progress) + 0.01;

            // Propose a move (flip a random bit)
            int flipIdx = rand.Next(p);
            current[flipIdx] = !current[flipIdx];

            // Ensure valid solution
            int selectedCount = current.Count(s => s);
            if (selectedCount == 0)
            {
                current[flipIdx] = !current[flipIdx];
                continue;
            }

            double newEnergy = ComputeEnergy(correlations, current, p);

            // Accept or reject with quantum tunneling probability
            double delta = newEnergy - currentEnergy;
            double tunnelProb = Math.Exp(-Math.Abs(delta) / gamma);
            double thermalProb = Math.Exp(-delta / temp);
            double acceptProb = (delta <= 0) ? 1.0 : Math.Max(tunnelProb, thermalProb);

            if (rand.NextDouble() < acceptProb)
            {
                currentEnergy = newEnergy;
                if (currentEnergy < bestEnergy)
                {
                    bestEnergy = currentEnergy;
                    Array.Copy(current, best, p);
                }
            }
            else
            {
                current[flipIdx] = !current[flipIdx]; // Revert
            }

            // Track selection frequency
            for (int j = 0; j < p; j++)
                if (current[j])
                    selectionCounts[j]++;
        }

        // Compute feature scores
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            _featureScores[j] = correlations[j] + selectionCounts[j] / _nSteps;
            if (best[j])
                _featureScores[j] += 1.0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeEnergy(double[] correlations, bool[] solution, int p)
    {
        // Energy = negative sum of correlations (we minimize energy)
        double energy = 0;
        int count = 0;
        for (int j = 0; j < p; j++)
        {
            if (solution[j])
            {
                energy -= correlations[j];
                count++;
            }
        }

        // Penalty for deviation from target count
        energy += 0.1 * Math.Abs(count - _nFeaturesToSelect);

        return energy;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumAnnealingSelector has not been fitted.");

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
        throw new NotSupportedException("QuantumAnnealingSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("QuantumAnnealingSelector has not been fitted.");

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
