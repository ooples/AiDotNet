using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Particle Swarm Optimization based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses particle swarm optimization to find optimal feature subsets where particles
/// represent feature combinations moving through the solution space.
/// </para>
/// <para><b>For Beginners:</b> Imagine a swarm of birds searching for food. Each bird
/// (particle) explores feature combinations, remembers its best finds, and learns from
/// the swarm's best discoveries. Over time, they converge on good feature subsets.
/// </para>
/// </remarks>
public class ParticleSwarmSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _swarmSize;
    private readonly int _nIterations;
    private readonly double _inertia;
    private readonly double _cognitive;
    private readonly double _social;
    private readonly int? _randomState;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int SwarmSize => _swarmSize;
    public int NIterations => _nIterations;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ParticleSwarmSelector(
        int nFeaturesToSelect = 10,
        int swarmSize = 30,
        int nIterations = 100,
        double inertia = 0.7,
        double cognitive = 1.5,
        double social = 1.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _swarmSize = swarmSize;
        _nIterations = nIterations;
        _inertia = inertia;
        _cognitive = cognitive;
        _social = social;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ParticleSwarmSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize particles (positions as probabilities)
        var positions = new double[_swarmSize, p];
        var velocities = new double[_swarmSize, p];
        var personalBest = new double[_swarmSize, p];
        var personalBestFitness = new double[_swarmSize];
        var globalBest = new double[p];
        double globalBestFitness = double.MinValue;

        for (int i = 0; i < _swarmSize; i++)
        {
            personalBestFitness[i] = double.MinValue;
            for (int j = 0; j < p; j++)
            {
                positions[i, j] = rand.NextDouble();
                velocities[i, j] = (rand.NextDouble() - 0.5) * 0.1;
                personalBest[i, j] = positions[i, j];
            }
        }

        _featureImportances = new double[p];

        // PSO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            for (int i = 0; i < _swarmSize; i++)
            {
                // Convert position to binary mask
                var mask = new bool[p];
                var sortedIndices = Enumerable.Range(0, p)
                    .OrderByDescending(j => positions[i, j])
                    .Take(numToSelect)
                    .ToList();
                foreach (int idx in sortedIndices)
                    mask[idx] = true;

                double fitness = EvaluateFitness(X, y, mask, n, p);

                // Update personal best
                if (fitness > personalBestFitness[i])
                {
                    personalBestFitness[i] = fitness;
                    for (int j = 0; j < p; j++)
                        personalBest[i, j] = positions[i, j];
                }

                // Update global best
                if (fitness > globalBestFitness)
                {
                    globalBestFitness = fitness;
                    for (int j = 0; j < p; j++)
                        globalBest[j] = positions[i, j];
                }

                // Track feature importance
                for (int j = 0; j < p; j++)
                {
                    if (mask[j])
                        _featureImportances[j] += fitness;
                }
            }

            // Update velocities and positions
            for (int i = 0; i < _swarmSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double r1 = rand.NextDouble();
                    double r2 = rand.NextDouble();

                    velocities[i, j] = _inertia * velocities[i, j]
                        + _cognitive * r1 * (personalBest[i, j] - positions[i, j])
                        + _social * r2 * (globalBest[j] - positions[i, j]);

                    velocities[i, j] = Math.Max(-1, Math.Min(1, velocities[i, j]));
                    positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j] + velocities[i, j]));
                }
            }
        }

        // Normalize and select
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => globalBest[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

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
            throw new InvalidOperationException("ParticleSwarmSelector has not been fitted.");

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
        throw new NotSupportedException("ParticleSwarmSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ParticleSwarmSelector has not been fitted.");

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
