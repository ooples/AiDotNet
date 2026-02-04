using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Particle Swarm Optimization for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Particle Swarm Optimization (PSO) simulates a swarm of particles moving through
/// the search space. Each particle represents a candidate feature subset and moves
/// based on its own best position and the swarm's best position.
/// </para>
/// <para><b>For Beginners:</b> Imagine a flock of birds searching for food. Each bird
/// remembers where it found the best food, and the flock shares information about
/// the best location found by any bird. They balance exploring new areas with
/// returning to known good spots. Here, "food" is the best feature subset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ParticleSwarmFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nParticles;
    private readonly int _nIterations;
    private readonly double _w; // Inertia weight
    private readonly double _c1; // Cognitive coefficient
    private readonly double _c2; // Social coefficient
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scorer;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InertiaWeight => _w;
    public double CognitiveCoefficient => _c1;
    public double SocialCoefficient => _c2;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ParticleSwarmFS(
        int nFeaturesToSelect = 10,
        int nParticles = 30,
        int nIterations = 100,
        double inertiaWeight = 0.7,
        double cognitiveCoefficient = 1.5,
        double socialCoefficient = 1.5,
        Func<Matrix<T>, Vector<T>, int[], double>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nParticles = nParticles;
        _nIterations = nIterations;
        _w = inertiaWeight;
        _c1 = cognitiveCoefficient;
        _c2 = socialCoefficient;
        _scorer = scorer;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ParticleSwarmFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize particles (positions and velocities)
        var positions = new double[_nParticles, p];
        var velocities = new double[_nParticles, p];
        var pBest = new double[_nParticles, p]; // Personal best positions
        var pBestFitness = new double[_nParticles];
        var gBest = new double[p]; // Global best position
        double gBestFitness = double.NegativeInfinity;

        // Initialize
        for (int i = 0; i < _nParticles; i++)
        {
            for (int j = 0; j < p; j++)
            {
                positions[i, j] = random.NextDouble();
                velocities[i, j] = (random.NextDouble() - 0.5) * 0.1;
                pBest[i, j] = positions[i, j];
            }

            pBestFitness[i] = EvaluatePosition(data, target, positions, i, p, scorer);

            if (pBestFitness[i] > gBestFitness)
            {
                gBestFitness = pBestFitness[i];
                for (int j = 0; j < p; j++)
                    gBest[j] = positions[i, j];
            }
        }

        // PSO main loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            for (int i = 0; i < _nParticles; i++)
            {
                // Update velocity
                for (int j = 0; j < p; j++)
                {
                    double r1 = random.NextDouble();
                    double r2 = random.NextDouble();

                    velocities[i, j] = _w * velocities[i, j]
                        + _c1 * r1 * (pBest[i, j] - positions[i, j])
                        + _c2 * r2 * (gBest[j] - positions[i, j]);

                    // Clamp velocity
                    velocities[i, j] = Math.Max(-1, Math.Min(1, velocities[i, j]));
                }

                // Update position
                for (int j = 0; j < p; j++)
                {
                    positions[i, j] += velocities[i, j];
                    positions[i, j] = Math.Max(0, Math.Min(1, positions[i, j]));
                }

                // Evaluate new position
                double fitness = EvaluatePosition(data, target, positions, i, p, scorer);

                // Update personal best
                if (fitness > pBestFitness[i])
                {
                    pBestFitness[i] = fitness;
                    for (int j = 0; j < p; j++)
                        pBest[i, j] = positions[i, j];

                    // Update global best
                    if (fitness > gBestFitness)
                    {
                        gBestFitness = fitness;
                        for (int j = 0; j < p; j++)
                            gBest[j] = positions[i, j];
                    }
                }
            }
        }

        // Extract selected features from global best
        _featureScores = gBest;
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = gBest
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EvaluatePosition(Matrix<T> data, Vector<T> target, double[,] positions,
        int particleIdx, int p, Func<Matrix<T>, Vector<T>, int[], double> scorer)
    {
        // Convert continuous position to binary selection (threshold = 0.5)
        var selected = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (positions[particleIdx, j] > 0.5)
                selected.Add(j);
        }

        // If no features selected, take top by position value
        if (selected.Count == 0)
        {
            var topIdx = Enumerable.Range(0, p)
                .OrderByDescending(j => positions[particleIdx, j])
                .First();
            selected.Add(topIdx);
        }

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
            throw new InvalidOperationException("ParticleSwarmFS has not been fitted.");

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
        throw new NotSupportedException("ParticleSwarmFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ParticleSwarmFS has not been fitted.");

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
