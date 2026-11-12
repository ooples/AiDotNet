using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CuckooSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _discoveryRate;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public double DiscoveryRate => _discoveryRate;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CuckooSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int targetCount = Math.Min(_nFeaturesToSelect, p);
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize nests
        var nests = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(nests, correlations, targetCount, 0.1);

        // Find best nest
        int bestIdx = PopulationSearchOps.ArgMax(fitness);
        var bestFitness = PopulationSearchOps.At(fitness, bestIdx);
        var bestNest = PopulationSearchOps.Row(nests, bestIdx);

        // Mantegna's sigma for the Levy exponent beta = 1.5
        T sigma = NumOps.FromDouble(Math.Pow(
            Math.Exp(LogGamma(1 + LevyBeta)) * Math.Sin(Math.PI * LevyBeta / 2) /
            (Math.Exp(LogGamma((1 + LevyBeta) / 2)) * LevyBeta * Math.Pow(2, (LevyBeta - 1) / 2)),
            1 / LevyBeta));
        T inverseBeta = NumOps.FromDouble(1 / LevyBeta);
        T stepSize = NumOps.FromDouble(0.01);
        int nAbandon = (int)(_populationSize * _discoveryRate);

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Get a cuckoo by Levy flights: per dimension two Box-Muller normals u, v from four uniforms, drawn in
            // the order of the scalar draws, and the step u * sigma / |v|^(1/beta)
            int cuckooIdx = random.Next(_populationSize);
            var draws = PopulationSearchOps.Uniform<T>(random, p, 4);
            var u = engine.TensorMultiplyScalar(GaussianFromUniforms(draws, 0), sigma);
            var v = GaussianFromUniforms(draws, 2);
            var levy = engine.TensorDivide(u, engine.TensorPower(engine.TensorAbs(v), inverseBeta));

            var cuckoo = PopulationSearchOps.Row(nests, cuckooIdx);
            var step = engine.TensorMultiply(
                engine.TensorMultiplyScalar(levy, stepSize), engine.TensorSubtract(cuckoo, bestNest));
            var newSolution = PopulationSearchOps.Clamp01(engine.TensorAdd(cuckoo, step));
            var newFitness = PopulationSearchOps.ThresholdFitness(
                engine.Reshape(newSolution, new[] { 1, p }), correlations, targetCount, 0.1);

            // Replace a random nest if better
            int randomNest = random.Next(_populationSize);
            if (PopulationSearchOps.Greater(newFitness, PopulationSearchOps.At(fitness, randomNest)))
            {
                nests = PopulationSearchOps.ReplaceRow(nests, randomNest, newSolution);
                fitness = PopulationSearchOps.ReplaceAt(fitness, randomNest, newFitness);

                if (PopulationSearchOps.Greater(newFitness, bestFitness))
                {
                    bestFitness = newFitness;
                    bestNest = newSolution;
                }
            }

            // Abandon the worst nests (discovery) and build new ones at random
            if (nAbandon > 0)
            {
                var worstIndices = PreprocessingTensorOps.StableOrder(fitness, descending: false)
                    .Take(nAbandon)
                    .ToArray();
                var freshNests = PopulationSearchOps.Uniform<T>(random, nAbandon, p);
                var freshFitness = PopulationSearchOps.ThresholdFitness(freshNests, correlations, targetCount, 0.1);
                nests = PopulationSearchOps.ScatterRows(nests, worstIndices, freshNests);
                fitness = PopulationSearchOps.ScatterAt(fitness, worstIndices, freshFitness);

                // The first fresh nest that beats the best becomes the best, as with one comparison per nest
                int freshBest = PopulationSearchOps.ArgMax(freshFitness);
                var freshBestFitness = PopulationSearchOps.At(freshFitness, freshBest);
                if (PopulationSearchOps.Greater(freshBestFitness, bestFitness))
                {
                    bestFitness = freshBestFitness;
                    bestNest = PopulationSearchOps.Row(freshNests, freshBest);
                }
            }
        }

        _featureScores = bestNest.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(bestNest, numToSelect);

        IsFitted = true;
    }

    private const double LevyBeta = 1.5;

    /// <summary>
    /// Box-Muller normals sqrt(-2 ln(u1 + 1e-10)) cos(2 pi u2) from columns <paramref name="column"/> and
    /// <paramref name="column"/> + 1 of a [p, k] uniform tensor.
    /// </summary>
    internal static Tensor<T> GaussianFromUniforms(Tensor<T> draws, int column)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = draws.Shape[0];
        var u1 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, column }, new[] { p, 1 }), new[] { p });
        var u2 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, column + 1 }, new[] { p, 1 }), new[] { p });
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(
            engine.TensorLog(engine.TensorAddScalar(u1, PopulationSearchOps.Tiny<T>(1e-10))), ops.FromDouble(-2)));
        return engine.TensorMultiply(radius, engine.TensorCos(engine.TensorMultiplyScalar(u2, ops.FromDouble(2 * Math.PI))));
    }

    private static double LogGamma(double x)
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

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CuckooSearch has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CuckooSearch does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CuckooSearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
