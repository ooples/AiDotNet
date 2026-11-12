using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Firefly Algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Firefly Algorithm is inspired by the flashing behavior of fireflies.
/// Brighter fireflies attract dimmer ones, with attraction decreasing with
/// distance. This creates a natural clustering toward optimal solutions.
/// </para>
/// <para><b>For Beginners:</b> Imagine fireflies searching for the best spot
/// in a forest. Brighter fireflies (better solutions) attract dimmer ones,
/// so all fireflies gradually move toward the brightest areas. In feature
/// selection, "brightness" represents how good a set of features is.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FireflyAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _alpha;
    private readonly double _beta0;
    private readonly double _gamma;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FireflyAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 25,
        int maxIterations = 50,
        double alpha = 0.5,
        double beta0 = 1.0,
        double gamma = 1.0,
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
        _alpha = alpha;
        _beta0 = beta0;
        _gamma = gamma;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FireflyAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int targetCount = Math.Min(_nFeaturesToSelect, p);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);
        Func<Tensor<T>, Tensor<T>> evaluate = rows => PopulationSearchOps.ThresholdFitness(rows, correlations, targetCount, 0.1);

        // Initialize fireflies
        var fireflies = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var intensity = evaluate(fireflies);

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
            Generation(ref fireflies, ref intensity, evaluate, random, _alpha, _beta0, _gamma);

        // Find best firefly
        var best = PopulationSearchOps.Row(fireflies, PopulationSearchOps.ArgMax(intensity));
        _featureScores = best.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(best, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// One generation of Yang (2009): every firefly i moves towards each brighter firefly j,
    /// x_i = x_i + beta0 exp(-gamma r_ij^2) (x_j - x_i) + alpha (u - 1/2) with u ~ U[0, 1]^d, and its light
    /// intensity is updated after each move; a firefly with no brighter one moves randomly,
    /// x_i = x_i + alpha (u - 1/2). Positions are clamped to [0, 1].
    /// </summary>
    internal static void Generation(
        ref Tensor<T> fireflies, ref Tensor<T> intensity, Func<Tensor<T>, Tensor<T>> evaluate, Random random,
        double alpha, double beta0, double gamma)
    {
        var engine = AiDotNetEngine.Current;
        int m = fireflies.Shape[0];
        int p = fireflies.Shape[1];
        for (int i = 0; i < m; i++)
        {
            bool moved = false;
            for (int j = 0; j < m; j++)
            {
                if (!PopulationSearchOps.Greater(PopulationSearchOps.At(intensity, j), PopulationSearchOps.At(intensity, i)))
                    continue;

                var position = Attract(
                    PopulationSearchOps.Row(fireflies, i), PopulationSearchOps.Row(fireflies, j),
                    PopulationSearchOps.Uniform<T>(random, p), alpha, beta0, gamma);
                fireflies = PopulationSearchOps.ReplaceRow(fireflies, i, position);
                intensity = PopulationSearchOps.ReplaceAt(intensity, i, evaluate(engine.Reshape(position, new[] { 1, p })));
                moved = true;
            }

            if (!moved)
            {
                var position = RandomMove(PopulationSearchOps.Row(fireflies, i), PopulationSearchOps.Uniform<T>(random, p), alpha);
                fireflies = PopulationSearchOps.ReplaceRow(fireflies, i, position);
                intensity = PopulationSearchOps.ReplaceAt(intensity, i, evaluate(engine.Reshape(position, new[] { 1, p })));
            }
        }
    }

    /// <summary>
    /// The move of firefly i towards j: x_i + beta0 exp(-gamma |x_i - x_j|^2) (x_j - x_i) + alpha (u - 1/2),
    /// clamped to [0, 1].
    /// </summary>
    internal static Tensor<T> Attract(Tensor<T> xi, Tensor<T> xj, Tensor<T> u, double alpha, double beta0, double gamma)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var difference = engine.TensorSubtract(xj, xi);
        var squaredDistance = engine.ReduceSum(engine.TensorSquare(difference), new[] { 0 }, true); // [1]
        var attractiveness = engine.TensorMultiplyScalar(
            engine.TensorExp(engine.TensorMultiplyScalar(squaredDistance, ops.FromDouble(-gamma))), ops.FromDouble(beta0));
        var step = engine.TensorAdd(
            PopulationSearchOps.Scale(difference, attractiveness), PopulationSearchOps.Affine(u, alpha, -alpha / 2));
        return PopulationSearchOps.Clamp01(engine.TensorAdd(xi, step));
    }

    /// <summary>The random move of a firefly with no brighter one: x + alpha (u - 1/2), clamped to [0, 1].</summary>
    internal static Tensor<T> RandomMove(Tensor<T> x, Tensor<T> u, double alpha) =>
        PopulationSearchOps.Clamp01(AiDotNetEngine.Current.TensorAdd(x, PopulationSearchOps.Affine(u, alpha, -alpha / 2)));

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
            throw new InvalidOperationException("FireflyAlgorithm has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FireflyAlgorithm does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FireflyAlgorithm has not been fitted.");

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
