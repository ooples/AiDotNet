using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Ant Colony Optimization for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Ant Colony Optimization (ACO) is a swarm intelligence algorithm inspired by how ants
/// find optimal paths using pheromone trails. In feature selection, ants construct feature
/// subsets, and pheromone is deposited on good feature combinations.
/// </para>
/// <para><b>For Beginners:</b> Imagine ants searching for food. When an ant finds a good path,
/// it leaves a scent (pheromone) that attracts other ants. Over time, better paths get
/// stronger scents. Here, "paths" are feature subsets, and "food quality" is model performance.
/// Features that are part of good subsets accumulate more pheromone and are more likely to be selected.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AntColonyFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _alpha; // Pheromone importance
    private readonly double _beta; // Heuristic importance
    private readonly double _evaporationRate;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private Vector<T>? _pheromones;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    public Vector<T>? Pheromones => _pheromones;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AntColonyFS(
        int nFeaturesToSelect = 10,
        int nAnts = 30,
        int nIterations = 50,
        double alpha = 1.0,
        double beta = 2.0,
        double evaporationRate = 0.1,
        Func<Matrix<T>, Vector<T>, int[], T>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _alpha = alpha;
        _beta = beta;
        _evaporationRate = evaporationRate;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonyFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
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

        // Compute initial heuristic (correlation-based)
        var heuristic = ComputeHeuristic(data, target, p);

        // Initialize pheromones
        var pheromones = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);

        int[] bestSolution = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
        T bestFitness = NumOps.FromDouble(double.NegativeInfinity);
        T zero = NumOps.Zero;
        T minimumDeposit = NumOps.FromDouble(0.01);
        T keep = NumOps.FromDouble(1 - _evaporationRate);

        // ACO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            var antSolutions = new List<int[]>();
            var deposits = new T[_nAnts];

            // Each ant constructs a solution (tau^alpha eta^beta is fixed while the ants build)
            var weights = Engine.TensorMultiply(
                Engine.TensorPower(pheromones, NumOps.FromDouble(_alpha)),
                Engine.TensorPower(heuristic, NumOps.FromDouble(_beta)));
            for (int ant = 0; ant < _nAnts; ant++)
            {
                var solution = ConstructSolution(p, weights, random);
                antSolutions.Add(solution);
                T fitness = scorer(data, target, solution);
                deposits[ant] = WrapperScoringOps.IsAbove(fitness, zero) ? fitness : minimumDeposit;

                if (WrapperScoringOps.IsAbove(fitness, bestFitness))
                {
                    bestFitness = fitness;
                    bestSolution = (int[])solution.Clone();
                }
            }

            // Update pheromones: evaporation, then each ant deposits its fitness (at least 0.01) on its features
            pheromones = Engine.TensorMultiplyScalar(pheromones, keep);
            var depositRow = Engine.Reshape(new Tensor<T>(deposits, new[] { _nAnts }), new[] { 1, _nAnts });
            pheromones = Engine.TensorAdd(pheromones, Engine.Reshape(
                Engine.TensorMatMul(depositRow, WrapperScoringOps.MembershipRows<T>(antSolutions, p)), new[] { p }));

            // Elitist strategy: extra deposit for best solution
            T elite = WrapperScoringOps.IsAbove(bestFitness, zero)
                ? NumOps.Multiply(bestFitness, NumOps.FromDouble(2))
                : NumOps.FromDouble(0.02);
            pheromones = Engine.TensorAdd(pheromones,
                Engine.TensorMultiplyScalar(WrapperScoringOps.Membership<T>(bestSolution, p), elite));
        }

        _pheromones = pheromones.ToVector();
        _featureScores = pheromones.ToVector();
        _selectedIndices = bestSolution.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private int[] ConstructSolution(int p, Tensor<T> weights, Random random)
    {
        var selected = new HashSet<int>();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var available = PreprocessingTensorOps.Filled(new[] { p }, NumOps.One);
        T limit = NumOps.FromDouble(1e-10);

        while (selected.Count < numToSelect)
        {
            // Compute selection probabilities tau^alpha eta^beta of the features not yet selected
            var probabilities = Engine.TensorMultiply(weights, available);
            T sum = Engine.TensorSum(probabilities);

            int chosen;
            if (WrapperScoringOps.IsBelow(sum, limit))
            {
                // Random selection if all probabilities are zero
                var remaining = Enumerable.Range(0, p).Where(j => !selected.Contains(j)).ToList();
                if (remaining.Count == 0)
                    break;
                chosen = remaining[random.Next(remaining.Count)];
            }
            else
            {
                // Roulette wheel selection: the first unselected feature whose cumulative probability reaches the
                // threshold (none when rounding leaves the total just short; the ant then draws again)
                T threshold = WrapperScoringOps.Multiply(WrapperScoringOps.Scalar<T>(random.NextDouble()), sum);
                var reached = Engine.TensorMultiply(
                    PreprocessingTensorOps.AtLeast(Engine.TensorCumSum(probabilities, 0), threshold), available);
                var positions = PreprocessingTensorOps.MaskPositions(reached);
                if (positions.Length == 0)
                    continue;
                chosen = positions[0];
            }

            selected.Add(chosen);
            available = Engine.TensorSubtract(available, WrapperScoringOps.Membership<T>(new[] { chosen }, p));
        }

        return selected.ToArray();
    }

    private Tensor<T> ComputeHeuristic(Matrix<T> data, Vector<T> target, int p)
    {
        // |Pearson r| with the target plus a small value to avoid zero
        var correlations = WrapperScoringOps.AbsTargetCorrelations(data, target, Enumerable.Range(0, p).ToArray());
        return Engine.TensorAddScalar(correlations, NumOps.FromDouble(0.01));
    }

    private T DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        // Simple correlation-based score: mean |Pearson r| with the target
        return WrapperScoringOps.MeanAbsTargetCorrelation(data, target, featureIndices);
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
            throw new InvalidOperationException("AntColonyFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AntColonyFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyFS has not been fitted.");

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
