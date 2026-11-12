using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Grey Wolf Optimizer for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Inspired by grey wolf pack hunting behavior with alpha, beta, delta leaders.
/// Wolves encircle prey (optimal solution) guided by the three best solutions.
/// </para>
/// <para><b>For Beginners:</b> Like wolves hunting together, this algorithm
/// uses the three best current solutions (alpha, beta, delta) to guide the
/// pack toward better feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GreyWolfFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly int _nFeaturesToSelect;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, bool[], T>? _fitnessFunc;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private T _bestFitness = MathHelper.GetNumericOperations<T>().Zero;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    public T BestFitness => _bestFitness;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GreyWolfFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        Func<Matrix<T>, Vector<T>, bool[], T>? fitnessFunc = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 4)
            throw new ArgumentException("Population size must be at least 4.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _fitnessFunc = fitnessFunc;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GreyWolfFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize wolves (population): positions uniform on [0, 1), drawn from the selector's generator in
        // master's order
        var wolves = new List<(Tensor<T> Position, T Fitness)>();
        for (int i = 0; i < _populationSize; i++)
        {
            var draws = WrapperScoringOps.NextDoubles(random, p);
            var position = WrapperScoringOps.FromDoubles<T>(p, j => draws[j]);
            var solution = PositionToSolution(position);
            T fitness = EvaluateFitness(data, target, solution);
            wolves.Add((position, fitness));
        }

        var importances = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);

        // Sort to get alpha, beta, delta
        wolves = SortByFitness(wolves);
        var alpha = wolves[0];
        var beta = wolves[1];
        var delta = wolves[2];
        _bestFitness = alpha.Fitness;
        T zero = NumOps.Zero;
        T one = NumOps.One;
        T third = NumOps.FromDouble(1.0 / 3);

        // Main GWO loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            double a = 2 - iter * (2.0 / _nIterations);  // Linearly decreasing from 2 to 0

            for (int i = 3; i < _populationSize; i++)
            {
                var wolf = wolves[i];

                // Each leader L pulls the wolf to L - A |C L - x| with A = 2 a r1 - a and C = 2 r2 (r1, r2 uniform on
                // [0, 1)); the new position is the mean of the three pulls, clamped to [0, 1]. The draws keep master's
                // order: per dimension, (r1, r2) for alpha, beta, then delta
                var draws = WrapperScoringOps.NextDoubles(random, 6 * p);
                var x1 = LeaderPull(alpha.Position, wolf.Position, a, p, draws, 0);
                var x2 = LeaderPull(beta.Position, wolf.Position, a, p, draws, 1);
                var x3 = LeaderPull(delta.Position, wolf.Position, a, p, draws, 2);
                var newPosition = Engine.TensorClamp(
                    Engine.TensorMultiplyScalar(Engine.TensorAdd(Engine.TensorAdd(x1, x2), x3), third), zero, one);

                var newSolution = PositionToSolution(newPosition);
                T newFitness = EvaluateFitness(data, target, newSolution);

                // Update feature importances
                importances = Engine.TensorAdd(importances, WrapperScoringOps.ScoreSums(
                    new[] { WrapperScoringOps.Positions(newSolution) }, WrapperScoringOps.FromList(new[] { newFitness }), p));

                wolves[i] = (newPosition, newFitness);
            }

            // Update alpha, beta, delta
            wolves = SortByFitness(wolves);
            if (WrapperScoringOps.IsAbove(wolves[0].Fitness, alpha.Fitness))
            {
                alpha = wolves[0];
                _bestFitness = alpha.Fitness;
            }
            if (WrapperScoringOps.IsAbove(wolves[1].Fitness, beta.Fitness))
                beta = wolves[1];
            if (WrapperScoringOps.IsAbove(wolves[2].Fitness, delta.Fitness))
                delta = wolves[2];
        }

        _featureImportances = importances.ToVector();

        // Select from best (alpha) solution
        var selectedList = WrapperScoringOps.Positions(PositionToSolution(alpha.Position));

        if (selectedList.Length > _nFeaturesToSelect)
        {
            selectedList = PreprocessingTensorOps.TopKIndicesSorted(
                WrapperScoringOps.Gather(importances, selectedList), _nFeaturesToSelect)
                .Select(position => selectedList[position])
                .ToArray();
        }
        else if (selectedList.Length == 0)
        {
            selectedList = PreprocessingTensorOps.TopKIndicesSorted(importances, Math.Min(_nFeaturesToSelect, p));
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private List<(Tensor<T> Position, T Fitness)> SortByFitness(List<(Tensor<T> Position, T Fitness)> wolves)
    {
        // By decreasing fitness, the earlier wolf first among ties
        var order = PreprocessingTensorOps.StableOrder(
            WrapperScoringOps.FromList(wolves.Select(w => w.Fitness).ToList()), descending: true);
        return order.Select(i => wolves[i]).ToList();
    }

    private Tensor<T> LeaderPull(Tensor<T> leader, Tensor<T> position, double a, int p, double[] draws, int leaderIndex)
    {
        var coefficientA = WrapperScoringOps.FromDoubles<T>(p, j => 2 * a * draws[6 * j + 2 * leaderIndex] - a);
        var coefficientC = WrapperScoringOps.FromDoubles<T>(p, j => 2 * draws[6 * j + 2 * leaderIndex + 1]);
        var distance = Engine.TensorAbs(Engine.TensorSubtract(Engine.TensorMultiply(coefficientC, leader), position));
        return Engine.TensorSubtract(leader, Engine.TensorMultiply(coefficientA, distance));
    }

    private bool[] PositionToSolution(Tensor<T> position)
    {
        int length = position.Length;
        var solution = new bool[length];
        var above = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            Engine.TensorGreaterThan(position, NumOps.FromDouble(0.5))));

        // Ensure roughly nFeaturesToSelect features: by decreasing value, take those above 0.5 and at least the
        // requested number, at most nFeaturesToSelect + 3
        int count = 0;
        foreach (int idx in PreprocessingTensorOps.StableOrder(position, descending: true))
        {
            if (above.Contains(idx) || count < Math.Min(_nFeaturesToSelect, length))
            {
                solution[idx] = true;
                count++;
            }
            if (count >= _nFeaturesToSelect + 3) break;
        }

        return solution;
    }

    private T EvaluateFitness(Matrix<T> data, Vector<T> target, bool[] mask)
    {
        if (_fitnessFunc is not null)
            return _fitnessFunc(data, target, mask);

        var selected = WrapperScoringOps.Positions(mask);
        int nSelected = selected.Length;
        if (nSelected == 0) return NumOps.FromDouble(double.NegativeInfinity);

        // Mean |Pearson r| with the target, penalized for selecting more than the requested number of features
        double penalty = nSelected > _nFeaturesToSelect ? 0.1 * (nSelected - _nFeaturesToSelect) : 0;
        return NumOps.Subtract(WrapperScoringOps.MeanAbsTargetCorrelation(data, target, selected), NumOps.FromDouble(penalty));
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
            throw new InvalidOperationException("GreyWolfFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GreyWolfFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GreyWolfFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
