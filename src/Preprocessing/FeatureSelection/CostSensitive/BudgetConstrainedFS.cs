using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.CostSensitive;

/// <summary>
/// Budget-Constrained Feature Selection with a maximum cost budget.
/// </summary>
/// <remarks>
/// <para>
/// Budget-Constrained Feature Selection selects the most informative features
/// while staying within a specified cost budget. It uses a greedy approach
/// to maximize value per unit cost.
/// </para>
/// <para><b>For Beginners:</b> Imagine you have $100 to spend on data collection.
/// This selector picks the most valuable features you can afford within your budget.
/// It's like a knapsack problem - get the most value without exceeding your limit.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BudgetConstrainedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _budget;
    private readonly Vector<T>? _featureCosts;

    private Vector<T>? _relevanceScores;
    private Vector<T>? _valuePerCost;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private T _usedBudget;

    public double Budget => _budget;
    public Vector<T>? FeatureCosts => _featureCosts;
    public Vector<T>? RelevanceScores => _relevanceScores;
    public Vector<T>? ValuePerCost => _valuePerCost;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    public T UsedBudget => _usedBudget;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BudgetConstrainedFS(
        double budget = 100.0,
        Vector<T>? featureCosts = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (budget <= 0)
            throw new ArgumentException("Budget must be positive.", nameof(budget));

        _budget = budget;
        _featureCosts = featureCosts;
        _usedBudget = NumOps.Zero;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BudgetConstrainedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Get or create feature costs
        var costs = _featureCosts ?? CreateUnitCosts(p);
        if (costs.Length != p)
            throw new ArgumentException("Feature costs length must match number of features.");

        FitWithCosts(data, target, costs);
    }

    public void Fit(Matrix<T> data, Vector<T> target, Vector<T> featureCosts)
    {
        if (featureCosts.Length != data.Columns)
            throw new ArgumentException("Feature costs must match number of columns.");

        _nInputFeatures = data.Columns;
        FitWithCosts(data, target, featureCosts);
    }

    private void FitWithCosts(Matrix<T> data, Vector<T> target, Vector<T> costs)
    {
        var engine = Engine;
        int p = data.Columns;

        // Compute relevance scores (|correlation| with target)
        var relevance = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
        _relevanceScores = relevance.ToVector();

        // Compute value per cost (a (near-)free feature gets 1000x its relevance)
        var costTensor = Tensor<T>.FromVector(costs);
        var valuePerCost = engine.TensorWhere(engine.TensorGreaterThan(costTensor, PreprocessingTensorOps.Tolerance<T>(1e-10)),
            engine.TensorDivide(relevance, costTensor), engine.TensorMultiplyScalar(relevance, NumOps.FromDouble(1000)));
        _valuePerCost = valuePerCost.ToVector();

        // Greedy selection within budget, by value per cost: every prefix of the remaining features whose
        // running total stays within budget is taken at once, the first feature that does not fit is skipped
        var order = PreprocessingTensorOps.StableOrder(valuePerCost, descending: true);
        var sortedCosts = p == 0
            ? new Tensor<T>(new[] { 0 })
            : engine.TensorIndexSelect(costTensor, new Tensor<int>(order, new[] { p }), 0);
        var selected = new List<int>();
        T budget = NumOps.FromDouble(_budget);
        var used = new Tensor<T>(new[] { 1 });
        int start = 0;
        while (start < p)
        {
            int remaining = p - start;
            var rest = engine.TensorSlice(sortedCosts, new[] { start }, new[] { remaining });
            var running = engine.TensorSlice(
                engine.TensorCumSum(engine.TensorConcatenate(new[] { used, rest }, 0), 0), new[] { 1 }, new[] { remaining });
            var over = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(
                engine.TensorLessEqual(running, PreprocessingTensorOps.Filled(new[] { remaining }, budget))));
            int fit = over.Length == 0 ? remaining : over[0];
            selected.AddRange(order.Skip(start).Take(fit));
            if (fit > 0)
                used = engine.TensorSlice(running, new[] { fit - 1 }, new[] { 1 });
            start += fit + 1;
        }

        _usedBudget = engine.TensorSum(used);
        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private Vector<T> CreateUnitCosts(int p)
    {
        return PreprocessingTensorOps.Filled(new[] { p }, NumOps.One).ToVector();
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
            throw new InvalidOperationException("BudgetConstrainedFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BudgetConstrainedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BudgetConstrainedFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public T GetRemainingBudget()
    {
        return NumOps.Subtract(NumOps.FromDouble(_budget), _usedBudget);
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
