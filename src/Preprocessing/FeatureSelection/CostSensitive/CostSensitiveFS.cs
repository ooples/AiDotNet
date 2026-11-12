using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.CostSensitive;

/// <summary>
/// Cost-Sensitive Feature Selection considering feature acquisition costs.
/// </summary>
/// <remarks>
/// <para>
/// Cost-Sensitive Feature Selection balances the predictive value of features against
/// their acquisition costs. This is important in scenarios where collecting certain
/// features is expensive, time-consuming, or invasive.
/// </para>
/// <para><b>For Beginners:</b> Sometimes getting data for a feature costs money or time.
/// This selector picks features that give you the most predictive power for the least cost.
/// It's like shopping for ingredients - you want quality ingredients that don't break
/// your budget.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CostSensitiveFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _costWeight;
    private Vector<T>? _featureCosts;

    private Vector<T>? _relevanceScores;
    private Vector<T>? _costBenefitScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CostWeight => _costWeight;
    public Vector<T>? FeatureCosts => _featureCosts;
    public Vector<T>? RelevanceScores => _relevanceScores;
    public Vector<T>? CostBenefitScores => _costBenefitScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CostSensitiveFS(
        int nFeaturesToSelect = 10,
        double costWeight = 0.5,
        Vector<T>? featureCosts = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (costWeight < 0 || costWeight > 1)
            throw new ArgumentException("Cost weight must be between 0 and 1.", nameof(costWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _costWeight = costWeight;
        _featureCosts = featureCosts;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CostSensitiveFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var engine = Engine;

        // Compute relevance scores (|correlation| with target)
        var relevance = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
        _relevanceScores = relevance.ToVector();

        // Get or create feature costs
        var costs = _featureCosts ?? CreateUniformCosts(p);
        if (costs.Length != p)
            throw new ArgumentException("Feature costs length must match number of features.");
        _featureCosts = costs;

        // Normalize costs to [0, 1]
        var normalizedCosts = NormalizeCosts(Tensor<T>.FromVector(costs), p);

        // Compute cost-benefit scores
        // Higher relevance and lower cost = better score
        // Weighted combination: (1-w) * relevance - w * cost
        var costBenefit = engine.TensorSubtract(
            engine.TensorMultiplyScalar(relevance, NumOps.FromDouble(1 - _costWeight)),
            engine.TensorMultiplyScalar(normalizedCosts, NumOps.FromDouble(_costWeight)));
        _costBenefitScores = costBenefit.ToVector();

        // Select features with best cost-benefit scores
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(costBenefit, numToSelect);

        IsFitted = true;
    }

    public void Fit(Matrix<T> data, Vector<T> target, Vector<T> featureCosts)
    {
        if (featureCosts.Length != data.Columns)
            throw new ArgumentException("Feature costs must match number of columns.");

        _featureCosts = featureCosts;
        Fit(data, target);
    }

    private Vector<T> CreateUniformCosts(int p)
    {
        return PreprocessingTensorOps.Filled(new[] { p }, NumOps.One).ToVector();
    }

    private Tensor<T> NormalizeCosts(Tensor<T> costs, int p)
    {
        // (cost - min) / range, or 0.5 everywhere when the range is not above 1e-10
        if (p == 0)
            return costs;

        var engine = Engine;
        T min = engine.TensorMinValue(costs);
        T range = NumOps.Subtract(engine.TensorMaxValue(costs), min);
        if (!NumOps.GreaterThan(range, NumOps.FromDouble(1e-10)))
            return PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(0.5));

        return engine.TensorDivideScalar(engine.TensorSubtractScalar(costs, min), range);
    }

    public T GetTotalCost()
    {
        if (_selectedIndices is null || _featureCosts is null)
            return NumOps.Zero;

        return Engine.TensorSum(Tensor<T>.FromVector(PreprocessingTensorOps.SelectElements(_featureCosts, _selectedIndices)));
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
            throw new InvalidOperationException("CostSensitiveFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CostSensitiveFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CostSensitiveFS has not been fitted.");

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
