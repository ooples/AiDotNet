using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Correlation-Mutual Information Hybrid feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines linear correlation (Pearson) with mutual information (non-linear)
/// to capture both types of feature-target relationships.
/// </para>
/// <para><b>For Beginners:</b> Correlation finds straight-line relationships
/// while mutual information finds any type of pattern. A feature might be
/// strongly related to the target in a curved way that correlation misses.
/// This hybrid catches both types of useful features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CorrelationMIHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _correlationWeight;
    private readonly int _nBins;

    private Vector<T>? _correlationScores;
    private Vector<T>? _miScores;
    private Vector<T>? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CorrelationWeight => _correlationWeight;
    public Vector<T>? CorrelationScores => _correlationScores;
    public Vector<T>? MIScores => _miScores;
    public Vector<T>? HybridScores => _hybridScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CorrelationMIHybrid(
        int nFeaturesToSelect = 10,
        double correlationWeight = 0.5,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (correlationWeight < 0 || correlationWeight > 1)
            throw new ArgumentException("Correlation weight must be between 0 and 1.", nameof(correlationWeight));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _correlationWeight = correlationWeight;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CorrelationMIHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);

        // |Pearson r|, 0 when the feature or the target is (near) constant
        var (sxx, sxy, syy) = FilterStatisticTensorOps.CenteredCrossSums(x, FilterStatisticTensorOps.AsTensor(target));
        T tolerance = NumOps.FromDouble(1e-10);
        var correlation = NumOps.GreaterThan(syy, tolerance)
            ? Engine.TensorAbs(FilterStatisticTensorOps.GuardedDivide(
                sxy, Engine.TensorSqrt(Engine.TensorMultiplyScalar(sxx, syy)), 0, NumOps.Zero))
            : FilterStatisticTensorOps.FilledLike(sxx, NumOps.Zero);
        correlation = FeatureSelectionTensorOps.KeepOrFill(correlation, Engine.TensorGreaterThan(sxx, tolerance), NumOps.Zero);

        // plug-in mutual information of the binned feature with the target classes
        var mi = DiscreteInformation<T>.ForTarget(FeatureSelectionTensorOps.RangeBins(x, _nBins), _nBins, target).Relevance();
        _correlationScores = correlation.ToVector();
        _miScores = mi.ToVector();

        T weight = NumOps.FromDouble(_correlationWeight);
        var hybrid = Engine.TensorAdd(
            Engine.TensorMultiplyScalar(MinMaxScaled(correlation), weight),
            Engine.TensorMultiplyScalar(MinMaxScaled(mi), NumOps.Subtract(NumOps.One, weight)));
        _hybridScores = hybrid.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(hybrid, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // (s - min) / (max - min), or 0.5 everywhere when the range is below 1e-10
    private Tensor<T> MinMaxScaled(Tensor<T> scores)
    {
        if (scores.Length == 0)
            return scores;
        var asColumn = Engine.Reshape(scores, new[] { scores.Length, 1 });
        var range = Engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(asColumn), PreprocessingTensorOps.ColumnMin(asColumn));
        if (NumOps.LessThan(range.ToArray()[0], NumOps.FromDouble(1e-10)))
            return FilterStatisticTensorOps.FilledLike(scores, NumOps.FromDouble(0.5));
        return Engine.Reshape(FeatureSelectionTensorOps.RangeScaled(asColumn), new[] { scores.Length });
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
            throw new InvalidOperationException("CorrelationMIHybrid has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CorrelationMIHybrid does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CorrelationMIHybrid has not been fitted.");

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
