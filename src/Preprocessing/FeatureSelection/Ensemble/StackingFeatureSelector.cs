using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Stacking-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a stacking ensemble approach where multiple base feature scorers are
/// combined using a meta-learner to produce final feature rankings.
/// </para>
/// <para><b>For Beginners:</b> Stacking combines multiple ways of measuring
/// feature importance (like correlation, variance, and mutual information).
/// A second layer then learns how to best combine these measures, giving you
/// more reliable feature selection than any single method alone.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StackingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _stackedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? StackedScores => _stackedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public StackingFeatureSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "StackingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Base scorers: correlation, variance, mutual information approximation
        var correlationScores = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y));
        var varianceScores = engine.TensorDivideScalar(FeatureScoringTensorOps.ColumnSumSquaredDeviations(X),
            NumOps.FromDouble(n - 1));
        var miScores = ComputeMIScores(X, y, n, p);

        // Normalize scores to [0, 1]
        correlationScores = NormalizeScores(correlationScores);
        varianceScores = NormalizeScores(varianceScores);
        miScores = NormalizeScores(miScores);

        // Meta-learner: weighted combination (learn weights from data)
        // Use correlation between base scores and target variance explained
        T wCorr = ComputeMetaWeight(X, y, correlationScores, n, p);
        T wVar = ComputeMetaWeight(X, y, varianceScores, n, p);
        T wMI = ComputeMetaWeight(X, y, miScores, n, p);

        T totalWeight = NumOps.Add(NumOps.Add(NumOps.Add(wCorr, wVar), wMI), PreprocessingTensorOps.Tolerance<T>(1e-10));

        // Combine scores
        var stackedScores = engine.TensorAdd(engine.TensorAdd(
            engine.TensorMultiplyScalar(correlationScores, NumOps.Divide(wCorr, totalWeight)),
            engine.TensorMultiplyScalar(varianceScores, NumOps.Divide(wVar, totalWeight))),
            engine.TensorMultiplyScalar(miScores, NumOps.Divide(wMI, totalWeight)));
        _stackedScores = stackedScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(stackedScores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeMIScores(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        // sqrt(n) + 1 equal-width bins (width (max - min) / bins + 1e-10); MI in bits, clamped at 0
        int nBins = (int)Math.Sqrt(n) + 1;
        var xBins = PreprocessingTensorOps.EqualWidthBinOneHot(X, nBins, out _);
        var yBins = PreprocessingTensorOps.EqualWidthBinOneHot(Engine.Reshape(y, new[] { n, 1 }), nBins, out _);
        return Engine.Reshape(PreprocessingTensorOps.HistogramMutualInformation(xBins, yBins, nBins), new[] { p });
    }

    private Tensor<T> NormalizeScores(Tensor<T> scores)
    {
        var shifted = Engine.TensorSubtractScalar(scores, Engine.TensorMinValue(scores));
        return Engine.TensorDivideScalar(shifted, NumOps.Add(Engine.TensorMaxValue(shifted), PreprocessingTensorOps.Tolerance<T>(1e-10)));
    }

    private T ComputeMetaWeight(Tensor<T> X, Tensor<T> y, Tensor<T> scores, int n, int p)
    {
        // Weight based on how well the top-scored features explain target variance
        var engine = Engine;
        var topFeatures = PreprocessingTensorOps.StableOrder(scores, descending: true).Take(Math.Min(5, p)).ToArray();
        var top = new Tensor<int>(topFeatures, new[] { topFeatures.Length });
        var topScores = engine.TensorIndexSelect(scores, top, 0);

        T ssTot = FeatureScoringTensorOps.SumSquaredDeviations(y);

        // Simple weighted average prediction (unnormalized when the weights sum to at most 1e-10)
        var pred = engine.Reshape(engine.TensorMatMul(engine.TensorIndexSelect(X, top, 1),
            engine.Reshape(topScores, new[] { topFeatures.Length, 1 })), new[] { n });
        T weightSum = engine.TensorSum(topScores);
        if (PreprocessingTensorOps.IsAbove(weightSum, PreprocessingTensorOps.Tolerance<T>(1e-10)))
            pred = engine.TensorDivideScalar(pred, weightSum);
        T ssRes = engine.TensorSumOfSquares(engine.TensorSubtract(y, pred));

        return PreprocessingTensorOps.IsAbove(ssTot, PreprocessingTensorOps.Tolerance<T>(1e-10))
            ? PreprocessingTensorOps.ScalarMax(NumOps.Zero, NumOps.Subtract(NumOps.One, NumOps.Divide(ssRes, ssTot)))
            : NumOps.Zero;
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
            throw new InvalidOperationException("StackingFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("StackingFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("StackingFeatureSelector has not been fitted.");

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
