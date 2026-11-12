using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Neural;

/// <summary>
/// Attention Flow based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features using attention flow analysis, which tracks how
/// attention propagates through feature interactions.
/// </para>
/// <para><b>For Beginners:</b> Attention flow measures how "attention" spreads
/// from the output back to input features through a graph of feature interactions.
/// Features that receive more attention flow are considered more important.
/// This captures both direct and indirect feature contributions.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AttentionFlowSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;

    private Vector<T>? _attentionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NIterations => _nIterations;
    public Vector<T>? AttentionScores => _attentionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AttentionFlowSelector(
        int nFeaturesToSelect = 10,
        int nIterations = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AttentionFlowSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Build attention matrix from feature |correlations| (1 on the diagonal)
        var eye = engine.TensorEye<T>(p);
        var correlations = engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(X, X));
        var attentionMatrix = engine.TensorAdd(engine.TensorMultiply(correlations, FeatureSelectionTensorOps.OneMinus(eye)), eye);

        // Softmax normalize rows
        var rowMax = engine.TensorMax(engine.ReduceMax(attentionMatrix, new[] { 1 }, true), NumOps.Zero);
        var exponentials = engine.TensorExp(engine.TensorSubtract(attentionMatrix, rowMax));
        attentionMatrix = engine.TensorDivide(exponentials, engine.ReduceSum(exponentials, new[] { 1 }, true));

        // Initial attention from target |correlation|, as a [1, p] row
        var attention = engine.Reshape(engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(X, y)), new[] { 1, p });

        // Propagate attention flow: new[j1] = sum_j2 A[j2, j1] * attention[j2], normalized when the sum is above 1e-10
        T threshold = NumOps.FromDouble(1e-10);
        var one = FeatureSelectionTensorOps.Filled(new[] { 1, 1 }, NumOps.One);
        for (int iter = 0; iter < _nIterations; iter++)
        {
            var newAttention = engine.TensorMatMul(attention, attentionMatrix);
            var sum = engine.ReduceSum(newAttention, new[] { 1 }, true);
            var divisor = engine.TensorWhere(engine.TensorGreaterThan(sum, threshold), sum, one);
            attention = engine.TensorDivide(newAttention, engine.TensorBroadcastTo(divisor, new[] { 1, p }));
        }

        var attentionScores = engine.Reshape(attention, new[] { p });
        _attentionScores = attentionScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(attentionScores, numToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("AttentionFlowSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AttentionFlowSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AttentionFlowSelector has not been fitted.");

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
