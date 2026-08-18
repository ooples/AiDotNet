using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Bidirectional Cloze Network (BCN) attention per Fang et al., CVPR 2021,
/// "Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text
/// Recognition" (arXiv:2103.06495).
/// </summary>
/// <remarks>
/// <para>This is ABINet's <b>Bidirectional</b> principle. Each position attends to every OTHER
/// position — both left and right — but is forbidden from attending to itself:</para>
/// <code>
///   M[i, j] = 0     when i != j
///   M[i, j] = -inf  when i == j
///   attention = softmax(QK^T / sqrt(d) + M) . V
/// </code>
/// <para>Blocking the diagonal is what makes it a <i>cloze</i>: the representation at position
/// <c>i</c> is built purely from its surrounding context, so predicting the character there
/// cannot trivially copy the character itself. That is precisely the information leak an
/// unmasked bidirectional attention would introduce, and why this cannot be expressed with the
/// causal (triangular) mask an ordinary attention layer offers — a causal mask would also remove
/// all right-hand context, collapsing the model back to unidirectional.</para>
/// <para><b>Gradient tracking:</b> the mask is a constant tensor added to the scores, and every
/// other step is an <c>IEngine</c> op, so the tape records the whole block.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, ChangesShape = false, ExpectedInputRank = 3, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 8", TestConstructorArgs = "8")]
// Roles are this layer's own, quoted from its guard: "expects rank-2 [S, D] or rank-3 [B, S, D]".
// S is the sequence position (Time), D the model width (Features). Batch is optional rather than a
// second declaration because ForwardTraced treats the rank-2 case as literally the same computation —
// it reshapes to [1, S, D], runs, and reshapes back — so there is no second relation to describe.
//
// NOT [ElementWiseShape], even though [LayerProperty(ChangesShape = false)] is accurate. That
// shorthand generates an identity across EVERY rank, and this layer throws for any rank but 2 and 3;
// it is also not element-wise in the sense the shorthand implies, since attention mixes across
// positions. Declaring the two ranks it really accepts, with named axes, is the honest form.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    BatchOptional = true, Direction = TensorLayoutDirection.Output,
    Note = "Attention re-weights positions in place: every axis survives at its input size.")]
[AutoParameters]
public partial class ClozeAttentionLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDim;

    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _output;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Initializes a new bidirectional cloze attention block.</summary>
    /// <param name="modelDim">Model width; input and output are both this wide.</param>
    public ClozeAttentionLayer(
        [LayerState] int modelDim)
        : base(new[] { -1, -1, modelDim }, new[] { -1, -1, modelDim })
    {
        if (modelDim <= 0) throw new ArgumentOutOfRangeException(nameof(modelDim));

        _modelDim = modelDim;
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();

        _query = new DenseLayer<T>(modelDim, identity);
        _key = new DenseLayer<T>(modelDim, identity);
        _value = new DenseLayer<T>(modelDim, identity);
        _output = new DenseLayer<T>(modelDim, identity);

        RegisterSubLayer(_query);
        RegisterSubLayer(_key);
        RegisterSubLayer(_value);
        RegisterSubLayer(_output);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);

        if (input.Shape.Length != 3)
            throw new ArgumentException($"ClozeAttentionLayer expects rank-2 [S, D] or rank-3 [B, S, D], got rank {input.Shape.Length}.", nameof(input));

        int B = input.Shape[0];
        int S = input.Shape[1];
        int D = input.Shape[2];

        if (D != _modelDim)
            throw new ArgumentException($"ClozeAttentionLayer was configured for modelDim={_modelDim} but got D={D}.", nameof(input));

        var q = _query.Forward(input);
        var k = _key.Forward(input);
        var v = _value.Forward(input);

        // scores = Q . K^T / sqrt(d)
        var kT = Engine.TensorPermute(k, new[] { 0, 2, 1 });
        var scores = Engine.TensorBatchMatMul<T>(q, kT);
        scores = Engine.TensorDivideScalar(scores, NumOps.FromDouble(Math.Sqrt(D)));

        // Cloze mask: a large negative on the diagonal only, so softmax drives self-attention to
        // zero while every other position stays available in both directions. Built as a
        // constant, so it contributes no gradient of its own.
        var mask = new Tensor<T>(new[] { B, S, S });
        T blocked = NumOps.FromDouble(-1e9);
        for (int b = 0; b < B; b++)
            for (int i = 0; i < S; i++)
                mask[b, i, i] = blocked;

        scores = Engine.TensorAdd(scores, mask);

        var weights = Engine.TensorSoftmax(scores, axis: scores.Shape.Length - 1);
        var attended = Engine.TensorBatchMatMul<T>(weights, v);
        var result = _output.Forward(attended);

        return unbatched ? Engine.Reshape(result, [S, D]) : result;
    }

    /// <summary>
    /// Materializes the lazily-allocated Q/K/V/output projections from the known model width,
    /// without executing them. Guarded by <c>IsShapeResolved</c>.
    /// </summary>
    private void ResolveChildShapes()
    {
        if (!_query.IsShapeResolved) _query.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_key.IsShapeResolved) _key.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_value.IsShapeResolved) _value.ResolveFromShape(new[] { 1, 1, _modelDim });
        if (!_output.IsShapeResolved) _output.ResolveFromShape(new[] { 1, 1, _modelDim });
    }

    // The four projections' tensors used to be listed here as this layer's own, on the stated
    // grounds that "LayerBase does not recurse into registered sub-layers". That is true of the base
    // GetTrainableParameters, which returns only this layer's own registrations, and false of the
    // walk ParameterCount, GetParameters and SetParameters are built from: it appends every
    // registered sub-layer that no declaration already covers, and its duplicate check compares
    // LAYER references, so it cannot tell that a child's tensors already arrived through the
    // parent's own list. Listing them here entered all eight TWICE. The failure the remark feared --
    // an empty trainable set beside a non-zero count -- is real, but it is what happens to a
    // composite that registers no sub-layer at all; these four are registered, so the base reaches
    // them.

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDim"] = _modelDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _query.ResetState();
        _key.ResetState();
        _value.ResetState();
        _output.ResetState();
    }
}
