using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Biaffine span scorer per Yu et al., ACL 2020, "Named Entity Recognition as Dependency Parsing"
/// (arXiv:2005.07150). Scores every <c>(start, end)</c> span against every entity category in one
/// pass.
/// </summary>
/// <remarks>
/// <para><b>Scoring function (paper §2):</b></para>
/// <code>
///   r_m(i) = h_s(i)^T . U_m . h_e(i) + W_m . (h_s(i) (+) h_e(i)) + b_m
/// </code>
/// <para>where <c>U_m</c> is a <c>d x c x d</c> tensor, <c>W_m</c> is <c>2d x c</c>, <c>h_s</c>
/// and <c>h_e</c> come from two SEPARATE feed-forward networks over the encoder output, and
/// <c>(+)</c> is concatenation. The bilinear term is what couples the start and end boundaries;
/// a plain MLP over concatenated endpoints (the additive term alone) cannot represent it, which
/// is precisely the paper's point.</para>
/// <para><b>Shapes:</b> input <c>[B, S, D]</c> token representations; output
/// <c>[B, S * S, C]</c> — the flattened <c>S x S</c> span grid scored over <c>C</c> categories,
/// with row-major index <c>start * S + end</c>. The grid is kept rank-3 so it composes with the
/// rest of the layer stack; consumers reshape to <c>[S, S, C]</c> and mask the lower triangle,
/// since the paper only considers spans with <c>start &lt;= end</c>.</para>
/// <para><b>Gradient tracking:</b> every operation goes through <c>IEngine</c> so the tape records
/// the whole scorer automatically. The bilinear term is evaluated as
/// <c>(H_s . U_c) . H_e^T</c> per category using batched matmuls rather than an explicit triple
/// loop, so there is no scalar arithmetic on the gradient path.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
// Finite-difference gradchecks require a stationary forward function. Disable dropout only in the
// generated fixture while preserving the paper-faithful 0.2 production default below.
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 4, 3, (AiDotNet.Interfaces.IActivationFunction<double>?)null, 0.0, 2")]
// Exactly the two ranks ForwardTraced admits - "expects rank-2 [S, D] or rank-3 [B, S, D]" - and it
// really does accept both: the rank-2 path promotes to a single-element batch and squeezes the batch
// axis back off the result. Declared as two layouts rather than one BatchOptional layout, because the
// OUTPUT roles are not the input roles, so nothing here could be derived by matching them anyway.
//
// The output's middle axis is the flattened S x S span grid, indexed `start * S + end`. That is not a
// sequence position and not a feature, so it takes Other - the escape hatch, used here for exactly what
// it is documented for: a real axis whose role is genuinely model-specific.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Classes,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class BiaffineSpanScorerLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Read off the two returns at the end of <c>ForwardTraced</c>:
    /// <c>Engine.Reshape(summed, [S * S, C])</c> unbatched and <c>[B, S * S, C]</c> batched, which the
    /// class docs state as "input <c>[B, S, D]</c> ... output <c>[B, S * S, C]</c>".
    /// </para>
    /// <para>
    /// <c>S * S</c> is <c>Product(Time, Time)</c> and not an approximation: a product resolves each named
    /// source against the input, so naming the sequence axis twice multiplies its size by itself. Writing
    /// it as <c>Scaled</c> would be impossible - the factor is the sequence length, which is not known
    /// until an input arrives - and writing it as <c>Unknown</c> would throw away a shape that is in fact
    /// completely determined.
    /// </para>
    /// <para>
    /// The last axis is <c>Fixed(_numCategories)</c>, the entity-category count the constructor takes;
    /// the feature width <c>D</c> does not survive, because the boundary FFNNs project it to
    /// <c>_spanDim</c> and the biaffine form contracts it away entirely.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3) || _numCategories <= 0) return null;

        var spans = new OutputAxisContract(
            TensorAxis.Other, AxisRelation.Product(TensorAxis.Time, TensorAxis.Time));
        var categories = new OutputAxisContract(
            TensorAxis.Classes, AxisRelation.Fixed(_numCategories));

        return inputRank == 2
            ? new[] { spans, categories }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                spans, categories,
            };
    }

    private readonly int _inputDim;
    private readonly int _spanDim;
    private readonly int _numCategories;

    /// <summary>
    /// Start-boundary FFNN, <c>ffnnDepth</c> layers deep. Separate weights from the end FFNN so a
    /// token is represented differently depending on which end of a span it occupies.
    /// </summary>
    private readonly DenseLayer<T>[] _startFfnn;
    private readonly DenseLayer<T>[] _endFfnn;

    /// <summary>Layers per boundary FFNN; the reference uses 2.</summary>
    private readonly int _ffnnDepth;

    /// <summary>Dropout on the boundary FFNN outputs; Yu et al. Table 1 specifies 0.2.</summary>
    private readonly double _ffnnDropout;
    private readonly DropoutLayer<T>? _startDropout;
    private readonly DropoutLayer<T>? _endDropout;

    /// <summary>Bilinear tensor U, stored as [C, d, d].</summary>
    private Tensor<T> _bilinear;

    /// <summary>Additive weight W over the concatenated endpoints, stored as [2d, C].</summary>
    private Tensor<T> _additive;

    /// <summary>Per-category bias b, stored as [C].</summary>
    private Tensor<T> _bias;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new biaffine span scorer.
    /// </summary>
    /// <param name="inputDim">Encoder output width <c>D</c>.</param>
    /// <param name="spanDim">
    /// Width <c>d</c> of the start/end FFNNs. The paper's "FFNN size" is 150.
    /// </param>
    /// <param name="numCategories">
    /// Number of entity categories <c>C</c>, including the non-entity class — Biaffine-NER
    /// classifies every span rather than sampling negatives.
    /// </param>
    /// <param name="activation">
    /// Activation for the two boundary FFNNs. Defaults to ReLU.
    /// </param>
    public BiaffineSpanScorerLayer(
        [LayerState] int inputDim,
        [LayerState] int spanDim,
        [LayerState] int numCategories,
        IActivationFunction<T>? activation = null,
        [LayerState] double ffnnDropout = 0.2,
        [LayerState] int ffnnDepth = 2)
        : base(new[] { -1, -1, inputDim }, new[] { -1, -1, numCategories })
    {
        if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (spanDim <= 0) throw new ArgumentOutOfRangeException(nameof(spanDim));
        if (numCategories <= 0) throw new ArgumentOutOfRangeException(nameof(numCategories));

        _inputDim = inputDim;
        _spanDim = spanDim;
        _numCategories = numCategories;

        var act = activation ?? new ReLUActivation<T>();

        // Two SEPARATE FFNNs, per the paper: the same token gets a different representation
        // depending on whether it is acting as a span start or a span end.
        // ffnn_depth layers per boundary, matching the reference configuration
        // (juntaoy/biaffine-ner, experiments.conf: ffnn_size = 150, ffnn_depth = 2).
        int depth = ffnnDepth > 0 ? ffnnDepth : 1;
        _ffnnDepth = depth;
        _startFfnn = new DenseLayer<T>[depth];
        _endFfnn = new DenseLayer<T>[depth];
        for (int i = 0; i < depth; i++)
        {
            _startFfnn[i] = new DenseLayer<T>(spanDim, act);
            _endFfnn[i] = new DenseLayer<T>(spanDim, act);
            RegisterSubLayer(_startFfnn[i]);
            RegisterSubLayer(_endFfnn[i]);
        }

        // Yu et al. Table 1 lists an FFNN dropout of 0.2. Without it the two boundary
        // representations are free to co-adapt, which is exactly what the separate-FFNN design
        // exists to prevent. Disabled automatically at inference like any dropout.
        _ffnnDropout = ffnnDropout;
        if (ffnnDropout > 0.0)
        {
            _startDropout = new DropoutLayer<T>(ffnnDropout);
            _endDropout = new DropoutLayer<T>(ffnnDropout);
            RegisterSubLayer(_startDropout);
            RegisterSubLayer(_endDropout);
        }

        _bilinear = new Tensor<T>(new[] { numCategories, spanDim, spanDim });
        _additive = new Tensor<T>(new[] { 2 * spanDim, numCategories });
        _bias = new Tensor<T>(new[] { numCategories });

        // The biaffine tensor starts at ZERO, matching the reference implementation
        // (juntaoy/biaffine-ner, util.py: bilinear_map is created with tf.zeros_initializer();
        // only the boundary FFNNs get a random init there). Every span's score therefore begins as
        // the additive term plus bias, and the interaction is learned from there.
        for (int i = 0; i < _bilinear.Length; i++) _bilinear[i] = NumOps.Zero;
        InitializeParameter(_additive, 2 * spanDim);

        // Register so the tape and ParameterBuffer see these tensors. Omitting this is silent:
        // GetTrainableParameters() still hands the optimizer all three tensors, so nothing
        // throws and the parameter count looks right — but the engine never marks them
        // persistent, no gradient is accumulated for them, and the biaffine term this layer
        // exists to compute (Yu et al., ACL 2020 §3) never learns. Measured before this fix:
        // BiaffineNER's memorization loss sat at 14.278996 for all 15 steps, and raising the
        // learning rate 100x (1e-5 -> 1e-3) moved it only in the sixth decimal — the residual
        // motion came entirely from the two boundary FFNNs, which ARE registered as sub-layers.
        RegisterTrainableParameter(_bilinear, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_additive, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_bias, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Materializes the lazily-allocated boundary FFNNs from this layer's known geometry,
    /// without executing them.
    /// </summary>
    /// <remarks>
    /// Both FFNNs consume the encoder width, so both resolve against <c>[1, 1, inputDim]</c>.
    /// Guarded by <c>IsShapeResolved</c>, so this is a no-op once the layer has run.
    /// </remarks>
    private void ResolveChildShapes()
    {
        // Only the first layer of each stack sees the encoder width; the rest are spanDim -> spanDim.
        for (int i = 0; i < _startFfnn.Length; i++)
        {
            int width = i == 0 ? _inputDim : _spanDim;
            if (!_startFfnn[i].IsShapeResolved) _startFfnn[i].ResolveFromShape(new[] { 1, 1, width });
            if (!_endFfnn[i].IsShapeResolved) _endFfnn[i].ResolveFromShape(new[] { 1, 1, width });
        }
    }

    private static long SumParameterCounts(DenseLayer<T>[] layers)
    {
        long total = 0;
        foreach (var layer in layers) total += layer.ParameterCount;
        return total;
    }

    private void InitializeParameter(Tensor<T> tensor, int fanIn)
    {
        // Xavier/Glorot scale; deterministic so repeated construction is reproducible.
        double scale = Math.Sqrt(6.0 / (fanIn + _numCategories));
        var rng = new Random(_inputDim * 31 + _spanDim * 17 + _numCategories);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.FromDouble(((rng.NextDouble() * 2.0) - 1.0) * scale);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // The NER stack passes unbatched [S, D] token representations, so accept rank 2 by
        // promoting to a single-element batch and squeezing the batch axis back off the result.
        // Rank 3 [B, S, D] is passed through unchanged.
        bool unbatched = input.Shape.Length == 2;
        if (unbatched)
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);

        if (input.Shape.Length != 3)
            throw new ArgumentException($"BiaffineSpanScorerLayer expects rank-2 [S, D] or rank-3 [B, S, D], got rank {input.Shape.Length}.", nameof(input));

        int B = input.Shape[0];
        int S = input.Shape[1];
        int D = input.Shape[2];

        if (D != _inputDim)
            throw new ArgumentException($"BiaffineSpanScorerLayer was configured for inputDim={_inputDim} but got D={D}.", nameof(input));

        // Separate start/end boundary representations: [B, S, d].
        var hs = input;
        foreach (var layer in _startFfnn) hs = layer.Forward(hs);

        var he = input;
        foreach (var layer in _endFfnn) he = layer.Forward(he);

        if (_startDropout is not null && _endDropout is not null)
        {
            hs = _startDropout.Forward(hs);
            he = _endDropout.Forward(he);
        }

        int d = _spanDim;
        int C = _numCategories;

        // Bilinear term, one category at a time:
        //     scores_c = (H_s . U_c) . H_e^T      -> [B, S, S]
        // Both steps are batched matmuls so the tape records them.
        var heT = Engine.TensorPermute(he, new[] { 0, 2, 1 });          // [B, d, S]
        var categoryScores = new Tensor<T>[C];

        for (int c = 0; c < C; c++)
        {
            // Slice U_c as [d, d] and broadcast it across the batch.
            var uc = Engine.TensorSliceAxis(_bilinear, axis: 0, index: c);   // [d, d]
            var ucBatched = Engine.Reshape(uc, [1, d, d]);
            var ucAll = ucBatched;
            if (B > 1) ucAll = Engine.TensorBroadcastTo(ucBatched, [B, d, d]);

            var hsU = Engine.TensorBatchMatMul<T>(hs, ucAll);        // [B, S, d]
            categoryScores[c] = Engine.TensorBatchMatMul<T>(hsU, heT); // [B, S, S]
        }

        // Additive term W . (h_s (+) h_e) + b. Splitting W into its start and end halves lets
        // this be two matmuls plus a broadcast add, instead of materializing the [S*S, 2d]
        // concatenation:
        //     W . (h_s (+) h_e) = W_start . h_s + W_end . h_e
        var wStart = Engine.TensorNarrow(_additive, dim: 0, start: 0, length: d);      // [d, C]
        var wEnd = Engine.TensorNarrow(_additive, dim: 0, start: d, length: d);        // [d, C]

        var startTerm = Engine.TensorBatchMatMul<T>(hs, BroadcastMatrix(wStart, B, d, C));     // [B, S, C]
        var endTerm = Engine.TensorBatchMatMul<T>(he, BroadcastMatrix(wEnd, B, d, C));         // [B, S, C]

        // Assemble bilinear(i,j,c) + startTerm(i,c) + endTerm(j,c) + b(c) entirely through
        // Engine ops. Doing this with scalar indexers would detach the result from the tape and
        // silently freeze every parameter feeding it.

        // Stack the per-category [B, S, S] grids directly into [B, S, S, C]. This is a
        // new category axis, not a concatenation of pre-existing singleton axes. Besides matching
        // that geometry exactly, TensorStack writes every output slot; building singleton views and
        // concatenating them could leave arena-rented output storage partially unwritten, allowing
        // stale NaN/Inf values from an earlier tensor to leak into otherwise-finite initial logits.
        var bilinearGrid = Engine.TensorStack(categoryScores, axis: 3);   // [B, S, S, C]

        // startTerm depends on the START index, so it repeats along j; endTerm depends on the
        // END index, so it repeats along i.
        var startGrid = Engine.TensorBroadcastTo(
            Engine.Reshape(startTerm, [B, S, 1, C]), [B, S, S, C]);
        var endGrid = Engine.TensorBroadcastTo(
            Engine.Reshape(endTerm, [B, 1, S, C]), [B, S, S, C]);
        var biasGrid = Engine.TensorBroadcastTo(
            Engine.Reshape(_bias, [1, 1, 1, C]), [B, S, S, C]);

        var summed = Engine.TensorAdd(
            Engine.TensorAdd(bilinearGrid, startGrid),
            Engine.TensorAdd(endGrid, biasGrid));

        return unbatched
            ? Engine.Reshape(summed, [S * S, C])
            : Engine.Reshape(summed, [B, S * S, C]);
    }

    private Tensor<T> BroadcastMatrix(Tensor<T> matrix, int batch, int rows, int cols)
    {
        var reshaped = Engine.Reshape(matrix, [1, rows, cols]);
        return batch > 1 ? Engine.TensorBroadcastTo(reshaped, [batch, rows, cols]) : reshaped;
    }

    /// <summary>Concatenates a boundary stack's parameters in layer order.</summary>
    private static Vector<T> StackParameters(DenseLayer<T>[] layers)
    {
        int total = 0;
        foreach (var layer in layers) total += layer.GetParameters().Length;

        var flat = new Vector<T>(total);
        int k = 0;
        foreach (var layer in layers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) flat[k++] = p[i];
        }
        return flat;
    }

    /// <summary>Distributes a flat slice back across a boundary stack, in the same order.</summary>
    private static void SetStackParameters(DenseLayer<T>[] layers, Vector<T> source, ref int offset)
    {
        foreach (var layer in layers)
        {
            int count = layer.GetParameters().Length;
            var slice = new Vector<T>(count);
            for (int i = 0; i < count; i++) slice[i] = source[offset++];
            layer.SetParameters(slice);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Includes the boundary FFNNs' tensors as well as this layer's own, because the base
    /// implementation does not recurse into registered sub-layers.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
    {
        var result = new List<Tensor<T>>();
        foreach (var layer in _startFfnn) result.AddRange(layer.GetTrainableParameters());
        foreach (var layer in _endFfnn) result.AddRange(layer.GetTrainableParameters());
        result.Add(_bilinear);
        result.Add(_additive);
        result.Add(_bias);
        return result;
    }

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        int startCount = 0, endCount = 0;
        foreach (var layer in _startFfnn) startCount += layer.GetTrainableParameters().Count;
        foreach (var layer in _endFfnn) endCount += layer.GetTrainableParameters().Count;
        int expected = startCount + endCount + 3;

        if (parameters.Count != expected)
            throw new ArgumentException($"Expected {expected} trainable tensors, got {parameters.Count}.", nameof(parameters));

        int cursor = 0;
        foreach (var layer in _startFfnn)
        {
            int n = layer.GetTrainableParameters().Count;
            layer.SetTrainableParameters(parameters.Skip(cursor).Take(n).ToList());
            cursor += n;
        }
        foreach (var layer in _endFfnn)
        {
            int n = layer.GetTrainableParameters().Count;
            layer.SetTrainableParameters(parameters.Skip(cursor).Take(n).ToList());
            cursor += n;
        }

        int at = cursor;
        _bilinear = parameters[at];
        _additive = parameters[at + 1];
        _bias = parameters[at + 2];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Publishes the geometry deserialization needs. Without it the helper falls back to
    /// defaults (spanDim = 150) and rebuilds a differently-shaped layer, which surfaces as a
    /// parameter-count mismatch on clone rather than as a missing-metadata error.
    /// </remarks>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputDim"] = _inputDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["SpanDim"] = _spanDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumCategories"] = _numCategories.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        foreach (var layer in _startFfnn) layer.ResetState();
        foreach (var layer in _endFfnn) layer.ResetState();
    }
}
