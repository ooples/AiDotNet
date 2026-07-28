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
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 4, 3")]
public class BiaffineSpanScorerLayer<T> : LayerBase<T>
{
    private readonly int _inputDim;
    private readonly int _spanDim;
    private readonly int _numCategories;

    private readonly DenseLayer<T> _startFfnn;
    private readonly DenseLayer<T> _endFfnn;

    /// <summary>Bilinear tensor U, stored as [C, d, d].</summary>
    private Tensor<T> _bilinear;

    /// <summary>Additive weight W over the concatenated endpoints, stored as [2d, C].</summary>
    private Tensor<T> _additive;

    /// <summary>Per-category bias b, stored as [C].</summary>
    private Tensor<T> _bias;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override long ParameterCount =>
        _startFfnn.ParameterCount + _endFfnn.ParameterCount +
        _bilinear.Length + _additive.Length + _bias.Length;

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
        int inputDim,
        int spanDim,
        int numCategories,
        IActivationFunction<T>? activation = null)
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
        _startFfnn = new DenseLayer<T>(spanDim, act);
        _endFfnn = new DenseLayer<T>(spanDim, act);
        RegisterSubLayer(_startFfnn);
        RegisterSubLayer(_endFfnn);

        _bilinear = new Tensor<T>(new[] { numCategories, spanDim, spanDim });
        _additive = new Tensor<T>(new[] { 2 * spanDim, numCategories });
        _bias = new Tensor<T>(new[] { numCategories });

        InitializeParameter(_bilinear, spanDim);
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
        if (!_startFfnn.IsShapeResolved) _startFfnn.ResolveFromShape(new[] { 1, 1, _inputDim });
        if (!_endFfnn.IsShapeResolved) _endFfnn.ResolveFromShape(new[] { 1, 1, _inputDim });
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
    public override Tensor<T> Forward(Tensor<T> input)
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
        var hs = _startFfnn.Forward(input);
        var he = _endFfnn.Forward(input);

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

        // Stack the per-category [B, S, S] grids into [B, S, S, C].
        for (int c = 0; c < C; c++)
            categoryScores[c] = Engine.Reshape(categoryScores[c], [B, S, S, 1]);

        var bilinearGrid = categoryScores.Length == 1
            ? categoryScores[0]
            : Engine.TensorConcatenate(categoryScores, axis: 3);          // [B, S, S, C]

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

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var start = _startFfnn.GetParameters();
        var end = _endFfnn.GetParameters();
        int total = start.Length + end.Length + _bilinear.Length + _additive.Length + _bias.Length;

        var flat = new Vector<T>(total);
        int k = 0;
        for (int i = 0; i < start.Length; i++) flat[k++] = start[i];
        for (int i = 0; i < end.Length; i++) flat[k++] = end[i];
        for (int i = 0; i < _bilinear.Length; i++) flat[k++] = _bilinear[i];
        for (int i = 0; i < _additive.Length; i++) flat[k++] = _additive[i];
        for (int i = 0; i < _bias.Length; i++) flat[k++] = _bias[i];

        return flat;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        // The boundary FFNNs allocate lazily on first Forward. Resolve their shapes from this
        // layer's known geometry first, or restoring a trained scorer into a fresh instance
        // compares the payload against a count of just the bilinear/additive/bias tensors.
        ResolveChildShapes();

        var start = _startFfnn.GetParameters();
        var end = _endFfnn.GetParameters();
        int expected = start.Length + end.Length + _bilinear.Length + _additive.Length + _bias.Length;

        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        int k = 0;
        var newStart = new Vector<T>(start.Length);
        for (int i = 0; i < start.Length; i++) newStart[i] = parameters[k++];
        _startFfnn.SetParameters(newStart);

        var newEnd = new Vector<T>(end.Length);
        for (int i = 0; i < end.Length; i++) newEnd[i] = parameters[k++];
        _endFfnn.SetParameters(newEnd);

        for (int i = 0; i < _bilinear.Length; i++) _bilinear[i] = parameters[k++];
        for (int i = 0; i < _additive.Length; i++) _additive[i] = parameters[k++];
        for (int i = 0; i < _bias.Length; i++) _bias[i] = parameters[k++];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Includes the boundary FFNNs' tensors as well as this layer's own, because the base
    /// implementation does not recurse into registered sub-layers.
    /// </remarks>
    public override IReadOnlyList<Tensor<T>> GetTrainableParameters()
    {
        var result = new List<Tensor<T>>();
        result.AddRange(_startFfnn.GetTrainableParameters());
        result.AddRange(_endFfnn.GetTrainableParameters());
        result.Add(_bilinear);
        result.Add(_additive);
        result.Add(_bias);
        return result;
    }

    /// <inheritdoc/>
    public override void SetTrainableParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        var startParams = _startFfnn.GetTrainableParameters();
        var endParams = _endFfnn.GetTrainableParameters();
        int expected = startParams.Count + endParams.Count + 3;

        if (parameters.Count != expected)
            throw new ArgumentException($"Expected {expected} trainable tensors, got {parameters.Count}.", nameof(parameters));

        _startFfnn.SetTrainableParameters(parameters.Take(startParams.Count).ToList());
        _endFfnn.SetTrainableParameters(parameters.Skip(startParams.Count).Take(endParams.Count).ToList());

        int at = startParams.Count + endParams.Count;
        _bilinear = parameters[at];
        _additive = parameters[at + 1];
        _bias = parameters[at + 2];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Tape-based autodiff drives the update through the optimizer and the registered trainable
    /// tensors, so there is no manual gradient step here. The boundary FFNNs are updated through
    /// their own registration as sub-layers.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
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
        _startFfnn.ResetState();
        _endFfnn.ResetState();
    }
}
