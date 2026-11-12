using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The Triaffine nested-NER classifier (Yuan et al. 2022, "Fusing Heterogeneous Factors with Triaffine Mechanism for
/// Nested Named Entity Recognition", section 3). The deep triaffine transformation TriAff(u, v, w, W) = W x1 u' x2 w' x3 v'
/// with u' = [MLP(u); 1], v' = [MLP(v); 1] and w' = MLP(w) (Equations 1 to 3) scores, for every span (i, j) and label r,
/// each inside token k against the span's boundaries; a softmax over the inside tokens gives label-wise span
/// representations (Equations 5 to 7) and the auxiliary scores p_ijr (triaffine scoring, section 3.5). The top-m spans by
/// their best non-None auxiliary score then attend over each other with triaffine attention, and their final scores come
/// from the decomposed triaffine scoring (Equations 8 to 13).
/// </summary>
/// <remarks>
/// <para><b>Shapes:</b> input <c>[B, S, D]</c> (or <c>[S, D]</c>); output <c>[B, S * S, R]</c>, the span grid at index
/// <c>start * S + end</c>, R labels with None first. The top-m spans carry their final scores p^c. In training the other
/// spans carry their auxiliary scores, so the cross-entropy over the grid trains both predictions (the paper applies the
/// auxiliary loss to every span and the main loss to the top-m); in inference they are assigned None (section 3.6).
/// Spans with end before start score None.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.FeatureFusion)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 4, 8", TestConstructorArgs = "8, 4, 3, 3")]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Other, TensorAxis.Classes, Direction = TensorLayoutDirection.Output)]
public partial class TriaffineSpanLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _inputDim;
    private readonly int _dim;
    private readonly int _labels;
    private readonly int _topM;
    private readonly FullyConnectedLayer<T> _head;
    private readonly FullyConnectedLayer<T> _tail;
    private readonly FullyConnectedLayer<T> _inside;
    private readonly FullyConnectedLayer<T> _value;
    private readonly FullyConnectedLayer<T> _spanKey;
    private readonly BiasFreeLinearLayer<T> _attention;
    private readonly BiasFreeLinearLayer<T> _scoring;

    /// <summary>Creates the classifier.</summary>
    /// <param name="inputDim">Width of the token representations.</param>
    /// <param name="dim">Width d of the triaffine transformations (section 4.2: 256).</param>
    /// <param name="numLabels">Labels R, None first.</param>
    /// <param name="topM">Spans m retained for cross-span interaction (section 4.2: 30).</param>
    public TriaffineSpanLayer([LayerState] int inputDim, [LayerState] int dim, [LayerState] int numLabels, [LayerState] int topM)
        : base(new[] { -1, -1, inputDim }, new[] { -1, -1, numLabels })
    {
        if (inputDim <= 0 || dim <= 0 || numLabels <= 0 || topM <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
        _inputDim = inputDim; _dim = dim; _labels = numLabels; _topM = topM;
        var gelu = (IActivationFunction<T>)new ActivationFunctions.GELUActivation<T>();
        _head = new FullyConnectedLayer<T>(inputDim, dim, gelu);
        _tail = new FullyConnectedLayer<T>(inputDim, dim, gelu);
        _inside = new FullyConnectedLayer<T>(inputDim, dim, gelu);
        _value = new FullyConnectedLayer<T>(inputDim, dim, gelu);
        _spanKey = new FullyConnectedLayer<T>(dim, dim, gelu);
        // W_r and V_r, each [(d + 1) x d x (d + 1)] per label, initialized N(0, sigma^2) by the layer's default.
        _attention = new BiasFreeLinearLayer<T>(1, numLabels * (dim + 1) * dim * (dim + 1));
        _scoring = new BiasFreeLinearLayer<T>(1, numLabels * (dim + 1) * dim * (dim + 1));
        RegisterSubLayer(_head); RegisterSubLayer(_tail); RegisterSubLayer(_inside); RegisterSubLayer(_value);
        RegisterSubLayer(_spanKey); RegisterSubLayer(_attention); RegisterSubLayer(_scoring);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank is not (2 or 3)) return null;
        var spans = new OutputAxisContract(TensorAxis.Other, AxisRelation.Product(TensorAxis.Time, TensorAxis.Time));
        var classes = new OutputAxisContract(TensorAxis.Classes, AxisRelation.Fixed(_labels));
        return inputRank == 2 ? new[] { spans, classes }
            : new[] { new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)), spans, classes };
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        foreach (var layer in GetSubLayers()) layer.ResetState();
    }

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder, int count) =>
        Engine.Reshape(holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One }))), new[] { count });

    private Tensor<T> WithOne(Tensor<T> x) => Engine.TensorConcatenate(new[] { x, new Tensor<T>(new[] { x.Shape[0], 1 }, new Vector<T>(Enumerable.Repeat(NumOps.One, x.Shape[0]).ToArray())) }, 1);

    // A_r[i, j, b] = sum_{a, c} W_r[a, b, c] u'_i[a] v'_j[c] for heads u' [S, d+1] and tails v' [S, d+1]: [R, S, S, d].
    private Tensor<T> BoundaryContraction(Tensor<T> tensor, Tensor<T> head, Tensor<T> tail, int s)
    {
        int d = _dim, d1 = d + 1, r = _labels;
        var w = Engine.Reshape(tensor, new[] { r * d1, d * d1 });                                                  // [R*(d+1), d*(d+1)]
        var withHead = Engine.TensorMatMul(head, Engine.Reshape(Engine.TensorPermute(Engine.Reshape(w, new[] { r, d1, d * d1 }), new[] { 1, 0, 2 }), new[] { d1, r * d * d1 }));   // [S, R*d*(d+1)]
        var shaped = Engine.Reshape(Engine.TensorPermute(Engine.Reshape(withHead, new[] { s, r, d, d1 }), new[] { 1, 0, 2, 3 }), new[] { r * s * d, d1 });   // [R*S*d, d+1]
        var withTail = Engine.TensorMatMul(shaped, Engine.TensorTranspose(tail));                                 // [R*S*d, S]
        return Engine.TensorPermute(Engine.Reshape(withTail, new[] { r, s, d, s }), new[] { 0, 1, 3, 2 });     // [R, S(i), S(j), d]
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Shape.Length == 2;
        if (unbatched) input = Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1] });
        if (input.Shape.Length != 3 || input.Shape[2] != _inputDim) throw new ArgumentException($"TriaffineSpanLayer expects [B, S, {_inputDim}].", nameof(input));
        int b = input.Shape[0], s = input.Shape[1];
        var outputs = new Tensor<T>[b];
        for (int n = 0; n < b; n++)
            outputs[n] = Sentence(Engine.Reshape(Engine.TensorSlice(input, new[] { n, 0, 0 }, new[] { 1, s, _inputDim }), new[] { s, _inputDim }), s);
        var output = Engine.TensorStack(outputs, 0);
        return unbatched ? Engine.Reshape(output, new[] { s * s, _labels }) : output;
    }

    private Tensor<T> Sentence(Tensor<T> h, int s)
    {
        int d = _dim, r = _labels, d1 = d + 1, count = r * d1 * d * d1;
        var head = WithOne(_head.Forward(h));
        var tail = WithOne(_tail.Forward(h));
        var inside = _inside.Forward(h);                                                                           // [S, d]
        var values = _value.Forward(h);                                                                            // [S, d]
        var attention = Value(_attention, count);
        var scoring = Value(_scoring, count);
        // Equations 5 to 7: s_ijkr = A_r[i, j] . w'_k over the inside tokens i <= k <= j, softmax, weighted values.
        var aAttention = BoundaryContraction(attention, head, tail, s);                                            // [R, S, S, d]
        var scores = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(aAttention, new[] { r * s * s, d }), Engine.TensorTranspose(inside)), new[] { r, s, s, s });
        var insideMask = new Tensor<T>(new[] { 1, s, s, s });
        var blocked = NumOps.FromDouble(-1e9);
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
                for (int k = 0; k < s; k++)
                    if (j < i || k < i || k > j) insideMask[0, i, j, k] = (j < i && k == i) ? NumOps.Zero : blocked;
        var alpha = Engine.Softmax(Engine.TensorAdd(scores, Engine.TensorBroadcastTo(insideMask, new[] { r, s, s, s })), axis: 3);
        var spans = Engine.Reshape(Engine.TensorMatMul(Engine.Reshape(alpha, new[] { r * s * s, s }), values), new[] { r, s, s, d });   // h_ijr
        // Auxiliary scores p_ijr = TriAff(h_i, h_j, h_ijr, V_r), w being the 0-layer identity.
        var aScoring = BoundaryContraction(scoring, head, tail, s);                                                // [R, S, S, d]
        var auxiliary = Engine.ReduceSum(Engine.TensorMultiply(aScoring, spans), new[] { 3 }, keepDims: false);    // [R, S, S]
        // Select the top-m valid spans by their best non-None auxiliary score (section 3.6).
        var candidates = new List<(double Score, int I, int J)>();
        for (int i = 0; i < s; i++)
            for (int j = i; j < s; j++)
            {
                double best = double.NegativeInfinity;
                for (int label = 1; label < r; label++) best = Math.Max(best, NumOps.ToDouble(auxiliary[label, i, j]));
                candidates.Add((r > 1 ? best : 0.0, i, j));
            }
        var top = candidates.OrderByDescending(c => c.Score).Take(Math.Min(_topM, candidates.Count)).ToList();
        int m = top.Count;
        // Equations 8 to 13 over the retained spans: q_lgr = TriAff(h_i, h_j, h_g, W_r), beta = softmax over g, and the
        // final score p^c_lr = sum_g beta_lgr t_lgr with t_lgr = TriAff(h_i, h_j, h_g, V_r).
        var spanIndex = new Tensor<int>(new[] { m }, new Vector<int>(top.Select(c => c.I * s + c.J).ToArray()));
        Tensor<T> Retained(Tensor<T> perSpan, int width) =>
            Engine.Reshape(Engine.TensorIndexSelect(Engine.Reshape(Engine.TensorPermute(perSpan, new[] { 1, 2, 0, 3 }), new[] { s * s, r * width }), spanIndex, 0), new[] { m, r, width });
        var retainedSpans = Retained(spans, d);                                                                   // [m, R, d] h_{ig jg r}
        var retainedAttention = Retained(aAttention, d);                                                          // [m, R, d] A_r[i_l, j_l]
        var retainedScoring = Retained(aScoring, d);
        var keys = Engine.Reshape(_spanKey.Forward(Engine.Reshape(retainedSpans, new[] { m * r, d })), new[] { m, r, d });
        var byLabel = new[] { 1, 0, 2 };
        var q = Engine.BatchMatMul(Engine.TensorPermute(retainedAttention, byLabel), Engine.TensorPermute(Engine.TensorPermute(keys, byLabel), new[] { 0, 2, 1 }));     // [R, m, m]
        var beta = Engine.Softmax(q, axis: 2);
        var t = Engine.BatchMatMul(Engine.TensorPermute(retainedScoring, byLabel), Engine.TensorPermute(Engine.TensorPermute(retainedSpans, byLabel), new[] { 0, 2, 1 }));   // [R, m, m]
        var final = Engine.ReduceSum(Engine.TensorMultiply(beta, t), new[] { 2 }, keepDims: false);                // [R, m]
        // The grid: retained spans take p^c; the rest take p (training) or None (inference); end-before-start spans None.
        var noneScores = new Tensor<T>(new[] { s * s, r });
        for (int c = 0; c < s * s; c++)
            for (int label = 1; label < r; label++) noneScores[c, label] = NumOps.FromDouble(-1e4);
        var auxiliaryGrid = Engine.Reshape(Engine.TensorPermute(auxiliary, new[] { 1, 2, 0 }), new[] { s * s, r });
        var keep = new Tensor<T>(new[] { s * s, r });
        var fromFinal = new Tensor<T>(new[] { s * s, r });
        var retainedRow = new int[s * s];
        for (int c = 0; c < s * s; c++) retainedRow[c] = m;
        for (int l = 0; l < m; l++) retainedRow[top[l].I * s + top[l].J] = l;
        for (int i = 0; i < s; i++)
            for (int j = 0; j < s; j++)
            {
                int c = i * s + j;
                bool retained = retainedRow[c] < m, useAuxiliary = !retained && j >= i && IsTrainingMode;
                for (int label = 0; label < r; label++)
                {
                    keep[c, label] = useAuxiliary ? NumOps.One : NumOps.Zero;
                    fromFinal[c, label] = retained ? NumOps.One : NumOps.Zero;
                }
            }
        var finalTable = Engine.TensorConcatenate(new[] { Engine.TensorTranspose(final), new Tensor<T>(new[] { 1, r }) }, 0);   // [m + 1, R]
        var finalGrid = Engine.TensorIndexSelect(finalTable, new Tensor<int>(new[] { s * s }, new Vector<int>(retainedRow)), 0);
        var rest = Engine.TensorSubtract(Engine.TensorSubtract(new Tensor<T>(new[] { s * s, r }, new Vector<T>(Enumerable.Repeat(NumOps.One, s * s * r).ToArray())), keep), fromFinal);
        return Engine.TensorAdd(Engine.TensorAdd(Engine.TensorMultiply(finalGrid, fromFinal), Engine.TensorMultiply(auxiliaryGrid, keep)), Engine.TensorMultiply(noneScores, rest));
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var ci = System.Globalization.CultureInfo.InvariantCulture;
        metadata["InputDim"] = _inputDim.ToString(ci);
        metadata["Dim"] = _dim.ToString(ci);
        metadata["NumLabels"] = _labels.ToString(ci);
        metadata["TopM"] = _topM.ToString(ci);
        return metadata;
    }
}
