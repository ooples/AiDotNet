using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Foundations;

/// <summary>The decoder blocks of a Paraformer.</summary>
internal enum ParaformerDecoderKind
{
    /// <summary>FunASR's <c>ParaformerSANDecoder</c>: pre-norm self-attention, cross-attention and a ReLU feed-forward,
    /// each residual, with no causal mask.</summary>
    San,
    /// <summary>FunASR's <c>ParaformerSANMDecoder</c>: a feed-forward (with a norm on its hidden layer) feeding an FSMN
    /// memory (a depthwise convolution plus its input) added to the block input, then cross-attention; a last block of the
    /// feed-forward alone.</summary>
    Sanm,
}

/// <summary>The CIF weight predictor of a Paraformer.</summary>
internal enum CifPredictorKind
{
    /// <summary>FunASR's <c>CifPredictor</c>: a depthwise convolution added to its input, ReLU, a linear layer and a
    /// sigmoid.</summary>
    DepthwiseResidual,
    /// <summary>FunASR's <c>CifPredictorV2</c>: a full convolution, ReLU, a linear layer and a sigmoid.</summary>
    Convolution,
}

/// <summary>The sizes and training settings of a Paraformer's predictor, sampler and decoder.</summary>
internal sealed record ParaformerConfiguration
{
    /// <summary>The encoder's width, which the predictor and the decoder share.</summary>
    public int Width { get; init; } = 256;
    public int Heads { get; init; } = 4;
    public int FeedForward { get; init; } = 2048;
    public int DecoderLayers { get; init; } = 6;
    public ParaformerDecoderKind Decoder { get; init; } = ParaformerDecoderKind.San;
    public CifPredictorKind Predictor { get; init; } = CifPredictorKind.DepthwiseResidual;
    /// <summary>The predictor convolution's context on each side (<c>l_order</c>, <c>r_order</c>: 1).</summary>
    public int PredictorContext { get; init; } = 1;
    /// <summary>The SAN-M decoder's FSMN kernel (11).</summary>
    public int FsmnKernel { get; init; } = 11;
    /// <summary>Output vocabulary: blank 0, start 1 and end 2 included.</summary>
    public int VocabSize { get; init; } = 8404;
    /// <summary>The weight appended after the last frame at inference so a trailing partial token can fire (0.45).</summary>
    public double TailThreshold { get; init; } = 0.45;
    /// <summary>The glancing sampler's λ: the share of the first pass's wrong tokens replaced by target embeddings.</summary>
    public double SamplingRatio { get; init; } = 0.4;
    public double LabelSmoothing { get; init; } = 0.1;
    /// <summary>The CTC branch's weight on the encoder; zero for none.</summary>
    public double CtcWeight { get; init; } = 0.3;
    /// <summary>The MAE (token count) loss's weight.</summary>
    public double PredictorWeight { get; init; } = 1.0;
    /// <summary>Whether the end token is appended to every target, so the predictor counts it (<c>predictor_bias</c> 1).</summary>
    public bool AppendEndToken { get; init; }
    public int StartToken { get; init; } = 1;
    public int EndToken { get; init; } = 2;
    /// <summary>Whether the decoder and MAE losses are averaged over the target tokens rather than summed.</summary>
    public bool LengthNormalizedLoss { get; init; }
    /// <summary>The decoder's layer-norm epsilon (FunASR's LayerNorm: 1e-12).</summary>
    public double NormEpsilon { get; init; } = 1e-12;
}

/// <summary>
/// Paraformer's predictor, sampler and parallel decoder over an encoder's output (Gao et al. 2022, section 2; FunASR's
/// <c>Paraformer</c>, <c>CifPredictor</c>/<c>CifPredictorV2</c> and <c>ParaformerSANDecoder</c>/<c>ParaformerSANMDecoder</c>).
/// </summary>
/// <remarks>
/// <para>The predictor turns each encoder frame into a weight α in [0, 1]; continuous integrate-and-fire (CIF, Dong and Xu
/// 2020) integrates the frames, each weighted by its share of α, into one acoustic embedding per unit of accumulated
/// weight (threshold 1). Token n integrates the part of the cumulative weight lying in [n, n + 1), which is exactly what
/// the release's loop fires and stays differentiable in α. In training the weights are scaled to sum to the target length
/// (the release's form of the paper's dynamic threshold, Equation 1); at inference a tail weight is appended after the last
/// frame and the floor of the total is the token count.</para>
/// <para>The decoder reads the acoustic embeddings and the encoder output and predicts every token at once. In training
/// the glancing sampler (Equations 4 and 5) decodes once without gradients, counts the wrong tokens d, and replaces
/// ⌊λ d⌋ randomly chosen positions with the targets' embeddings (truncating as the release does; the paper writes ⌈λ d⌉)
/// before the second, trained pass. The loss is the label-smoothed decoder loss over every target token, plus the MAE
/// between the target length and Σα, plus the encoder's CTC loss when weighted (Equation 6 without MWER, which the release
/// does not train either).</para>
/// </remarks>
internal sealed class ParaformerNetwork<T> : ILayerAliasOwner<T>
{
    /// <inheritdoc />
    public void RebindLayerAliases(IReadOnlyList<ILayer<T>> previousLayers, IReadOnlyList<ILayer<T>> replacementLayers)
        => LayerAliasGraph.Rebind(this, previousLayers, replacementLayers);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly ParaformerConfiguration _o;

    internal sealed record SanBlock(LayerNormalizationLayer<T> Norm1, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output,
        LayerNormalizationLayer<T> Norm2, DenseLayer<T> CrossQuery, DenseLayer<T> CrossKey, DenseLayer<T> CrossValue, DenseLayer<T> CrossOutput,
        LayerNormalizationLayer<T> Norm3, DenseLayer<T> FeedForward1, DenseLayer<T> FeedForward2);

    public ParaformerNetwork(IEngine engine, List<LayerBase<T>> layers, ParaformerConfiguration configuration)
    {
        _engine = engine;
        _o = configuration;
        int d = _o.Width;
        if (d % _o.Heads != 0) throw new ArgumentException("The width must divide by the heads.", nameof(configuration));
        if (_o.DecoderLayers < 1) throw new ArgumentException("The decoder needs at least one block.", nameof(configuration));
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        DenseLayer<T> Linear(int outputs, bool bias = true) => Own(new DenseLayer<T>(outputs, identity, biasMode: bias ? BiasMode.Always : BiasMode.Never));
        LayerNormalizationLayer<T> Norm(int width) => Own(new LayerNormalizationLayer<T>(width, _o.NormEpsilon));

        int kernel = 2 * _o.PredictorContext + 1;
        PredictorConvolution = Own(new NormedConv1DLayer<T>(d, d, kernel, 1, 1, _o.Predictor == CifPredictorKind.DepthwiseResidual ? d : 1, 0, false,
            ConvolutionNormalization.None, useBias: true));
        PredictorOutput = Linear(1);
        Embedding = Own(new EmbeddingLayer<T>(_o.VocabSize, d));
        if (_o.Decoder == ParaformerDecoderKind.San)
        {
            for (int i = 0; i < _o.DecoderLayers; i++)
                SanBlocks.Add(new SanBlock(Norm(d), Linear(d), Linear(d), Linear(d), Linear(d), Norm(d), Linear(d), Linear(d), Linear(d), Linear(d),
                    Norm(d), Linear(_o.FeedForward), Linear(d)));
            AfterNorm = Norm(d);
        }
        else
        {
            SanmDecoder = new SanmDecoder<T>(engine, layers, d, _o.Heads, _o.FeedForward, _o.DecoderLayers, _o.FsmnKernel, _o.NormEpsilon);
            AfterNorm = SanmDecoder.AfterNorm;
        }
        Output = Linear(_o.VocabSize);
    }

    public ParaformerConfiguration Configuration => _o;
    public NormedConv1DLayer<T> PredictorConvolution { get; }
    public DenseLayer<T> PredictorOutput { get; }
    public EmbeddingLayer<T> Embedding { get; }
    public List<SanBlock> SanBlocks { get; } = new();
    /// <summary>The SAN-M decoder, or null for the SAN decoder.</summary>
    public SanmDecoder<T>? SanmDecoder { get; }
    public LayerNormalizationLayer<T> AfterNorm { get; }
    public DenseLayer<T> Output { get; }

    private Tensor<T> Convolve(NormedConv1DLayer<T> convolution, Tensor<T> x, int left, int right) =>
        SanmDecoder<T>.Convolve(_engine, convolution, x, left, right);

    /// <summary>The predictor's weights α <c>[1, T]</c> for the encoder output <c>[T, width]</c>.</summary>
    public Tensor<T> Alphas(Tensor<T> encoded)
    {
        var memory = Convolve(PredictorConvolution, encoded, _o.PredictorContext, _o.PredictorContext);
        if (_o.Predictor == CifPredictorKind.DepthwiseResidual) memory = _engine.TensorAdd(memory, encoded);
        var alphas = _engine.Sigmoid(PredictorOutput.Forward(_engine.ReLU(memory)));
        return _engine.Reshape(alphas, new[] { 1, encoded.Shape[0] });
    }

    /// <summary>
    /// CIF with threshold 1: <paramref name="tokens"/> acoustic embeddings <c>[n, width]</c>, token n integrating each frame
    /// by the share of its weight whose cumulative sum lies in [n, n + 1).
    /// </summary>
    public Tensor<T> Integrate(Tensor<T> encoded, Tensor<T> alphas, int tokens) => Integrate(_engine, encoded, alphas, tokens);

    /// <summary>CIF with threshold 1 (see the instance overload), for any encoder output <c>[T, width]</c> and weights
    /// <c>[1, T]</c>.</summary>
    internal static Tensor<T> Integrate(IEngine engine, Tensor<T> encoded, Tensor<T> alphas, int tokens)
    {
        int t = encoded.Shape[0];
        var inclusive = new Tensor<T>(new[] { t, t });
        for (int i = 0; i < t; i++) for (int j = i; j < t; j++) inclusive[i, j] = NumOps.One;
        var cumulative = engine.TensorMatMul(alphas, inclusive);                                        // [1, T]
        var previous = engine.TensorSubtract(cumulative, alphas);
        var ones = new Tensor<T>(new[] { tokens, 1 });
        for (int n = 0; n < tokens; n++) ones[n, 0] = NumOps.One;
        var upper = engine.TensorMatMul(ones, cumulative);                                             // [n, T]
        var lower = engine.TensorMatMul(ones, previous);
        var floor = new Tensor<T>(new[] { tokens, t });
        var ceiling = new Tensor<T>(new[] { tokens, t });
        for (int n = 0; n < tokens; n++)
            for (int j = 0; j < t; j++) { floor[n, j] = NumOps.FromDouble(n); ceiling[n, j] = NumOps.FromDouble(n + 1); }
        // min(c_t, n + 1) - max(c_{t-1}, n), clipped at zero.
        var top = engine.TensorSubtract(upper, engine.ReLU(engine.TensorSubtract(upper, ceiling)));
        var bottom = engine.TensorAdd(lower, engine.ReLU(engine.TensorSubtract(floor, lower)));
        var weights = engine.ReLU(engine.TensorSubtract(top, bottom));
        return engine.TensorMatMul(weights, encoded);
    }

    /// <summary>The inference token count and acoustic embeddings: the tail weight appended after a zero frame, the floor
    /// of the total weight (at least <paramref name="minimumTokens"/>).</summary>
    public Tensor<T>? AcousticEmbeddings(Tensor<T> encoded, int minimumTokens = 0)
    {
        var alphas = Alphas(encoded);
        if (_o.TailThreshold > 0)
        {
            var tail = new Tensor<T>(new[] { 1, 1 });
            tail[0, 0] = NumOps.FromDouble(_o.TailThreshold);
            alphas = _engine.TensorConcatenate(new[] { alphas, tail }, 1);
            encoded = _engine.TensorConcatenate(new[] { encoded, new Tensor<T>(new[] { 1, _o.Width }) }, 0);
        }
        double total = 0;
        for (int i = 0; i < alphas.Length; i++) total += NumOps.ToDouble(alphas.GetFlat(i));
        int tokens = Math.Max(minimumTokens, (int)Math.Floor(total + 1e-9));
        return tokens == 0 ? null : Integrate(encoded, alphas, tokens);
    }

    private Tensor<T> Attend(Tensor<T> q, Tensor<T> k, Tensor<T> v) => SanmDecoder<T>.Attend(_engine, q, k, v, _o.Heads);

    /// <summary>The decoder's logits <c>[n, VocabSize]</c> for decoder inputs <c>[n, width]</c> (acoustic or semantic
    /// embeddings) reading the encoder output <c>[T, width]</c>.</summary>
    public Tensor<T> Decode(Tensor<T> encoded, Tensor<T> embeddings) => Output.Forward(Hidden(encoded, embeddings));

    /// <summary>The decoder's hidden states <c>[n, width]</c> before the output layer (FunASR's <c>return_hidden</c>).</summary>
    public Tensor<T> Hidden(Tensor<T> encoded, Tensor<T> embeddings)
    {
        if (SanmDecoder is not null) return SanmDecoder.Forward(encoded, embeddings);
        var x = embeddings;
        foreach (var b in SanBlocks)
        {
            var h = b.Norm1.Forward(x);
            x = _engine.TensorAdd(x, b.Output.Forward(Attend(b.Query.Forward(h), b.Key.Forward(h), b.Value.Forward(h))));
            h = b.Norm2.Forward(x);
            x = _engine.TensorAdd(x, b.CrossOutput.Forward(Attend(b.CrossQuery.Forward(h), b.CrossKey.Forward(encoded), b.CrossValue.Forward(encoded))));
            x = _engine.TensorAdd(x, b.FeedForward2.Forward(_engine.ReLU(b.FeedForward1.Forward(b.Norm3.Forward(x)))));
        }
        return AfterNorm.Forward(x);
    }

    /// <summary>The targets' embeddings <c>[n, width]</c>: the token embedding, scaled by √width with sinusoidal
    /// positions for the SAN decoder (ESPnet's embed + PositionalEncoding), plain for the SAN-M decoder.</summary>
    public Tensor<T> TargetEmbeddings(IReadOnlyList<int> tokens)
    {
        int n = tokens.Count, d = _o.Width;
        var ids = new Tensor<T>(new[] { 1, n });
        for (int i = 0; i < n; i++) ids[0, i] = NumOps.FromDouble(tokens[i]);
        var x = _engine.Reshape(Embedding.Forward(ids), new[] { n, d });
        if (_o.Decoder == ParaformerDecoderKind.Sanm) return x;
        var positions = new Tensor<T>(new[] { n, d });
        for (int p = 0; p < n; p++)
            for (int k = 0; k < d; k += 2)
            {
                double angle = p * Math.Exp(-k * Math.Log(10000.0) / d);
                positions[p, k] = NumOps.FromDouble(Math.Sin(angle));
                if (k + 1 < d) positions[p, k + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(x, NumOps.FromDouble(Math.Sqrt(d))), positions);
    }

    /// <summary>Greedy decoding: the most likely token at each acoustic embedding, without blank, start and end.</summary>
    public List<int> Greedy(Tensor<T> encoded, int maxTokens)
    {
        var tokens = new List<int>();
        var acoustic = AcousticEmbeddings(encoded);
        if (acoustic is null) return tokens;
        var logits = Decode(encoded, acoustic);
        for (int n = 0; n < logits.Shape[0] && tokens.Count < maxTokens; n++)
        {
            int best = ArgMax(logits, n);
            if (best != 0 && best != _o.StartToken && best != _o.EndToken) tokens.Add(best);
        }
        return tokens;
    }

    private static int ArgMax(Tensor<T> logits, int row)
    {
        int best = 0;
        for (int v = 1; v < logits.Shape[1]; v++) if (NumOps.GreaterThan(logits[row, v], logits[row, best])) best = v;
        return best;
    }

    /// <summary>
    /// The training loss of one utterance (see the class remarks): (1 − w) L_att + w L_CTC + γ L_MAE, or L_att + γ L_MAE
    /// without a CTC loss.
    /// </summary>
    /// <param name="encoded">The encoder output <c>[T, width]</c>.</param>
    /// <param name="labels">The transcript's token ids.</param>
    /// <param name="random">The sampler's generator.</param>
    /// <param name="ctcLoss">The encoder's CTC loss, or null.</param>
    /// <param name="sample">Whether the glancing sampler runs (training); without it the decoder reads the acoustic
    /// embeddings alone, as at inference.</param>
    public Tensor<T> Loss(Tensor<T> encoded, IReadOnlyList<int> labels, Random random, Tensor<T>? ctcLoss, bool sample = true)
    {
        var targets = Targets(labels);
        int n = targets.Length;
        var alphas = Alphas(encoded);
        var sum = _engine.ReduceSum(alphas, new[] { 0, 1 }, keepDims: true);                           // [1, 1]
        var count = new Tensor<T>(new[] { 1, 1 });
        count[0, 0] = NumOps.FromDouble(n);
        var difference = _engine.TensorSubtract(count, sum);
        double normalizer = _o.LengthNormalizedLoss ? Math.Max(1, n) : 1;
        var mae = _engine.TensorMultiplyScalar(_engine.TensorAdd(_engine.ReLU(difference), _engine.ReLU(_engine.TensorNegate(difference))),
            NumOps.FromDouble(_o.PredictorWeight / normalizer));
        Tensor<T> total = mae;
        if (n > 0)
        {
            var acoustic = Integrate(encoded, ScaledAlphas(alphas, sum, n), n);
            var inputs = acoustic;
            if (sample && _o.SamplingRatio > 0)
            {
                int correct = 0;
                using (new Tensors.Engines.Autodiff.NoGradScope<T>())
                {
                    var first = Decode(encoded, acoustic);
                    for (int i = 0; i < n; i++) if (ArgMax(first, i) == targets[i]) correct++;
                }
                int replace = (int)((n - correct) * _o.SamplingRatio);
                if (replace > 0)
                {
                    var chosen = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(replace).ToArray();
                    var mask = new Tensor<T>(new[] { n, _o.Width });
                    var keep = new Tensor<T>(new[] { n, _o.Width });
                    for (int i = 0; i < n; i++) for (int k = 0; k < _o.Width; k++) keep[i, k] = NumOps.One;
                    foreach (int i in chosen) for (int k = 0; k < _o.Width; k++) { mask[i, k] = NumOps.One; keep[i, k] = NumOps.Zero; }
                    inputs = _engine.TensorAdd(_engine.TensorMultiply(acoustic, keep), _engine.TensorMultiply(TargetEmbeddings(targets), mask));
                }
            }
            var logProbabilities = _engine.TensorLogSoftmax(Decode(encoded, inputs), axis: -1);
            var attention = LabelSmoothedLoss(_engine, logProbabilities, targets, _o.LabelSmoothing, normalizer);
            double attentionWeight = ctcLoss is null ? 1 : 1 - _o.CtcWeight;
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(attention, NumOps.FromDouble(attentionWeight)));
        }
        if (ctcLoss is not null && _o.CtcWeight > 0)
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(_engine.Reshape(ctcLoss, new[] { 1, 1 }), NumOps.FromDouble(_o.CtcWeight)));
        return _engine.Reshape(total, new[] { 1 });
    }

    /// <summary>The decoder's targets: the labels, followed by the end token when it is appended.</summary>
    public int[] Targets(IReadOnlyList<int> labels) => _o.AppendEndToken ? labels.Concat(new[] { _o.EndToken }).ToArray() : labels.ToArray();

    /// <summary>The training acoustic embeddings <c>[tokens, width]</c>: the predictor's weights scaled to sum to
    /// <paramref name="tokens"/>, integrated.</summary>
    public Tensor<T> TrainingAcousticEmbeddings(Tensor<T> encoded, int tokens)
    {
        var alphas = Alphas(encoded);
        return Integrate(encoded, ScaledAlphas(alphas, _engine.ReduceSum(alphas, new[] { 0, 1 }, keepDims: true), tokens), tokens);
    }

    // Σα scaled to the target length.
    private Tensor<T> ScaledAlphas(Tensor<T> alphas, Tensor<T> sum, int tokens) =>
        _engine.TensorMultiplyScalar(_engine.TensorDivide(alphas, _engine.TensorTile(sum, new[] { 1, alphas.Shape[1] })), NumOps.FromDouble(tokens));

    /// <summary>ESPnet's <c>LabelSmoothingLoss</c> <c>[1, 1]</c>: KL(q || p) summed over the rows of the log-probabilities
    /// <c>[n, classes]</c>, q putting 1 − ε on the target and ε / (classes − 1) on every other class, divided by
    /// <paramref name="normalizer"/>.</summary>
    internal static Tensor<T> LabelSmoothedLoss(IEngine engine, Tensor<T> logProbabilities, IReadOnlyList<int> targets, double smoothing, double normalizer)
    {
        int n = targets.Count, v = logProbabilities.Shape[1];
        double eps = smoothing, on = 1 - eps, off = v > 1 ? eps / (v - 1) : 0;
        var smoothed = new Tensor<T>(new[] { n, v });
        for (int i = 0; i < n; i++) { for (int k = 0; k < v; k++) smoothed[i, k] = NumOps.FromDouble(off); smoothed[i, targets[i]] = NumOps.FromDouble(on); }
        double entropy = (on > 0 ? on * Math.Log(on) : 0) + (off > 0 ? (v - 1) * off * Math.Log(off) : 0);
        var crossEntropy = engine.ReduceSum(engine.TensorMultiply(logProbabilities, smoothed), new[] { 0, 1 }, keepDims: true);
        return engine.TensorMultiplyScalar(engine.TensorAddScalar(engine.TensorNegate(crossEntropy), NumOps.FromDouble(n * entropy)),
            NumOps.FromDouble(1.0 / normalizer));
    }
}
