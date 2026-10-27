using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>The settings of one masked-generative decode (MaskGIT's iterative parallel decoding, §3.1).</summary>
internal sealed record MaskGctDecoding
{
    /// <summary>Decoding steps.</summary>
    public int Steps { get; init; } = 50;

    /// <summary>The sampling temperature at the first step; it anneals linearly with t to 0.</summary>
    public double Temperature { get; init; } = 0.9;

    /// <summary>The top-k filter's kept fraction: k = ⌈(1 − threshold) · vocabulary⌉.</summary>
    public double FilterThreshold { get; init; } = 0.98;

    /// <summary>Classifier-free guidance scale (0: none).</summary>
    public double Guidance { get; init; } = 2.5;

    /// <summary>Guidance rescaling (Lin et al.): the guided output is renormalized to the conditional output's standard
    /// deviation and mixed in at this weight.</summary>
    public double GuidanceRescale { get; init; } = 0.75;
}

/// <summary>Shared pieces of MaskGCT's T2S and S2A models (Amphion <c>maskgct_t2s.py</c>, <c>maskgct_s2a.py</c>).</summary>
internal static class MaskGctSampling
{
    /// <summary>The mask schedule γ(t) = sin(πt/2).</summary>
    public static double MaskProbability(double t) => Math.Sin(t * Math.PI / 2);

    /// <summary>
    /// One step's guided embeddings (Appendix C): <c>g = g_c + w (g_c − g_u)</c>, then <c>g</c> rescaled to the standard
    /// deviation of <c>g_c</c> (unbiased, over every value) and mixed: <c>r · rescaled + (1 − r) · g</c>.
    /// </summary>
    public static Tensor<T> Guide<T>(IEngine engine, Tensor<T> conditional, Tensor<T> unconditional, double scale, double rescale)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double std = Std(conditional);
        var guided = engine.TensorAdd(conditional, engine.TensorMultiplyScalar(engine.TensorSubtract(conditional, unconditional), ops.FromDouble(scale)));
        var rescaled = engine.TensorMultiplyScalar(guided, ops.FromDouble(std / Std(guided)));
        return engine.TensorAdd(engine.TensorMultiplyScalar(rescaled, ops.FromDouble(rescale)), engine.TensorMultiplyScalar(guided, ops.FromDouble(1 - rescale)));
    }

    private static double Std<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var v = x.ToArray().Select(a => ops.ToDouble(a)).ToArray();
        double mean = v.Average();
        return Math.Sqrt(v.Sum(a => (a - mean) * (a - mean)) / (v.Length - 1));
    }

    /// <summary>The Gumbel noise of uniform draws: <c>−log(−log(u + 1e-10) + 1e-10)</c>.</summary>
    public static double Gumbel(double u) => -Math.Log(-Math.Log(u + 1e-10) + 1e-10);

    /// <summary>
    /// MaskGIT's iterative decoding as the release writes it. Starting fully masked, each step scores the sequence
    /// (<paramref name="logitsOf"/> given the current tokens and mask), keeps the top-k logits, samples every position
    /// (Gumbel-max at the annealed temperature; argmax at the last step, or temperature 0.2 when there is only one step),
    /// keeps the samples at masked positions, and re-masks the ⌊γ(t_next) · N⌋ masked positions of lowest confidence —
    /// confidence being the sample's probability plus Gumbel noise scaled by the annealed choice temperature.
    /// <paramref name="uniforms"/> supplies the uniform draws, a row-major block per call.
    /// </summary>
    public static int[] Decode(int length, int vocabulary, MaskGctDecoding d, Func<int[], bool[], double, double[,]> logitsOf, Func<int, double[]> uniforms)
    {
        var seq = new int[length];
        var mask = Enumerable.Repeat(true, length).ToArray();
        int k = (int)Math.Ceiling((1 - d.FilterThreshold) * vocabulary);
        double h = 1.0 / d.Steps;
        var times = Enumerable.Range(0, d.Steps).Select(i => 1.0 - i * h).Append(0.0).ToArray();
        for (int step = 0; step < d.Steps; step++)
        {
            double t = times[step];
            var logits = logitsOf(seq, mask, t);
            double temperature = d.Temperature * t, choice = t;
            // top_k: keep the k largest per position (others −∞).
            var filtered = new double[length, vocabulary];
            for (int i = 0; i < length; i++)
            {
                var row = Enumerable.Range(0, vocabulary).Select(j => logits[i, j]).ToArray();
                var sorted = (double[])row.Clone();
                Array.Sort(sorted);
                double kth = sorted[vocabulary - k];
                for (int j = 0; j < vocabulary; j++) filtered[i, j] = row[j] >= kth ? row[j] : double.NegativeInfinity;
            }
            var sampled = new int[length];
            bool last = step == d.Steps - 1;
            if (last && d.Steps > 1)
            {
                for (int i = 0; i < length; i++) sampled[i] = ArgMax(filtered, i, vocabulary, null, 1);
            }
            else
            {
                double temp = Math.Max(last ? 0.2 : temperature, 1e-3);
                var u = uniforms(length * vocabulary);
                for (int i = 0; i < length; i++) sampled[i] = ArgMax(filtered, i, vocabulary, u, temp);
            }
            var scores = new double[length];
            for (int i = 0; i < length; i++)
            {
                if (mask[i]) seq[i] = sampled[i];
                double max = double.NegativeInfinity, sum = 0;
                for (int j = 0; j < vocabulary; j++) max = Math.Max(max, filtered[i, j]);
                for (int j = 0; j < vocabulary; j++) sum += double.IsNegativeInfinity(filtered[i, j]) ? 0 : Math.Exp(filtered[i, j] - max);
                scores[i] = Math.Exp(filtered[i, sampled[i]] - max) / sum;
            }
            var noise = uniforms(length);
            for (int i = 0; i < length; i++) scores[i] = 1 - (choice * Gumbel(noise[i]) + scores[i]);
            int next = (int)(MaskProbability(times[step + 1]) * length);
            if (next == 0) break;
            var order = Enumerable.Range(0, length).Where(i => mask[i]).OrderByDescending(i => scores[i]).ThenBy(i => i).Take(next).ToHashSet();
            for (int i = 0; i < length; i++)
            {
                mask[i] = order.Contains(i);
                if (mask[i]) seq[i] = 0;
            }
        }
        return seq;
    }

    // argmax_j (logits[i, j] / temperature + gumbel(u[i, j])), or of the logits alone without draws.
    private static int ArgMax(double[,] logits, int i, int vocabulary, double[]? u, double temperature)
    {
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int j = 0; j < vocabulary; j++)
        {
            double v = u is null ? logits[i, j] : logits[i, j] / temperature + Gumbel(u[i * vocabulary + j]);
            if (v > bestValue)
            {
                bestValue = v;
                best = j;
            }
        }
        return best;
    }

    internal static Tensor<T> Ids<T>(IReadOnlyList<int> ids)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(new[] { ids.Count }, new Vector<T>(ids.Select(i => ops.FromDouble(i)).ToArray()));
    }

    internal static double[,] Matrix<T>(Tensor<T> x)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var m = new double[x.Shape[0], x.Shape[1]];
        for (int i = 0; i < x.Shape[0]; i++)
            for (int j = 0; j < x.Shape[1]; j++) m[i, j] = ops.ToDouble(x[i, j]);
        return m;
    }
}

/// <summary>
/// MaskGCT's text-to-semantic model (§3.2.2; Amphion <c>MaskGCT_T2S</c>): the phones (embedded, through the prefix MLP),
/// the prompt's semantic tokens and the target's masked semantic tokens in one bidirectional sequence, the step t
/// conditioning every norm, and a head over the 8,192 semantic codes.
/// </summary>
internal sealed class MaskGctTextToSemantic<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MaskGctTextToSemantic(IEngine engine, List<LayerBase<T>> owner, MaskGctTransformerConfiguration backbone, int semanticCodes, int phones, Random initialization)
    {
        _engine = engine;
        SemanticCodes = semanticCodes;
        Phones = phones;
        int h = backbone.Hidden;
        TiedEmbeddingLayer<T> Table(int rows, int? zero = null)
        {
            var table = new TiedEmbeddingLayer<T>(rows, h);
            table.Reinitialize(() => 0.02 * Normal(initialization), zero);
            owner.Add(table);
            return table;
        }
        Mask = Table(1);
        Semantic = Table(semanticCodes);
        Phone = Table(phones, phones - 1);
        Head = new DenseLayer<T>(semanticCodes, (IActivationFunction<T>)new IdentityActivation<T>());
        owner.Add(Head);
        TorchParameters.Linear(engine, Head, h, semanticCodes, TorchParameters.Gaussian<T>(initialization, h * semanticCodes, 0.02).Select(v => NumOps.ToDouble(v)).ToArray(), null);
        Backbone = new MaskGctTransformer<T>(engine, owner, backbone with { PrefixCondition = true }, initialization);
    }

    internal static double Normal(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());

    public int SemanticCodes { get; }
    public int Phones { get; }
    public TiedEmbeddingLayer<T> Mask { get; }
    public TiedEmbeddingLayer<T> Semantic { get; }
    public TiedEmbeddingLayer<T> Phone { get; }
    public DenseLayer<T> Head { get; }
    public MaskGctTransformer<T> Backbone { get; }

    private Tensor<T> MaskRow(int rows) => _engine.TensorTile(Mask.Forward(MaskGctSampling.Ids<T>(new[] { 0 })), new[] { rows, 1 });

    // mask · mask_emb + (1 − mask) · cond_emb(tokens), over [rows, hidden].
    private Tensor<T> Masked(IReadOnlyList<int> tokens, IReadOnlyList<bool> masked)
    {
        int n = tokens.Count, h = Mask.Table.Shape[1];
        var m = new Tensor<T>(new[] { n, h });
        for (int i = 0; i < n; i++)
            if (masked[i]) for (int j = 0; j < h; j++) m[i, j] = NumOps.One;
        var keep = _engine.TensorAddScalar(_engine.TensorNegate(m), NumOps.One);
        return _engine.TensorAdd(_engine.TensorMultiply(m, MaskRow(n)), _engine.TensorMultiply(keep, Semantic.Forward(MaskGctSampling.Ids<T>(tokens))));
    }

    /// <summary>The logits <c>[target, codes]</c> of a masked target after a prompt (<c>loss_t</c>'s forward with the prompt
    /// in front, or a decoding step).</summary>
    public Tensor<T> Logits(IReadOnlyList<int> phones, IReadOnlyList<int> prompt, IReadOnlyList<int> target, IReadOnlyList<bool> masked, double t)
    {
        var x = prompt.Count == 0 ? Masked(target, masked)
            : _engine.TensorConcatenate(new[] { Semantic.Forward(MaskGctSampling.Ids<T>(prompt)), Masked(target, masked) }, 0);
        var phoneEmbedding = phones.Count == 0 ? null : Phone.Forward(MaskGctSampling.Ids<T>(phones));
        var hidden = Backbone.Forward(x, t, phoneEmbedding);
        if (prompt.Count > 0) hidden = _engine.TensorSlice(hidden, new[] { prompt.Count, 0 }, new[] { target.Count, hidden.Shape[1] });
        return Head.Forward(hidden);
    }

    /// <summary>The release's <c>reverse_diffusion</c>: decodes <paramref name="length"/> semantic tokens after the
    /// prompt, the unconditional pass for guidance seeing neither the phones nor the prompt.</summary>
    public int[] Generate(IReadOnlyList<int> phones, IReadOnlyList<int> prompt, int length, MaskGctDecoding decoding, Func<int, double[]> uniforms)
    {
        return MaskGctSampling.Decode(length, SemanticCodes, decoding, (seq, mask, t) =>
        {
            var target = Masked(seq, mask);
            var x = prompt.Count == 0 ? target : _engine.TensorConcatenate(new[] { Semantic.Forward(MaskGctSampling.Ids<T>(prompt)), target }, 0);
            var phoneEmbedding = Phone.Forward(MaskGctSampling.Ids<T>(phones));
            var embeds = Backbone.Forward(x, t, phoneEmbedding);
            embeds = _engine.TensorSlice(embeds, new[] { prompt.Count, 0 }, new[] { length, embeds.Shape[1] });
            if (decoding.Guidance > 0)
            {
                var unconditional = Backbone.Forward(target, t, null);
                embeds = MaskGctSampling.Guide(_engine, embeds, unconditional, decoding.Guidance, decoding.GuidanceRescale);
            }
            return MaskGctSampling.Matrix(Head.Forward(embeds));
        }, uniforms);
    }

    /// <summary>Loads Amphion's <c>MaskGCT_T2S</c> state dict.</summary>
    public void Load(Func<string, int[], double[]?> read)
    {
        int h = Mask.Table.Shape[1];
        double[] Need(string name, int[] shape) => read(name, shape) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        Mask.LoadTable(Need("mask_emb.weight", new[] { 1, h }));
        Semantic.LoadTable(Need("cond_emb.weight", new[] { SemanticCodes, h }));
        Phone.LoadTable(Need("phone_emb.weight", new[] { Phones, h }));
        TorchParameters.Linear(_engine, Head, h, SemanticCodes, Need("to_logit.weight", new[] { SemanticCodes, h }), Need("to_logit.bias", new[] { SemanticCodes }));
        Backbone.LoadTorchWeights(read, "diff_estimator.");
    }
}

/// <summary>
/// MaskGCT's semantic-to-acoustic model (§3.2.3; Amphion <c>MaskGCT_S2A</c>, after SoundStorm): every frame's semantic
/// token embedding plus the predicted layer's embedding conditions the sequence; the input sums the acoustic tokens of
/// the layers below the predicted one, the predicted layer (masked where it is still unknown) and mask embeddings for
/// the layers above; a head per acoustic layer. The release ships two: one predicting only the first layer, and one with
/// all twelve that predicts layers 2–12 given the first.
/// </summary>
internal sealed class MaskGctSemanticToAcoustic<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MaskGctSemanticToAcoustic(IEngine engine, List<LayerBase<T>> owner, MaskGctTransformerConfiguration backbone, int quantizers, int codebookSize,
        int semanticCodes, bool predictFirstLayer, Random initialization)
    {
        _engine = engine;
        Quantizers = quantizers;
        CodebookSize = codebookSize;
        PredictFirstLayer = predictFirstLayer;
        int h = backbone.Hidden;
        TiedEmbeddingLayer<T> Table(int rows)
        {
            var table = new TiedEmbeddingLayer<T>(rows, h);
            table.Reinitialize(() => 0.02 * MaskGctTextToSemantic<T>.Normal(initialization));
            owner.Add(table);
            return table;
        }
        LayerEmbedding = Table(quantizers);
        Mask = Table(1);
        Tokens = Enumerable.Range(0, quantizers).Select(_ => Table(codebookSize)).ToArray();
        Heads = Enumerable.Range(0, quantizers).Select(_ =>
        {
            var head = new DenseLayer<T>(codebookSize, (IActivationFunction<T>)new IdentityActivation<T>());
            owner.Add(head);
            TorchParameters.Linear(engine, head, h, codebookSize, TorchParameters.Gaussian<T>(initialization, h * codebookSize, 0.02).Select(v => NumOps.ToDouble(v)).ToArray(), null);
            return head;
        }).ToArray();
        Semantic = Table(semanticCodes);
        Backbone = new MaskGctTransformer<T>(engine, owner, backbone with { PrefixCondition = false }, initialization);
    }

    public int Quantizers { get; }
    public int CodebookSize { get; }
    public bool PredictFirstLayer { get; }
    public TiedEmbeddingLayer<T> LayerEmbedding { get; }
    public TiedEmbeddingLayer<T> Mask { get; }
    public TiedEmbeddingLayer<T>[] Tokens { get; }
    public DenseLayer<T>[] Heads { get; }
    public TiedEmbeddingLayer<T> Semantic { get; }
    public MaskGctTransformer<T> Backbone { get; }

    private Tensor<T> MaskRows(int rows) => _engine.TensorTile(Mask.Forward(MaskGctSampling.Ids<T>(new[] { 0 })), new[] { rows, 1 });

    private Tensor<T> Column(int layer, int[,] codes, int start, int count)
    {
        var ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = codes[start + i, layer];
        return Tokens[layer].Forward(MaskGctSampling.Ids<T>(ids));
    }

    /// <summary>The per-frame condition <c>[frames, hidden]</c>: <c>cond_emb(semantic) + layer_emb(layer)</c>.</summary>
    public Tensor<T> Condition(IReadOnlyList<int> semantic, int layer) =>
        _engine.TensorAdd(Semantic.Forward(MaskGctSampling.Ids<T>(semantic)),
            _engine.TensorTile(LayerEmbedding.Forward(MaskGctSampling.Ids<T>(new[] { layer })), new[] { semantic.Count, 1 }));

    /// <summary>The training forward (<c>forward_diffusion</c> and <c>loss_t</c>) for a chosen layer, prompt length and
    /// mask: the logits <c>[frames, codes]</c> of layer <paramref name="layer"/>.</summary>
    public Tensor<T> TrainingLogits(IReadOnlyList<int> semantic, int[,] codes, int layer, int promptLength, IReadOnlyList<bool> masked, double t)
    {
        int n = codes.GetLength(0), h = Mask.Table.Shape[1];
        Tensor<T> x = new(new[] { n, h });
        var maskRows = MaskRows(n);
        var prompt = new Tensor<T>(new[] { n, h });
        for (int i = 0; i < Math.Min(promptLength, n); i++)
            for (int j = 0; j < h; j++) prompt[i, j] = NumOps.One;
        var notPrompt = _engine.TensorAddScalar(_engine.TensorNegate(prompt), NumOps.One);
        for (int q = 0; q < Quantizers; q++)
        {
            var e = Column(q, codes, 0, n);
            if (q < layer) x = _engine.TensorAdd(x, e);
            else if (q == layer)
            {
                var m = new Tensor<T>(new[] { n, h });
                for (int i = 0; i < n; i++)
                    if (masked[i]) for (int j = 0; j < h; j++) m[i, j] = NumOps.One;
                var keep = _engine.TensorAddScalar(_engine.TensorNegate(m), NumOps.One);
                x = _engine.TensorAdd(x, _engine.TensorAdd(_engine.TensorMultiply(m, maskRows), _engine.TensorMultiply(keep, e)));
            }
            else
            {
                x = _engine.TensorAdd(x, _engine.TensorAdd(_engine.TensorMultiply(e, prompt), _engine.TensorMultiply(maskRows, notPrompt)));
            }
        }
        return Heads[layer].Forward(Backbone.Forward(x, t, Condition(semantic, layer)));
    }

    /// <summary>
    /// The release's <c>reverse_diffusion</c>: given the semantic tokens of the prompt and the target, the prompt's
    /// acoustic codes <c>[prompt, quantizers ≥ this model's]</c> and optionally the target's first <c>known</c> layers, decodes
    /// the target's remaining layers coarse to fine with <paramref name="steps"/>[layer] steps each. Returns
    /// <c>[target, layers]</c>.
    /// </summary>
    public int[,] Generate(IReadOnlyList<int> semantic, int[,] prompt, int[,]? known, int[] steps, MaskGctDecoding decoding, Func<int, double[]> uniforms)
    {
        int lp = prompt.GetLength(0), n = semantic.Count - lp, h = Mask.Table.Shape[1], maxLayer = Quantizers;
        if (n <= 0) throw new ArgumentException("The semantic tokens must cover the prompt and a target.", nameof(semantic));
        if (steps.Length != Quantizers) throw new ArgumentException($"Give a step count per layer ({Quantizers}).", nameof(steps));
        var result = new int[n, maxLayer];
        var cum = new Tensor<T>(new[] { n, h });
        int knownLayers = known?.GetLength(1) ?? 0;
        for (int q = 0; q < knownLayers; q++)
        {
            for (int i = 0; i < n; i++) result[i, q] = known![i, q];
            cum = _engine.TensorAdd(cum, Column(q, result, 0, n));
        }
        Tensor<T> promptSum = new(new[] { lp, h });
        for (int q = 0; q < Quantizers; q++) promptSum = _engine.TensorAdd(promptSum, Column(q, prompt, 0, lp));
        for (int layer = knownLayers; layer < maxLayer; layer++)
        {
            var condition = Condition(semantic, layer);
            var targetCondition = _engine.TensorSlice(condition, new[] { lp, 0 }, new[] { n, h });
            var maskRows = MaskRows(n);
            int current = layer;
            var cumulative = cum;
            var seq = MaskGctSampling.Decode(n, CodebookSize, decoding with { Steps = steps[layer] }, (tokens, mask, t) =>
            {
                var m = new Tensor<T>(new[] { n, h });
                for (int i = 0; i < n; i++)
                    if (mask[i]) for (int j = 0; j < h; j++) m[i, j] = NumOps.One;
                var keep = _engine.TensorAddScalar(_engine.TensorNegate(m), NumOps.One);
                var cur = _engine.TensorAdd(cumulative, _engine.TensorAdd(_engine.TensorMultiply(m, maskRows), _engine.TensorMultiply(keep, Tokens[current].Forward(MaskGctSampling.Ids<T>(tokens)))));
                cur = _engine.TensorAdd(cur, _engine.TensorMultiplyScalar(maskRows, NumOps.FromDouble(maxLayer - 1 - current)));
                var embeds = Backbone.Forward(lp == 0 ? cur : _engine.TensorConcatenate(new[] { promptSum, cur }, 0), t, condition);
                embeds = _engine.TensorSlice(embeds, new[] { lp, 0 }, new[] { n, h });
                if (decoding.Guidance > 0)
                {
                    var unconditional = Backbone.Forward(cur, t, targetCondition);
                    embeds = MaskGctSampling.Guide(_engine, embeds, unconditional, decoding.Guidance, decoding.GuidanceRescale);
                }
                return MaskGctSampling.Matrix(Heads[current].Forward(embeds));
            }, uniforms);
            for (int i = 0; i < n; i++) result[i, layer] = seq[i];
            cum = _engine.TensorAdd(cum, Tokens[layer].Forward(MaskGctSampling.Ids<T>(seq)));
        }
        return result;
    }

    /// <summary>Loads Amphion's <c>MaskGCT_S2A</c> state dict.</summary>
    public void Load(Func<string, int[], double[]?> read)
    {
        int h = Mask.Table.Shape[1];
        double[] Need(string name, int[] shape) => read(name, shape) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        LayerEmbedding.LoadTable(Need("layer_emb.weight", new[] { Quantizers, h }));
        Mask.LoadTable(Need("mask_emb.weight", new[] { 1, h }));
        for (int q = 0; q < Quantizers; q++)
        {
            Tokens[q].LoadTable(Need($"token_emb.{q}.weight", new[] { CodebookSize, h }));
            TorchParameters.Linear(_engine, Heads[q], h, CodebookSize, Need($"to_logits.{q}.weight", new[] { CodebookSize, h }), Need($"to_logits.{q}.bias", new[] { CodebookSize }));
        }
        Semantic.LoadTable(Need("cond_emb.weight", new[] { Semantic.Table.Shape[0], h }));
        Backbone.LoadTorchWeights(read, "diff_estimator.");
    }
}
