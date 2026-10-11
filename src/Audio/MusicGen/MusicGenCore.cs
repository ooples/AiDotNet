using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.Audio.MusicGen;

/// <summary>The sizes of MusicGen's language model.</summary>
/// <remarks>With <c>CrossAttention</c> the conditioning is attended to by every layer (text-to-music); without it the
/// conditioning is a prefix of the sequence (the melody model, §2.4: "we instead provide the conditioning tensor C as a
/// prefix to the transformer input"), and the layers have no cross-attention.</remarks>
internal sealed record MusicGenConfiguration(int Codebooks, int CodebookSize, int ModelDim, int Heads, int Layers,
    int FeedForwardDim, double Dropout, int[] Delays, bool CrossAttention = true);

/// <summary>
/// The "delay" codebook interleaving pattern (Copet et al. 2023, §2.2, Eq. 6; audiocraft
/// <c>DelayedPatternProvider</c> with an empty first step): step 0 holds no codes, and step s ≥ 1 holds code (t, k) with
/// t = s − 1 − d_k, so codebook k lags the first by d_k steps.
/// </summary>
internal static class MusicGenDelayPattern
{
    /// <summary>The model input <c>[K, T + 1]</c> for codes <c>[K, T]</c>: each step's present codes and the special
    /// token elsewhere (the pattern's valid steps, as training uses them).</summary>
    public static int[,] TrainingSequence(int[,] codes, int[] delays, int special)
    {
        int k = codes.GetLength(0), t = codes.GetLength(1);
        var sequence = new int[k, t + 1];
        for (int q = 0; q < k; q++)
            for (int s = 0; s <= t; s++)
            {
                int time = s - 1 - delays[q];
                sequence[q, s] = time >= 0 && time < t ? codes[q, time] : special;
            }
        return sequence;
    }

    /// <summary>Whether target (k, t) is predicted in a training sequence of T + 1 steps: output step s predicts the
    /// codes of step s + 1, and only steps 1..T of the pattern are fully defined (audiocraft <c>revert_pattern_logits</c>
    /// on the valid layout), so (k, t) counts when t + d_k &lt; T.</summary>
    public static bool Predicted(int t, int q, int timesteps, int[] delays) => t + delays[q] < timesteps;
}

/// <summary>
/// MusicGen's autoregressive language model (Copet et al. 2023, §2.4; audiocraft <c>LMModel</c> with
/// <c>StreamingTransformer</c>): per-codebook embeddings (N + 1 entries, the last the special token) summed with a
/// sinusoidal step embedding; L pre-norm layers of causal self-attention, cross-attention to the conditioning, and a
/// D → 4D → D ReLU feed-forward block, each residual; a final LayerNorm; and one linear head per codebook. Linear
/// layers carry no biases (the reference's MusicGen configuration).
/// </summary>
internal sealed class MusicGenCore<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public MusicGenCore(IEngine engine, List<LayerBase<T>> layers, MusicGenConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Delays.Length != c.Codebooks)
            throw new ArgumentException($"The delay pattern has {c.Delays.Length} delays for {c.Codebooks} codebooks.");
        if (c.ModelDim % c.Heads != 0 || c.ModelDim % 2 != 0)
            throw new ArgumentException($"The model width {c.ModelDim} must be even and divisible by the {c.Heads} heads.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        for (int q = 0; q < c.Codebooks; q++) Embeddings.Add(Own(new TiedEmbeddingLayer<T>(c.CodebookSize + 1, c.ModelDim)));
        for (int i = 0; i < c.Layers; i++)
        {
            var norm1 = Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5));
            var selfIn = Own(new BiasFreeLinearLayer<T>(c.ModelDim, 3 * c.ModelDim));
            var selfOut = Own(new BiasFreeLinearLayer<T>(c.ModelDim, c.ModelDim));
            var normCross = c.CrossAttention ? Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5)) : null;
            var crossIn = c.CrossAttention ? Own(new BiasFreeLinearLayer<T>(c.ModelDim, 3 * c.ModelDim)) : null;
            var crossOut = c.CrossAttention ? Own(new BiasFreeLinearLayer<T>(c.ModelDim, c.ModelDim)) : null;
            Blocks.Add(new MusicGenBlock(norm1, selfIn, selfOut, normCross, crossIn, crossOut,
                Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5)), Own(new BiasFreeLinearLayer<T>(c.ModelDim, c.FeedForwardDim)),
                Own(new BiasFreeLinearLayer<T>(c.FeedForwardDim, c.ModelDim))));
        }
        OutNorm = Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5));
        for (int q = 0; q < c.Codebooks; q++) Heads.Add(Own(new BiasFreeLinearLayer<T>(c.ModelDim, c.CodebookSize)));
    }

    /// <summary>One transformer layer's parameters.</summary>
    internal sealed record MusicGenBlock(
        LayerNormalizationLayer<T> Norm1, BiasFreeLinearLayer<T> SelfIn, BiasFreeLinearLayer<T> SelfOut,
        LayerNormalizationLayer<T>? NormCross, BiasFreeLinearLayer<T>? CrossIn, BiasFreeLinearLayer<T>? CrossOut,
        LayerNormalizationLayer<T> Norm2, BiasFreeLinearLayer<T> Linear1, BiasFreeLinearLayer<T> Linear2);

    public MusicGenConfiguration Configuration { get; }
    public List<TiedEmbeddingLayer<T>> Embeddings { get; } = new();
    public List<MusicGenBlock> Blocks { get; } = new();
    public LayerNormalizationLayer<T> OutNorm { get; }
    public List<BiasFreeLinearLayer<T>> Heads { get; } = new();

    /// <summary>The special (absent-code) token id, N.</summary>
    public int Special => Configuration.CodebookSize;

    // ------------------------------------------------------------------ building blocks

    /// <summary>The step embedding <c>[L, D]</c> of positions offset..offset + L − 1 (audiocraft
    /// <c>create_sin_embedding</c>: phase p / 10000^(i / (D/2 − 1)), cosines then sines).</summary>
    private Tensor<T> Positions(int offset, int length)
    {
        int d = Configuration.ModelDim, half = d / 2;
        var table = new Tensor<T>(new[] { length, d });
        for (int p = 0; p < length; p++)
            for (int i = 0; i < half; i++)
            {
                double phase = (offset + p) / Math.Pow(10000.0, (double)i / (half - 1));
                table[p, i] = NumOps.FromDouble(Math.Cos(phase));
                table[p, half + i] = NumOps.FromDouble(Math.Sin(phase));
            }
        return table;
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.Dropout;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    // [L, D] → [H, L, D/H]
    private Tensor<T> SplitHeads(Tensor<T> x)
    {
        int length = x.Shape[0], heads = Configuration.Heads, headDim = Configuration.ModelDim / heads;
        return _engine.TensorPermute(_engine.Reshape(x, new[] { length, heads, headDim }), new[] { 1, 0, 2 });
    }

    // Scaled dot-product attention of queries [Lq, D] over keys and values [Lk, D], optionally causal with the query
    // block starting at key position `queryOffset`; the result is [Lq, D].
    private Tensor<T> Attend(Tensor<T> queries, Tensor<T> keys, Tensor<T> values, bool causal, int queryOffset, bool training, Random random)
    {
        int lq = queries.Shape[0], lk = keys.Shape[0], d = Configuration.ModelDim, heads = Configuration.Heads;
        var q = _engine.TensorMultiplyScalar(SplitHeads(queries), NumOps.FromDouble(1.0 / Math.Sqrt(d / heads)));
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(SplitHeads(keys), new[] { 0, 2, 1 }));       // [H, Lq, Lk]
        if (causal && lk > 1)
        {
            var mask = new Tensor<T>(new[] { 1, lq, lk });
            var blocked = NumOps.FromDouble(-1e9);
            bool any = false;
            for (int i = 0; i < lq; i++)
                for (int j = queryOffset + i + 1; j < lk; j++)
                {
                    mask[0, i, j] = blocked;
                    any = true;
                }
            if (any) scores = _engine.TensorAdd(scores, _engine.TensorTile(mask, new[] { heads, 1, 1 }));
        }
        var weights = Drop(_engine.Softmax(scores, axis: 2), training, random);
        var context = _engine.BatchMatMul(weights, SplitHeads(values));                                         // [H, Lq, D/H]
        return _engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { lq, d });
    }

    private Tensor<T> Columns(Tensor<T> x, int part)
    {
        int d = Configuration.ModelDim;
        return _engine.TensorSlice(x, new[] { 0, part * d }, new[] { x.Shape[0], d });
    }

    // One layer over the new positions x [L, D] (at sequence offset `offset`), with the self-attention keys and values
    // of earlier positions in `cache` (appended to), and cross-attention to `memory` [Lc, D].
    private Tensor<T> Block(MusicGenBlock b, Tensor<T> x, Tensor<T> memory, int offset, List<(Tensor<T> K, Tensor<T> V)>? cache,
        int index, bool training, Random random)
    {
        var projected = b.SelfIn.Forward(b.Norm1.Forward(x));                                                 // [L, 3D]
        var keys = Columns(projected, 1);
        var values = Columns(projected, 2);
        if (cache is not null)
        {
            if (cache.Count > index)
            {
                keys = _engine.TensorConcatenate(new[] { cache[index].K, keys }, 0);
                values = _engine.TensorConcatenate(new[] { cache[index].V, values }, 0);
                cache[index] = (keys, values);
            }
            else cache.Add((keys, values));
        }
        var attended = Attend(Columns(projected, 0), keys, values, causal: true, queryOffset: offset, training, random);
        x = _engine.TensorAdd(x, Drop(b.SelfOut.Forward(attended), training, random));
        if (b.NormCross is not null && b.CrossIn is not null && b.CrossOut is not null)
        {
            var crossQueries = Columns(b.CrossIn.Forward(b.NormCross.Forward(x)), 0);
            var crossMemory = b.CrossIn.Forward(memory);
            var crossed = Attend(crossQueries, Columns(crossMemory, 1), Columns(crossMemory, 2), causal: false, 0, training, random);
            x = _engine.TensorAdd(x, Drop(b.CrossOut.Forward(crossed), training, random));
        }
        var hidden = Drop(_engine.ReLU(b.Linear1.Forward(b.Norm2.Forward(x))), training, random);
        return _engine.TensorAdd(x, Drop(b.Linear2.Forward(hidden), training, random));
    }

    private Tensor<T> Embed(int[,] sequence, int from, int length)
    {
        Tensor<T>? sum = null;
        for (int q = 0; q < Configuration.Codebooks; q++)
        {
            var ids = new Tensor<T>(new[] { length });
            for (int s = 0; s < length; s++) ids[s] = NumOps.FromDouble(sequence[q, from + s]);
            var embedded = Embeddings[q].Forward(ids);                                                         // [L, D]
            sum = sum is null ? embedded : _engine.TensorAdd(sum, embedded);
        }
        return sum ?? throw new InvalidOperationException("MusicGen has no codebooks.");
    }

    // The logits [K, L, N] of the new positions `sequence[:, from .. from + length)`. In prefix mode the conditioning
    // `memory` [P, D] precedes step 0, so the steps sit at positions P + s and the first call carries the prefix.
    private Tensor<T> Run(int[,] sequence, int from, int length, Tensor<T> memory, List<(Tensor<T>, Tensor<T>)>? cache, bool training, Random random)
    {
        int prefix = Configuration.CrossAttention ? 0 : memory.Shape[0];
        var x = Embed(sequence, from, length);
        int position = prefix + from;
        if (prefix > 0 && from == 0)
        {
            x = _engine.TensorConcatenate(new[] { memory, x }, 0);
            position = 0;
        }
        x = _engine.TensorAdd(x, Positions(position, x.Shape[0]));
        for (int i = 0; i < Blocks.Count; i++) x = Block(Blocks[i], x, memory, position, cache, i, training, random);
        if (x.Shape[0] > length)
            x = _engine.TensorSlice(x, new[] { x.Shape[0] - length, 0 }, new[] { length, Configuration.ModelDim });
        var output = OutNorm.Forward(x);
        var perCodebook = new Tensor<T>[Configuration.Codebooks];
        for (int q = 0; q < perCodebook.Length; q++)
            perCodebook[q] = _engine.Reshape(Heads[q].Forward(output), new[] { 1, length, Configuration.CodebookSize });
        return perCodebook.Length == 1 ? perCodebook[0] : _engine.TensorConcatenate(perCodebook, 0);
    }

    // ------------------------------------------------------------------ model operations

    /// <summary>The logits <c>[K, S, N]</c> of a pattern sequence <c>[K, S]</c> given the conditioning <c>[Lc, D]</c>.</summary>
    public Tensor<T> Logits(int[,] sequence, Tensor<T> memory, bool training, Random random)
        => Run(sequence, 0, sequence.GetLength(1), memory, null, training, random);

    /// <summary>
    /// The training objective (audiocraft <c>MusicGenSolver._compute_cross_entropy</c>): the cross-entropy of each
    /// codebook's codes over the steps the pattern predicts them at, averaged over its valid steps, then over the
    /// codebooks.
    /// </summary>
    public Tensor<T> Loss(int[,] codes, Tensor<T> memory, bool training, Random random)
    {
        var c = Configuration;
        int k = codes.GetLength(0), t = codes.GetLength(1);
        var logits = Logits(MusicGenDelayPattern.TrainingSequence(codes, c.Delays, Special), memory, training, random);   // [K, T+1, N]
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: 2);
        var weights = new Tensor<T>(logits._shape);
        for (int q = 0; q < k; q++)
        {
            int valid = 0;
            for (int time = 0; time < t; time++) if (MusicGenDelayPattern.Predicted(time, q, t, c.Delays)) valid++;
            if (valid == 0) throw new ArgumentException($"{t} frames leave no target for codebook {q} after its {c.Delays[q]}-step delay.");
            var weight = NumOps.FromDouble(-1.0 / (valid * k));
            for (int time = 0; time < t; time++)
                if (MusicGenDelayPattern.Predicted(time, q, t, c.Delays)) weights[q, time + c.Delays[q], codes[q, time]] = weight;
        }
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, weights), new[] { 0, 1, 2 }, keepDims: false);
    }

    /// <summary>
    /// Generates <paramref name="frames"/> frames of codes <c>[K, frames]</c> (audiocraft <c>LMModel.generate</c>): the
    /// pattern is filled one step at a time from cached keys and values; each step's logits are
    /// <c>uncond + guidance · (cond − uncond)</c> when an unconditional memory is given, and <paramref name="choose"/>
    /// picks each codebook's code from its logits; codes the pattern has no slot for stay special.
    /// </summary>
    public int[,] Generate(Tensor<T> memory, Tensor<T>? unconditional, double guidance, int frames, Func<double[], int> choose,
        int[,]? prompt = null)
    {
        var c = Configuration;
        int k = c.Codebooks, maxDelay = c.Delays.Max(), steps = frames + maxDelay + 1;
        const int Unknown = -1;
        var sequence = new int[k, steps];
        for (int q = 0; q < k; q++)
            for (int s = 0; s < steps; s++)
            {
                int time = s - 1 - c.Delays[q];
                sequence[q, s] = time >= 0 && time < frames
                    ? (prompt is not null && time < prompt.GetLength(1) ? prompt[q, time] : Unknown)
                    : Special;
            }
        int promptFrames = prompt?.GetLength(1) ?? 0;
        // The first step that holds a code of the first timestep still to generate (pattern.get_first_step_with_timesteps).
        int start = promptFrames + 1;
        var condCache = new List<(Tensor<T>, Tensor<T>)>();
        var uncondCache = unconditional is null ? null : new List<(Tensor<T>, Tensor<T>)>();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);
        using var _ = new NoGradScope<T>();
        int previous = 0;
        for (int offset = start; offset < steps; offset++)
        {
            int length = offset - previous;
            var condLogits = Run(sequence, previous, length, memory, condCache, false, random);
            Tensor<T>? uncondLogits = unconditional is null || uncondCache is null
                ? null
                : Run(sequence, previous, length, unconditional, uncondCache, false, random);
            for (int q = 0; q < k; q++)
            {
                var logits = new double[c.CodebookSize];
                for (int v = 0; v < logits.Length; v++)
                {
                    double conditional = NumOps.ToDouble(condLogits[q, length - 1, v]);
                    logits[v] = uncondLogits is null
                        ? conditional
                        : NumOps.ToDouble(uncondLogits[q, length - 1, v]) + guidance * (conditional - NumOps.ToDouble(uncondLogits[q, length - 1, v]));
                }
                if (sequence[q, offset] == Unknown) sequence[q, offset] = choose(logits);
            }
            previous = offset;
        }
        var codes = new int[k, frames];
        for (int q = 0; q < k; q++)
            for (int t = 0; t < frames; t++) codes[q, t] = sequence[q, t + 1 + c.Delays[q]];
        return codes;
    }

    // ------------------------------------------------------------------ parameters

    /// <summary>The reference's initialization (audiocraft <c>LMModel._init_weights</c> with <c>weight_init: gaussian</c>,
    /// <c>depthwise_init: current</c>): truncated-normal (±3σ) embeddings and heads with σ = 1/√fan-in, and each layer's
    /// output and feed-forward projections with σ = 1/√(fan-in · 2 · depth); the attention input projections keep
    /// PyTorch's <c>nn.Linear</c> default, uniform in ±1/√fan-in, because the reference stores them outside a Linear
    /// module; LayerNorms at one and zero.</summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        double TruncatedNormal(double std)
        {
            while (true)
            {
                double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                double z = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
                if (Math.Abs(z) <= 3) return z * std;
            }
        }
        void Fill(BiasFreeLinearLayer<T> layer, int inputs, int outputs, Func<double> sample)
        {
            var values = new Vector<T>(inputs * outputs);
            for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble(sample());
            layer.SetParameters(values);
        }
        foreach (var embedding in Embeddings) embedding.Reinitialize(() => TruncatedNormal(1 / Math.Sqrt(c.ModelDim)));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            double depth = Math.Sqrt(2.0 * (i + 1));
            double uniform = 1 / Math.Sqrt(c.ModelDim);
            Fill(b.SelfIn, c.ModelDim, 3 * c.ModelDim, () => (2 * random.NextDouble() - 1) * uniform);
            if (b.CrossIn is not null) Fill(b.CrossIn, c.ModelDim, 3 * c.ModelDim, () => (2 * random.NextDouble() - 1) * uniform);
            Fill(b.SelfOut, c.ModelDim, c.ModelDim, () => TruncatedNormal(1 / Math.Sqrt(c.ModelDim) / depth));
            if (b.CrossOut is not null) Fill(b.CrossOut, c.ModelDim, c.ModelDim, () => TruncatedNormal(1 / Math.Sqrt(c.ModelDim) / depth));
            Fill(b.Linear1, c.ModelDim, c.FeedForwardDim, () => TruncatedNormal(1 / Math.Sqrt(c.ModelDim) / depth));
            Fill(b.Linear2, c.FeedForwardDim, c.ModelDim, () => TruncatedNormal(1 / Math.Sqrt(c.FeedForwardDim) / depth));
        }
        foreach (var head in Heads) Fill(head, c.ModelDim, c.CodebookSize, () => TruncatedNormal(1 / Math.Sqrt(c.ModelDim)));
    }

    /// <summary>Loads an audiocraft <c>LMModel</c> state dict (names as in <c>lm.state_dict()</c>).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        for (int q = 0; q < c.Codebooks; q++) Embeddings[q].LoadTable(read($"emb.{q}.weight", new[] { c.CodebookSize + 1, d }));
        for (int i = 0; i < Blocks.Count; i++)
        {
            var b = Blocks[i];
            string p = $"transformer.layers.{i}";
            TorchParameters.Linear(b.SelfIn, d, 3 * d, read($"{p}.self_attn.in_proj_weight", new[] { 3 * d, d }));
            TorchParameters.Linear(b.SelfOut, d, d, read($"{p}.self_attn.out_proj.weight", new[] { d, d }));
            if (b.CrossIn is not null && b.CrossOut is not null && b.NormCross is not null)
            {
                TorchParameters.Linear(b.CrossIn, d, 3 * d, read($"{p}.cross_attention.in_proj_weight", new[] { 3 * d, d }));
                TorchParameters.Linear(b.CrossOut, d, d, read($"{p}.cross_attention.out_proj.weight", new[] { d, d }));
                TorchParameters.LayerNorm(_engine, b.NormCross, read($"{p}.norm_cross.weight", new[] { d }), read($"{p}.norm_cross.bias", new[] { d }));
            }
            TorchParameters.Linear(b.Linear1, d, c.FeedForwardDim, read($"{p}.linear1.weight", new[] { c.FeedForwardDim, d }));
            TorchParameters.Linear(b.Linear2, c.FeedForwardDim, d, read($"{p}.linear2.weight", new[] { d, c.FeedForwardDim }));
            TorchParameters.LayerNorm(_engine, b.Norm1, read($"{p}.norm1.weight", new[] { d }), read($"{p}.norm1.bias", new[] { d }));
            TorchParameters.LayerNorm(_engine, b.Norm2, read($"{p}.norm2.weight", new[] { d }), read($"{p}.norm2.bias", new[] { d }));
        }
        TorchParameters.LayerNorm(_engine, OutNorm, read("out_norm.weight", new[] { d }), read("out_norm.bias", new[] { d }));
        for (int q = 0; q < c.Codebooks; q++)
            TorchParameters.Linear(Heads[q], d, c.CodebookSize, read($"linears.{q}.weight", new[] { c.CodebookSize, d }));
    }
}
