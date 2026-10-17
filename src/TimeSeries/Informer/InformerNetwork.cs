using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TimeSeries;

/// <summary>The attention of Informer's encoder and decoder self-attention (Zhou et al. 2021, §3.1).</summary>
public enum InformerAttention
{
    /// <summary>ProbSparse self-attention: only the top-u queries by the sparsity measurement attend.</summary>
    ProbSparse,

    /// <summary>The canonical scaled dot-product attention.</summary>
    Full,
}

/// <summary>The sizes and switches of an Informer network (the reference's <c>Informer</c> constructor).</summary>
internal sealed record InformerConfiguration(
    int EncoderInputs, int DecoderInputs, int Outputs, int ModelDim, int Heads, int EncoderLayers, int DecoderLayers,
    int FeedForwardDim, int Factor, double Dropout, bool Distil, bool Mix, InformerAttention Attention, bool Gelu,
    int TimeFeatures, int LabelLength, int PredictionLength);

/// <summary>
/// The Informer network (Zhou et al., AAAI 2021), as the official code (zhouhaoyi/Informer2020, <c>models/</c>)
/// builds it: a convolutional token embedding with fixed positional and linear time-feature embeddings; an encoder of
/// ProbSparse self-attention layers with self-attention distilling (circular conv, batch norm, ELU, max-pool) between
/// them; a generative decoder over [start tokens; zero placeholders] with masked ProbSparse self-attention ("mix") and
/// full cross-attention; and a linear projection, of which the last prediction-length steps are the forecast.
/// </summary>
/// <remarks>
/// <para>Shared by <see cref="InformerModel{T}"/> and <see cref="AiDotNet.Finance.Forecasting.Transformers.Informer{T}"/>,
/// which differ only in their data interfaces. Its layers are added to the owner's list, so they train and save with
/// the owner.</para>
/// <para>ProbSparse attention samples U = c·⌈ln L_K⌉ keys per query to estimate the sparsity measurement
/// M̄(q, K) = max_j(q·k_j) − Σ_j(q·k_j)/L_K (Eq. 4; the reference divides the sampled sum by L_K), lets the top
/// u = c·⌈ln L_Q⌉ queries attend to every key, and gives the others the mean of V (or, masked, its cumulative sum).
/// Selection is a constant one-hot matrix, so gradients reach the selected queries and the lazy rows exactly as in the
/// reference.</para>
/// </remarks>
internal sealed class InformerNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <summary>The seeds an owner uses for the initialization and for ProbSparse key sampling and dropout, so every
    /// owner configured with the same seed builds and runs the same network.</summary>
    public static (int Initialization, int Sampling) Seeds(int? seed) => (seed ?? 42, (seed ?? 42) + 1);

    /// <summary>The generator an owner passes to an inference forward: a fresh one from the sampling seed, so a
    /// forecast is a function of its input (the reference draws new key samples on every call).</summary>
    public static Random InferenceRandom(int? seed) => RandomHelper.CreateSeededRandom(Seeds(seed).Sampling);

    /// <summary>Draws the sampled key indices <c>[L_Q, U]</c> of one ProbSparse attention (tests replay the reference's).</summary>
    internal Func<int, int, int, int[,]>? KeySampler { get; set; }

    public InformerNetwork(IEngine engine, List<LayerBase<T>> layers, InformerConfiguration configuration)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.ModelDim % c.Heads != 0)
            throw new ArgumentException($"The model width {c.ModelDim} must be divisible by the {c.Heads} heads.");
        if (c.ModelDim % 2 != 0)
            throw new ArgumentException($"The model width {c.ModelDim} must be even for the sinusoidal position embedding.");
        if (c.LabelLength < 0 || c.PredictionLength <= 0)
            throw new ArgumentException("The start-token length must be non-negative and the prediction length positive.");
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
        LayerNormalizationLayer<T> Norm() => Own(new LayerNormalizationLayer<T>(c.ModelDim, 1e-5));
        Attention AttentionLayer() => new(Linear(c.ModelDim), Linear(c.ModelDim), Linear(c.ModelDim), Linear(c.ModelDim));

        EncoderEmbedding = new Embedding(Linear(c.ModelDim), c.TimeFeatures > 0 ? Linear(c.ModelDim) : null);
        DecoderEmbedding = new Embedding(Linear(c.ModelDim), c.TimeFeatures > 0 ? Linear(c.ModelDim) : null);
        for (int i = 0; i < c.EncoderLayers; i++)
        {
            EncoderLayers.Add(new EncoderLayer(AttentionLayer(), Linear(c.FeedForwardDim), Linear(c.ModelDim), Norm(), Norm()));
            if (c.Distil && i < c.EncoderLayers - 1)
                Distilling.Add(new DistilLayer(Linear(c.ModelDim), Own(new BatchNormalizationLayer<T>(c.ModelDim, 1e-5, 0.9))));
        }
        EncoderNorm = Norm();
        for (int i = 0; i < c.DecoderLayers; i++)
            DecoderLayers.Add(new DecoderLayer(AttentionLayer(), AttentionLayer(), Linear(c.FeedForwardDim), Linear(c.ModelDim),
                Norm(), Norm(), Norm()));
        DecoderNorm = Norm();
        Projection = Linear(c.Outputs);
    }

    /// <summary>A <c>DataEmbedding</c>: the circular-conv token embedding (as a linear map of [x_{t−1}, x_t, x_{t+1}])
    /// and the optional time-feature embedding.</summary>
    internal sealed record Embedding(DenseLayer<T> Token, DenseLayer<T>? Time);

    /// <summary>An <c>AttentionLayer</c>'s query, key, value and output projections.</summary>
    internal sealed record Attention(DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value, DenseLayer<T> Output);

    /// <summary>An <c>EncoderLayer</c>: attention, the kernel-1 convolution feed-forward and two norms.</summary>
    internal sealed record EncoderLayer(Attention SelfAttention, DenseLayer<T> Conv1, DenseLayer<T> Conv2,
        LayerNormalizationLayer<T> Norm1, LayerNormalizationLayer<T> Norm2);

    /// <summary>A distilling <c>ConvLayer</c>: circular conv (k = 3), batch norm, ELU and max-pool (k = 3, s = 2).</summary>
    internal sealed record DistilLayer(DenseLayer<T> Conv, BatchNormalizationLayer<T> Norm);

    /// <summary>A <c>DecoderLayer</c>: masked self-attention, cross-attention, the feed-forward and three norms.</summary>
    internal sealed record DecoderLayer(Attention SelfAttention, Attention CrossAttention, DenseLayer<T> Conv1,
        DenseLayer<T> Conv2, LayerNormalizationLayer<T> Norm1, LayerNormalizationLayer<T> Norm2, LayerNormalizationLayer<T> Norm3);

    public InformerConfiguration Configuration { get; }
    public Embedding EncoderEmbedding { get; }
    public Embedding DecoderEmbedding { get; }
    public List<EncoderLayer> EncoderLayers { get; } = new();
    public List<DistilLayer> Distilling { get; } = new();
    public LayerNormalizationLayer<T> EncoderNorm { get; }
    public List<DecoderLayer> DecoderLayers { get; } = new();
    public LayerNormalizationLayer<T> DecoderNorm { get; }
    public DenseLayer<T> Projection { get; }

    // ------------------------------------------------------------------ the forward

    /// <summary>
    /// The forecast <c>[B, PredictionLength, Outputs]</c> of the encoder input <c>[B, L, EncoderInputs]</c> and the
    /// decoder input <c>[B, LabelLength + PredictionLength, DecoderInputs]</c> (start tokens, then zeros), with their
    /// time features <c>[B, ·, TimeFeatures]</c> when the network has them.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> encoderInput, Tensor<T>? encoderMarks, Tensor<T> decoderInput, Tensor<T>? decoderMarks,
        bool training, Random random)
    {
        var c = Configuration;
        if (decoderInput.Shape[1] != c.LabelLength + c.PredictionLength)
            throw new ArgumentException(
                $"The decoder input has {decoderInput.Shape[1]} steps; it needs {c.LabelLength} start tokens and {c.PredictionLength} placeholders.");
        var x = Embed(EncoderEmbedding, encoderInput, encoderMarks, training, random);
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            x = Encode(EncoderLayers[i], x, training, random);
            if (i < Distilling.Count) x = Distil(Distilling[i], x, training);
        }
        var memory = EncoderNorm.Forward(x);

        var y = Embed(DecoderEmbedding, decoderInput, decoderMarks, training, random);
        foreach (var layer in DecoderLayers) y = Decode(layer, y, memory, training, random);
        y = Dense(Projection, DecoderNorm.Forward(y));
        int steps = y.Shape[1];
        return _engine.TensorSlice(y, new[] { 0, steps - c.PredictionLength, 0 }, new[] { y.Shape[0], c.PredictionLength, c.Outputs });
    }

    /// <summary>The reference's decoder input: the last <see cref="InformerConfiguration.LabelLength"/> steps of
    /// <paramref name="history"/> <c>[B, L, C]</c>, then zeros for the prediction steps.</summary>
    public Tensor<T> DecoderInput(Tensor<T> history)
    {
        var c = Configuration;
        int batch = history.Shape[0], length = history.Shape[1], channels = history.Shape[2];
        if (c.LabelLength > length)
            throw new ArgumentException($"The start tokens ({c.LabelLength}) are longer than the history ({length}).");
        var zeros = new Tensor<T>(new[] { batch, c.PredictionLength, channels });
        if (c.LabelLength == 0) return zeros;
        var label = _engine.TensorSlice(history, new[] { 0, length - c.LabelLength, 0 }, new[] { batch, c.LabelLength, channels });
        return _engine.TensorConcatenate(new[] { label, zeros }, 1);
    }

    // ------------------------------------------------------------------ building blocks

    // A dense layer over the last axis of [B, L, F].
    private Tensor<T> Dense(DenseLayer<T> layer, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var y = layer.Forward(_engine.Reshape(x, new[] { batch * length, x.Shape[2] }));
        return _engine.Reshape(y, new[] { batch, length, y.Shape[1] });
    }

    // Inverted dropout; the mask is a per-step draw, so a compiled training plan redraws it on every replay.
    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = Configuration.Dropout;
        if (!training || rate <= 0) return x;
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        var mask = AiDotNet.Training.CompiledStepRandom<T>.Draw(x._shape, m =>
        {
            var span = m.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        });
        return _engine.TensorMultiply(x, mask);
    }

    private Tensor<T> Activation(Tensor<T> x) => Configuration.Gelu ? ActivationHelper.ExactGelu(x, _engine) : _engine.ReLU(x);

    // [x_{t−1}, x_t, x_{t+1}] with circular wrap: a kernel-3 circular convolution is a linear map of this.
    private Tensor<T> CircularNeighbours(Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], channels = x.Shape[2];
        if (length == 1) return _engine.TensorConcatenate(new[] { x, x, x }, 2);
        Tensor<T> Steps(int from, int count) => _engine.TensorSlice(x, new[] { 0, from, 0 }, new[] { batch, count, channels });
        var previous = _engine.TensorConcatenate(new[] { Steps(length - 1, 1), Steps(0, length - 1) }, 1);
        var next = _engine.TensorConcatenate(new[] { Steps(1, length - 1), Steps(0, 1) }, 1);
        return _engine.TensorConcatenate(new[] { previous, x, next }, 2);
    }

    // The fixed sinusoidal position embedding [L, D] (PositionalEmbedding: sin on even, cos on odd features).
    private Tensor<T> Positions(int length)
    {
        int d = Configuration.ModelDim;
        var table = new Tensor<T>(new[] { 1, length, d });
        for (int p = 0; p < length; p++)
            for (int i = 0; i < d; i += 2)
            {
                double angle = p * Math.Exp(i * -(Math.Log(10000.0) / d));
                table[0, p, i] = NumOps.FromDouble(Math.Sin(angle));
                table[0, p, i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return table;
    }

    private Tensor<T> Embed(Embedding e, Tensor<T> x, Tensor<T>? marks, bool training, Random random)
    {
        int batch = x.Shape[0], length = x.Shape[1];
        var sum = _engine.TensorAdd(Dense(e.Token, CircularNeighbours(x)),
            _engine.TensorTile(Positions(length), new[] { batch, 1, 1 }));
        if (e.Time is not null)
        {
            if (marks is null || marks.Shape[1] != length || marks.Shape[2] != Configuration.TimeFeatures)
                throw new ArgumentException($"The network embeds {Configuration.TimeFeatures} time features per step; supply them for all {length} steps.");
            sum = _engine.TensorAdd(sum, Dense(e.Time, marks));
        }
        return Drop(sum, training, random);
    }

    // [B, L, D] through a projection to heads [B·H, L, E].
    private Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> x)
    {
        int batch = x.Shape[0], length = x.Shape[1], heads = Configuration.Heads, e = Configuration.ModelDim / heads;
        var y = _engine.Reshape(Dense(projection, x), new[] { batch, length, heads, e });
        return _engine.Reshape(_engine.TensorPermute(y, new[] { 0, 2, 1, 3 }), new[] { batch * heads, length, e });
    }

    // An AttentionLayer: projections, the inner attention, the (mixed) head merge and the output projection.
    private Tensor<T> Attend(Attention a, Tensor<T> queries, Tensor<T> keys, bool prob, bool masked, bool mix, bool training, Random random)
    {
        int batch = queries.Shape[0], lq = queries.Shape[1], heads = Configuration.Heads, e = Configuration.ModelDim / heads;
        var q = Heads(a.Query, queries);
        var k = Heads(a.Key, keys);
        var v = Heads(a.Value, keys);
        var context = prob ? ProbAttention(q, k, v, masked, random) : FullAttention(q, k, v, masked, training, random);   // [B·H, Lq, E]
        var perHead = _engine.Reshape(context, new[] { batch, heads, lq, e });
        // The inner attention returns [B, L, H, E]; mix transposes it back to [B, H, L, E] before the flat view.
        var merged = mix
            ? _engine.Reshape(perHead, new[] { batch, lq, heads * e })
            : _engine.Reshape(_engine.TensorPermute(perHead, new[] { 0, 2, 1, 3 }), new[] { batch, lq, heads * e });
        return Dense(a.Output, merged);
    }

    private static Tensor<T> CausalMask(int rows, int columns, Func<int, int> position)
    {
        var mask = new Tensor<T>(new[] { 1, rows, columns });
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int i = 0; i < rows; i++)
            for (int j = position(i) + 1; j < columns; j++) mask[0, i, j] = blocked;
        return mask;
    }

    // FullAttention: softmax(scale · Q Kᵀ), causal when masked, with attention dropout.
    private Tensor<T> FullAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, bool masked, bool training, Random random)
    {
        int bh = q.Shape[0], lq = q.Shape[1], lk = k.Shape[1];
        var scores = _engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 }));
        if (masked) scores = _engine.TensorAdd(scores, _engine.TensorTile(CausalMask(lq, lk, i => i), new[] { bh, 1, 1 }));
        var weights = _engine.Softmax(_engine.TensorMultiplyScalar(scores, NumOps.FromDouble(1.0 / Math.Sqrt(q.Shape[2]))), axis: 2);
        return _engine.BatchMatMul(Drop(weights, training, random), v);
    }

    // ProbAttention (no attention dropout, as in the reference).
    private Tensor<T> ProbAttention(Tensor<T> q, Tensor<T> k, Tensor<T> v, bool masked, Random random)
    {
        int bh = q.Shape[0], lq = q.Shape[1], lk = k.Shape[1], e = q.Shape[2];
        int factor = Configuration.Factor;
        int samples = Math.Min(factor * (int)Math.Ceiling(Math.Log(lk)), lk);
        int top = Math.Min(factor * (int)Math.Ceiling(Math.Log(lq)), lq);
        if (masked && lq != lk) throw new InvalidOperationException("Masked ProbSparse attention is self-attention: L_Q must equal L_K.");

        // The sampled keys per query, shared by every batch row and head (torch.randint(L_K, (L_Q, U))): a per-step
        // draw, so a compiled training plan samples new keys on every replay.
        var sampled = SampledKeys(lq, samples, lk, random);

        // The sparsity measurement and the top-u queries per row, chosen from the step's own queries and keys (the
        // selection is not differentiated); a compiled plan chooses again on every replay.
        var select = AiDotNet.Training.CompiledStepHostValue<T>.Compute("InformerProbSparseQueries", new[] { q, k, sampled },
            new[] { bh, top, lq }, (inputs, result) => SelectActiveQueries(inputs[0], inputs[1], inputs[2], top, result));

        // The chosen queries attend to every key (masked: to keys at or before their own position).
        var reduced = _engine.BatchMatMul(select, q);                                                          // [BH, u, E]
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(reduced, _engine.TensorPermute(k, new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        if (masked)
        {
            // Each chosen query sees the keys at or before its own position.
            var mask = AiDotNet.Training.CompiledStepHostValue<T>.Compute("InformerProbSparseCausalMask", new[] { select },
                new[] { bh, top, lk }, (inputs, result) => CausalMaskOfChosen(inputs[0], lk, result));
            scores = _engine.TensorAdd(scores, mask);
        }
        var updated = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), v);                                // [BH, u, E]

        // Every other query keeps the initial context: mean(V), or masked, the cumulative sum of V.
        var lazy = new Tensor<T>(new[] { 1, lq, lk });
        for (int i = 0; i < lq; i++)
            for (int j = 0; j < lk; j++)
                if (masked ? j <= i : true) lazy[0, i, j] = masked ? NumOps.One : NumOps.FromDouble(1.0 / lk);
        var initial = _engine.BatchMatMul(_engine.TensorTile(lazy, new[] { bh, 1, 1 }), v);                    // [BH, Lq, E]
        var selectT = _engine.TensorPermute(select, new[] { 0, 2, 1 });
        return _engine.TensorAdd(initial, _engine.BatchMatMul(selectT, _engine.TensorSubtract(updated, _engine.BatchMatMul(select, initial))));
    }

    // The sampled key of every (query, sample) as a [L_Q, U] tensor of indices: KeySampler's when set, otherwise a
    // per-step draw in the reference's order.
    private Tensor<T> SampledKeys(int queries, int samples, int keys, Random random)
    {
        if (KeySampler is not null)
        {
            var fixedKeys = KeySampler(queries, samples, keys);
            var tensor = new Tensor<T>(new[] { queries, samples });
            for (int i = 0; i < queries; i++)
                for (int s = 0; s < samples; s++) tensor[i, s] = NumOps.FromDouble(fixedKeys[i, s]);
            return tensor;
        }
        return AiDotNet.Training.CompiledStepRandom<T>.Draw(new[] { queries, samples }, t =>
        {
            var span = t.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble(random.Next(keys));
        });
    }

    // The one-hot rows [BH, u, L_Q] of the u queries with the largest sparsity measurement
    // M(q_i) = max_s q_i·k_{j_s} − Σ_s q_i·k_{j_s} / L_K over the sampled keys j_s (ties keep the earlier query).
    private static void SelectActiveQueries(Tensor<T> q, Tensor<T> k, Tensor<T> sampledKeys, int top, Tensor<T> result)
    {
        int bh = q.Shape[0], lq = q.Shape[1], e = q.Shape[2], lk = k.Shape[1], samples = sampledKeys.Shape[1];
        var qd = q.GetCpuData();
        var kd = k.GetCpuData();
        var sd = sampledKeys.GetCpuData();
        var keys = new int[lq * samples];
        for (int i = 0; i < keys.Length; i++) keys[i] = (int)Math.Round(NumOps.ToDouble(sd[i]));
        var span = result.AsWritableSpan();
        span.Clear();
        var measure = new double[lq];
        for (int r = 0; r < bh; r++)
        {
            for (int i = 0; i < lq; i++)
            {
                double max = double.NegativeInfinity, sum = 0;
                for (int s = 0; s < samples; s++)
                {
                    int j = keys[i * samples + s];
                    double dot = 0;
                    for (int t = 0; t < e; t++)
                        dot += NumOps.ToDouble(qd[(r * lq + i) * e + t]) * NumOps.ToDouble(kd[(r * lk + j) * e + t]);
                    max = Math.Max(max, dot);
                    sum += dot;
                }
                measure[i] = max - sum / lk;
            }
            var chosen = Enumerable.Range(0, lq).OrderByDescending(i => measure[i]).Take(top).ToArray();
            for (int u = 0; u < top; u++) span[(r * top + u) * lq + chosen[u]] = NumOps.One;
        }
        result.IncrementVersion();
    }

    // −∞ on the keys after each chosen query's position, from the one-hot choice [BH, u, L_Q].
    private static void CausalMaskOfChosen(Tensor<T> select, int keys, Tensor<T> result)
    {
        int bh = select.Shape[0], top = select.Shape[1], lq = select.Shape[2];
        var sd = select.GetCpuData();
        var span = result.AsWritableSpan();
        span.Clear();
        var blocked = NumOps.FromDouble(double.NegativeInfinity);
        for (int r = 0; r < bh; r++)
            for (int u = 0; u < top; u++)
            {
                int position = 0;
                while (position < lq - 1 && NumOps.ToDouble(sd[(r * top + u) * lq + position]) == 0) position++;
                for (int j = position + 1; j < keys; j++) span[(r * top + u) * keys + j] = blocked;
            }
        result.IncrementVersion();
    }

    private Tensor<T> FeedForward(DenseLayer<T> conv1, DenseLayer<T> conv2, Tensor<T> x, bool training, Random random)
        => Drop(Dense(conv2, Drop(Activation(Dense(conv1, x)), training, random)), training, random);

    private Tensor<T> Encode(EncoderLayer l, Tensor<T> x, bool training, Random random)
    {
        bool prob = Configuration.Attention == InformerAttention.ProbSparse;
        x = l.Norm1.Forward(_engine.TensorAdd(x, Drop(Attend(l.SelfAttention, x, x, prob, masked: false, mix: false, training, random), training, random)));
        return l.Norm2.Forward(_engine.TensorAdd(x, FeedForward(l.Conv1, l.Conv2, x, training, random)));
    }

    private Tensor<T> Decode(DecoderLayer l, Tensor<T> y, Tensor<T> memory, bool training, Random random)
    {
        bool prob = Configuration.Attention == InformerAttention.ProbSparse;
        y = l.Norm1.Forward(_engine.TensorAdd(y, Drop(Attend(l.SelfAttention, y, y, prob, masked: true, mix: Configuration.Mix, training, random), training, random)));
        y = l.Norm2.Forward(_engine.TensorAdd(y, Drop(Attend(l.CrossAttention, y, memory, prob: false, masked: false, mix: false, training, random), training, random)));
        return l.Norm3.Forward(_engine.TensorAdd(y, FeedForward(l.Conv1, l.Conv2, y, training, random)));
    }

    // ConvLayer: circular conv, BatchNorm1d over the channels, ELU, MaxPool1d(3, 2, padding 1) over time.
    private Tensor<T> Distil(DistilLayer l, Tensor<T> x, bool training)
    {
        int batch = x.Shape[0], length = x.Shape[1], d = x.Shape[2];
        var conv = Dense(l.Conv, CircularNeighbours(x));
        l.Norm.SetTrainingMode(training);
        var normed = _engine.Reshape(l.Norm.Forward(_engine.Reshape(conv, new[] { batch * length, d })), new[] { batch, length, d });
        var activated = _engine.ELU(normed);
        int pooled = (length - 1) / 2 + 1;
        // Padding never wins the max: ELU outputs are above −1.
        var padding = new Tensor<T>(new[] { batch, 1, d });
        padding.Fill(NumOps.FromDouble(-1e30));
        var padded = _engine.TensorConcatenate(new[] { padding, activated, padding }, 1);                       // [B, L + 2, D]
        var windows = new Tensor<T>[3];
        for (int w = 0; w < 3; w++)
        {
            var pick = new Tensor<T>(new[] { 1, pooled, length + 2 });
            for (int i = 0; i < pooled; i++) pick[0, i, 2 * i + w] = NumOps.One;
            windows[w] = _engine.Reshape(_engine.BatchMatMul(_engine.TensorTile(pick, new[] { batch, 1, 1 }), padded), new[] { batch, pooled, 1, d });
        }
        return _engine.ReduceMax(_engine.TensorConcatenate(windows, 2), new[] { 2 }, false, out _);
    }

    // ------------------------------------------------------------------ initialization and loading

    /// <summary>The reference's initialization: PyTorch's default for <c>nn.Linear</c>/<c>nn.Conv1d</c>
    /// (U(±1/√fan_in) for weights and biases), Kaiming-normal (fan_in, leaky_relu) token convolutions, unit norms.</summary>
    public void InitializeLikeReference(Random random)
    {
        var c = Configuration;
        void Uniform(DenseLayer<T> layer, int inputs, int outputs)
        {
            double bound = 1.0 / Math.Sqrt(inputs);
            var w = new double[outputs * inputs];
            var b = new double[outputs];
            for (int i = 0; i < w.Length; i++) w[i] = (2 * random.NextDouble() - 1) * bound;
            for (int i = 0; i < b.Length; i++) b[i] = (2 * random.NextDouble() - 1) * bound;
            TorchParameters.Linear(_engine, layer, inputs, outputs, w, b);
        }
        void TokenConv(DenseLayer<T> layer, int channels)
        {
            int fanIn = 3 * channels;
            double std = Math.Sqrt(2.0) / Math.Sqrt(fanIn), bound = 1.0 / Math.Sqrt(fanIn);
            var w = new double[c.ModelDim * fanIn];
            var b = new double[c.ModelDim];
            for (int i = 0; i < w.Length; i++) w[i] = std * Gaussian(random);
            for (int i = 0; i < b.Length; i++) b[i] = (2 * random.NextDouble() - 1) * bound;
            TorchParameters.Linear(_engine, layer, fanIn, c.ModelDim, w, b);
        }
        void AttentionInit(Attention a)
        {
            Uniform(a.Query, c.ModelDim, c.ModelDim);
            Uniform(a.Key, c.ModelDim, c.ModelDim);
            Uniform(a.Value, c.ModelDim, c.ModelDim);
            Uniform(a.Output, c.ModelDim, c.ModelDim);
        }
        TokenConv(EncoderEmbedding.Token, c.EncoderInputs);
        if (EncoderEmbedding.Time is not null) Uniform(EncoderEmbedding.Time, c.TimeFeatures, c.ModelDim);
        TokenConv(DecoderEmbedding.Token, c.DecoderInputs);
        if (DecoderEmbedding.Time is not null) Uniform(DecoderEmbedding.Time, c.TimeFeatures, c.ModelDim);
        foreach (var l in EncoderLayers)
        {
            AttentionInit(l.SelfAttention);
            Uniform(l.Conv1, c.ModelDim, c.FeedForwardDim);
            Uniform(l.Conv2, c.FeedForwardDim, c.ModelDim);
        }
        foreach (var l in Distilling) Uniform(l.Conv, 3 * c.ModelDim, c.ModelDim);
        foreach (var l in DecoderLayers)
        {
            AttentionInit(l.SelfAttention);
            AttentionInit(l.CrossAttention);
            Uniform(l.Conv1, c.ModelDim, c.FeedForwardDim);
            Uniform(l.Conv2, c.FeedForwardDim, c.ModelDim);
        }
        Uniform(Projection, c.ModelDim, c.Outputs);
    }

    private static double Gaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Loads parameters named as the reference's <c>Informer</c> state dict.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        void Lin(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs }), read($"{name}.bias", new[] { outputs }));
        // Conv1d weight [out, in, k] → the linear map of [x_{t−1}, x_t, x_{t+1}]: column k·in + c.
        void Conv(DenseLayer<T> layer, string name, int inputs, int outputs)
        {
            var w = read($"{name}.weight", new[] { outputs, inputs, 3 });
            var unfolded = new double[outputs * 3 * inputs];
            for (int o = 0; o < outputs; o++)
                for (int ch = 0; ch < inputs; ch++)
                    for (int k = 0; k < 3; k++) unfolded[o * 3 * inputs + k * inputs + ch] = w[(o * inputs + ch) * 3 + k];
            TorchParameters.Linear(_engine, layer, 3 * inputs, outputs, unfolded, read($"{name}.bias", new[] { outputs }));
        }
        // Kernel-1 Conv1d weight [out, in, 1] is a linear weight.
        void Pointwise(DenseLayer<T> layer, string name, int inputs, int outputs)
            => TorchParameters.Linear(_engine, layer, inputs, outputs, read($"{name}.weight", new[] { outputs, inputs, 1 }), read($"{name}.bias", new[] { outputs }));
        void Norm(LayerNormalizationLayer<T> layer, string name)
            => TorchParameters.LayerNorm(_engine, layer, read($"{name}.weight", new[] { d }), read($"{name}.bias", new[] { d }));
        void Attn(Attention a, string name)
        {
            Lin(a.Query, $"{name}.query_projection", d, d);
            Lin(a.Key, $"{name}.key_projection", d, d);
            Lin(a.Value, $"{name}.value_projection", d, d);
            Lin(a.Output, $"{name}.out_projection", d, d);
        }
        void Embedding(Embedding e, string name, int channels)
        {
            Conv(e.Token, $"{name}.value_embedding.tokenConv", channels, d);
            if (e.Time is not null) Lin(e.Time, $"{name}.temporal_embedding.embed", c.TimeFeatures, d);
        }
        Embedding(EncoderEmbedding, "enc_embedding", c.EncoderInputs);
        Embedding(DecoderEmbedding, "dec_embedding", c.DecoderInputs);
        for (int i = 0; i < EncoderLayers.Count; i++)
        {
            var l = EncoderLayers[i];
            string p = $"encoder.attn_layers.{i}";
            Attn(l.SelfAttention, $"{p}.attention");
            Pointwise(l.Conv1, $"{p}.conv1", d, c.FeedForwardDim);
            Pointwise(l.Conv2, $"{p}.conv2", c.FeedForwardDim, d);
            Norm(l.Norm1, $"{p}.norm1");
            Norm(l.Norm2, $"{p}.norm2");
        }
        for (int i = 0; i < Distilling.Count; i++)
        {
            string p = $"encoder.conv_layers.{i}";
            Conv(Distilling[i].Conv, $"{p}.downConv", d, d);
            // The layer's vector is gamma, beta, then the running mean and variance buffers.
            var parts = new[] { "weight", "bias", "running_mean", "running_var" }.Select(n => read($"{p}.norm.{n}", new[] { d })).ToArray();
            var parameters = new Vector<T>(4 * d);
            for (int part = 0; part < 4; part++)
                for (int j = 0; j < d; j++) parameters[part * d + j] = NumOps.FromDouble(parts[part][j]);
            Distilling[i].Norm.SetParameters(parameters);
        }
        Norm(EncoderNorm, "encoder.norm");
        for (int i = 0; i < DecoderLayers.Count; i++)
        {
            var l = DecoderLayers[i];
            string p = $"decoder.layers.{i}";
            Attn(l.SelfAttention, $"{p}.self_attention");
            Attn(l.CrossAttention, $"{p}.cross_attention");
            Pointwise(l.Conv1, $"{p}.conv1", d, c.FeedForwardDim);
            Pointwise(l.Conv2, $"{p}.conv2", c.FeedForwardDim, d);
            Norm(l.Norm1, $"{p}.norm1");
            Norm(l.Norm2, $"{p}.norm2");
            Norm(l.Norm3, $"{p}.norm3");
        }
        Norm(DecoderNorm, "decoder.norm");
        Lin(Projection, "projection", d, c.Outputs);
    }
}
