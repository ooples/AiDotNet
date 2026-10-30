using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Audio.Whisper;

/// <summary>The shape of a Whisper model (openai-whisper's <c>ModelDimensions</c>).</summary>
internal sealed record WhisperDimensions(int Mels, int AudioContext, int AudioState, int AudioHeads, int AudioLayers, int Vocabulary,
    int TextContext, int TextState, int TextHeads, int TextLayers)
{
    /// <summary>large-v3-turbo ("turbo"), which OuteTTS 1.0 times words with.</summary>
    public static WhisperDimensions LargeV3Turbo => new(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 4);

    /// <summary>Multilingual (a vocabulary of at least 51,865 tokens).</summary>
    public bool IsMultilingual => Vocabulary >= 51865;

    /// <summary>The language tokens (99, or 100 from large-v3 on).</summary>
    public int NumLanguages => Vocabulary - 51765 - (IsMultilingual ? 1 : 0);

    /// <summary>Logits through a separate biased projection (<c>decoder.out_proj</c>) instead of the token table, as
    /// MegaTTS 3's aligner computes them.</summary>
    public bool UntiedOutput { get; init; }

    /// <summary>Encodes audio of any length up to the audio context (the positions cut to it), as MegaTTS 3's aligner does,
    /// instead of requiring the 30-second window.</summary>
    public bool VariableAudioLength { get; init; }

    /// <summary>Builds and loads the audio encoder only (LLaMA-Omni's speech encoder, <c>whisper.load_model(...).encoder</c>);
    /// the text decoder and its weights are left out.</summary>
    public bool EncoderOnly { get; init; }
}

/// <summary>
/// Whisper (Radford et al. 2022) as openai-whisper 20240930's <c>model.py</c> computes it: the audio encoder (two GELU
/// convolutions, the second with stride 2, fixed sinusoidal positions, pre-norm blocks), the text decoder (learned
/// positions, causal self-attention and cross-attention, logits through the token table), attention with the query and
/// key each scaled by d^(-1/4), a key projection without bias, a KV cache for decoding, and the cross-attention scores of
/// every layer and head kept for word timing.
/// </summary>
/// <remarks>Parameter names follow openai-whisper's checkpoints (<c>encoder.conv1</c>, <c>encoder.blocks.{i}.attn.query</c>,
/// <c>decoder.token_embedding</c>, ...).</remarks>
internal sealed class OpenAiWhisperNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed class Attention
    {
        public BiasFreeLinearLayer<T> Query = null!, Key = null!, Value = null!, Out = null!;
        public BiasFreeLinearLayer<T> QueryBias = null!, ValueBias = null!, OutBias = null!;
    }

    internal sealed class Block
    {
        public Attention SelfAttention = null!;
        public Attention? CrossAttention;
        public BiasFreeLinearLayer<T> AttentionNormWeight = null!, AttentionNormBias = null!;
        public BiasFreeLinearLayer<T>? CrossNormWeight, CrossNormBias;
        public BiasFreeLinearLayer<T> MlpNormWeight = null!, MlpNormBias = null!;
        public BiasFreeLinearLayer<T> Mlp1 = null!, Mlp1Bias = null!, Mlp2 = null!, Mlp2Bias = null!;
    }

    public OpenAiWhisperNetwork(IEngine engine, List<LayerBase<T>> layers, WhisperDimensions dimensions, Random initialization)
    {
        _engine = engine;
        Dimensions = dimensions;
        var d = dimensions;
        if (d.AudioState % d.AudioHeads != 0 || d.TextState % d.TextHeads != 0)
            throw new ArgumentException("The widths must divide by the heads.", nameof(dimensions));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            layers.Add(layer);
            return layer;
        }
        BiasFreeLinearLayer<T> Linear(int inputs, int outputs) => Add(new BiasFreeLinearLayer<T>(inputs, outputs));
        BiasFreeLinearLayer<T> Values(int count) => Add(new BiasFreeLinearLayer<T>(1, count));
        Attention NewAttention(int n) => new()
        {
            Query = Linear(n, n), QueryBias = Values(n), Key = Linear(n, n), Value = Linear(n, n), ValueBias = Values(n), Out = Linear(n, n), OutBias = Values(n),
        };
        Block NewBlock(int n, bool cross) => new()
        {
            SelfAttention = NewAttention(n), AttentionNormWeight = Values(n), AttentionNormBias = Values(n),
            CrossAttention = cross ? NewAttention(n) : null, CrossNormWeight = cross ? Values(n) : null, CrossNormBias = cross ? Values(n) : null,
            Mlp1 = Linear(n, 4 * n), Mlp1Bias = Values(4 * n), Mlp2 = Linear(4 * n, n), Mlp2Bias = Values(n), MlpNormWeight = Values(n), MlpNormBias = Values(n),
        };
        Conv1 = Add(new NormedConv1DLayer<T>(d.Mels, d.AudioState, 3, 1, 1, 1, 1, false, ConvolutionNormalization.None));
        Conv2 = Add(new NormedConv1DLayer<T>(d.AudioState, d.AudioState, 3, 2, 1, 1, 1, false, ConvolutionNormalization.None));
        AudioPositions = Values(d.AudioContext * d.AudioState);
        for (int i = 0; i < d.AudioLayers; i++) EncoderBlocks.Add(NewBlock(d.AudioState, false));
        PostNormWeight = Values(d.AudioState);
        PostNormBias = Values(d.AudioState);
        if (!d.EncoderOnly)
        {
            TokenEmbedding = Values(d.Vocabulary * d.TextState);
            TextPositions = Values(d.TextContext * d.TextState);
            for (int i = 0; i < d.TextLayers; i++) DecoderBlocks.Add(NewBlock(d.TextState, true));
            NormWeight = Values(d.TextState);
            NormBias = Values(d.TextState);
        }
        if (d.UntiedOutput && !d.EncoderOnly)
        {
            OutputProjection = Linear(d.TextState, d.Vocabulary);
            OutputBias = Values(d.Vocabulary);
        }
        InitializeLikeReference(initialization);
    }

    public WhisperDimensions Dimensions { get; }
    public NormedConv1DLayer<T> Conv1 { get; }
    public NormedConv1DLayer<T> Conv2 { get; }
    public BiasFreeLinearLayer<T> AudioPositions { get; }
    public List<Block> EncoderBlocks { get; } = new();
    public BiasFreeLinearLayer<T> PostNormWeight { get; }
    public BiasFreeLinearLayer<T> PostNormBias { get; }
    /// <summary>The text decoder's token table; null for an encoder-only network, as are its positions and final norm.</summary>
    public BiasFreeLinearLayer<T>? TokenEmbedding { get; }
    public BiasFreeLinearLayer<T>? TextPositions { get; }
    public List<Block> DecoderBlocks { get; } = new();
    public BiasFreeLinearLayer<T>? NormWeight { get; }
    public BiasFreeLinearLayer<T>? NormBias { get; }
    public BiasFreeLinearLayer<T>? OutputProjection { get; }
    public BiasFreeLinearLayer<T>? OutputBias { get; }

    /// <summary>The alignment heads (layer, head) whose cross-attention times words; by default every head of the
    /// decoder's second half, as openai-whisper sets them before a checkpoint names its own.</summary>
    public List<(int Layer, int Head)> AlignmentHeads { get; set; } = new();

    // ---------------------------------------------------------------- helpers

    private Tensor<T> Value(BiasFreeLinearLayer<T> holder) => holder.Forward(new Tensor<T>(new[] { 1, 1 }, new Vector<T>(new[] { NumOps.One })));

    private Tensor<T> Linear(BiasFreeLinearLayer<T> weight, BiasFreeLinearLayer<T>? bias, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = weight.Forward(_engine.Reshape(x, new[] { rows, last }));
        if (bias is not null) y = _engine.TensorAdd(y, _engine.TensorBroadcastTo(_engine.Reshape(Value(bias), new[] { 1, y.Shape[1] }), y.Shape.ToArray()));
        shape[^1] = y.Shape[1];
        return _engine.Reshape(y, shape);
    }

    // nn.LayerNorm (eps 1e-5) over the last axis.
    private Tensor<T> LayerNorm(BiasFreeLinearLayer<T> weight, BiasFreeLinearLayer<T> bias, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int rank = shape.Length, width = shape[^1];
        var mean = _engine.ReduceMean(x, new[] { rank - 1 }, true);
        var centered = _engine.TensorSubtract(x, _engine.TensorBroadcastTo(mean, shape));
        var variance = _engine.ReduceMean(_engine.TensorMultiply(centered, centered), new[] { rank - 1 }, true);
        var inverse = _engine.TensorPower(_engine.TensorAddScalar(variance, NumOps.FromDouble(1e-5)), NumOps.FromDouble(-0.5));
        var normalized = _engine.TensorMultiply(centered, _engine.TensorBroadcastTo(inverse, shape));
        var lead = Enumerable.Repeat(1, rank - 1).Concat(new[] { width }).ToArray();
        return _engine.TensorAdd(_engine.TensorMultiply(normalized, _engine.TensorBroadcastTo(_engine.Reshape(Value(weight), lead), shape)),
            _engine.TensorBroadcastTo(_engine.Reshape(Value(bias), lead), shape));
    }

    private Tensor<T> Gelu(Tensor<T> x) => ActivationHelper.ExactGelu(x, _engine);

    /// <summary>The keys and values a block's attention reads: computed from the source, or held over from earlier steps.</summary>
    internal sealed class Cache
    {
        public readonly Dictionary<Attention, (Tensor<T> Keys, Tensor<T> Values)> Entries = new();

        /// <summary>Positions read so far by the decoder.</summary>
        public int Length;
    }

    // MultiHeadAttention: q, k, v of [1, n, d]; scaled scores, softmax, output projection. Returns the output and the scores.
    private (Tensor<T> Output, Tensor<T> Scores) Attend(Attention a, Tensor<T> x, Tensor<T>? source, int heads, bool causal, Cache? cache)
    {
        int n = x.Shape[1], d = x.Shape[2], hd = d / heads;
        var q = Linear(a.Query, a.QueryBias, x);
        Tensor<T> k, v;
        if (source is not null && cache is not null && cache.Entries.TryGetValue(a, out var held))
        {
            (k, v) = held;   // cross-attention: the audio's keys and values, computed once
        }
        else
        {
            var input = source ?? x;
            k = Linear(a.Key, null, input);
            v = Linear(a.Value, a.ValueBias, input);
            if (cache is not null)
            {
                if (source is null && cache.Entries.TryGetValue(a, out var past))
                {
                    k = _engine.TensorConcatenate(new[] { past.Keys, k }, 1);
                    v = _engine.TensorConcatenate(new[] { past.Values, v }, 1);
                }
                cache.Entries[a] = (k, v);
            }
        }
        int m = k.Shape[1];
        double scale = Math.Pow(hd, -0.25);
        Tensor<T> Heads(Tensor<T> t, int length) => _engine.Reshape(_engine.TensorPermute(_engine.Reshape(t, new[] { 1, length, heads, hd }), new[] { 0, 2, 1, 3 }), new[] { heads, length, hd });
        var qh = _engine.TensorMultiplyScalar(Heads(q, n), NumOps.FromDouble(scale));
        var kh = _engine.TensorMultiplyScalar(Heads(k, m), NumOps.FromDouble(scale));
        var scores = _engine.BatchMatMul(qh, _engine.TensorPermute(kh, new[] { 0, 2, 1 }));   // [heads, n, m]
        if (causal && n > 1)
        {
            // mask[:n_ctx, :n_ctx]: -inf above the diagonal (only when the queries start at position 0, as the reference's
            // decoding always does).
            var mask = new Tensor<T>(new[] { 1, n, m });
            for (int i = 0; i < n; i++)
                for (int j = i + 1; j < Math.Min(n, m); j++) mask[0, i, j] = NumOps.FromDouble(double.NegativeInfinity);
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { heads, n, m }));
        }
        var weights = _engine.Softmax(scores, axis: 2);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.Reshape(_engine.BatchMatMul(weights, Heads(v, m)), new[] { 1, heads, n, hd }), new[] { 0, 2, 1, 3 }), new[] { 1, n, d });
        return (Linear(a.Out, a.OutBias, context), scores);
    }

    private Tensor<T> RunBlock(Block b, Tensor<T> x, Tensor<T>? audio, int heads, bool causal, Cache? cache, List<Tensor<T>>? crossScores)
    {
        x = _engine.TensorAdd(x, Attend(b.SelfAttention, LayerNorm(b.AttentionNormWeight, b.AttentionNormBias, x), null, heads, causal, cache).Output);
        if (b.CrossAttention is not null)
        {
            var (output, scores) = Attend(b.CrossAttention, LayerNorm(b.CrossNormWeight!, b.CrossNormBias!, x), audio, heads, false, cache);
            crossScores?.Add(scores);
            x = _engine.TensorAdd(x, output);
        }
        var hidden = Gelu(Linear(b.Mlp1, b.Mlp1Bias, LayerNorm(b.MlpNormWeight, b.MlpNormBias, x)));
        return _engine.TensorAdd(x, Linear(b.Mlp2, b.Mlp2Bias, hidden));
    }

    /// <summary>The encoder's states <c>[1, audio context, width]</c> of a log-mel window <c>[mels, 2 × audio context]</c>.</summary>
    public Tensor<T> Encode(Tensor<T> mel)
    {
        var d = Dimensions;
        var x = Gelu(Conv1.Forward(_engine.Reshape(mel, new[] { 1, d.Mels, mel.Shape[^1] })));
        x = Gelu(Conv2.Forward(x));
        x = _engine.TensorPermute(x, new[] { 0, 2, 1 });
        int frames = x.Shape[1];
        if (d.VariableAudioLength ? frames > d.AudioContext : frames != d.AudioContext) throw new ArgumentException("incorrect audio shape", nameof(mel));
        var positions = _engine.Reshape(Value(AudioPositions), new[] { 1, d.AudioContext, d.AudioState });
        if (frames != d.AudioContext) positions = _engine.TensorSlice(positions, new[] { 0, 0, 0 }, new[] { 1, frames, d.AudioState });
        x = _engine.TensorAdd(x, positions);
        foreach (var block in EncoderBlocks) x = RunBlock(block, x, null, d.AudioHeads, false, null, null);
        return LayerNorm(PostNormWeight, PostNormBias, x);
    }

    /// <summary>The decoder's logits <c>[n, vocabulary]</c> for tokens after those <paramref name="cache"/> has read (all of
    /// them without a cache); the cross-attention scores of each layer go to <paramref name="crossScores"/>.</summary>
    public Tensor<T> Logits(IReadOnlyList<int> tokens, Tensor<T> audio, Cache? cache = null, List<Tensor<T>>? crossScores = null)
    {
        var d = Dimensions;
        int offset = cache?.Length ?? 0, n = tokens.Count;
        if (TokenEmbedding is null || TextPositions is null || NormWeight is null || NormBias is null)
            throw new InvalidOperationException("This Whisper network is encoder-only.");
        var table = _engine.Reshape(Value(TokenEmbedding), new[] { d.Vocabulary, d.TextState });
        var index = new Tensor<int>(new[] { n }, new Vector<int>(tokens.ToArray()));
        var x = _engine.TensorIndexSelect(table, index, 0);
        var positions = _engine.TensorSlice(_engine.Reshape(Value(TextPositions), new[] { d.TextContext, d.TextState }), new[] { offset, 0 }, new[] { n, d.TextState });
        x = _engine.Reshape(_engine.TensorAdd(x, positions), new[] { 1, n, d.TextState });
        foreach (var block in DecoderBlocks) x = RunBlock(block, x, audio, d.TextHeads, true, cache, crossScores);
        x = LayerNorm(NormWeight, NormBias, x);
        if (cache is not null) cache.Length = offset + n;
        if (OutputProjection is not null) return Linear(OutputProjection, OutputBias, _engine.Reshape(x, new[] { n, d.TextState }));
        return _engine.TensorMatMul(_engine.Reshape(x, new[] { n, d.TextState }), _engine.TensorTranspose(table));
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads an openai-whisper checkpoint's <c>model_state_dict</c> (by name; shapes checked by count).</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var d = Dimensions;
        void Linear(BiasFreeLinearLayer<T> layer, string name, int inputs, int outputs)
            => AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(layer, inputs, outputs, read(name, new[] { outputs, inputs }));
        void Vector(BiasFreeLinearLayer<T> holder, string name, int[] shape)
            => AiDotNet.TextToSpeech.CodecBased.TorchParameters.Linear(holder, 1, shape.Aggregate(1, (a, s) => a * s), read(name, shape));
        void LoadAttention(Attention a, string n, int width)
        {
            Linear(a.Query, n + ".query.weight", width, width);
            Vector(a.QueryBias, n + ".query.bias", new[] { width });
            Linear(a.Key, n + ".key.weight", width, width);
            Linear(a.Value, n + ".value.weight", width, width);
            Vector(a.ValueBias, n + ".value.bias", new[] { width });
            Linear(a.Out, n + ".out.weight", width, width);
            Vector(a.OutBias, n + ".out.bias", new[] { width });
        }
        void LoadBlock(Block b, string n, int width)
        {
            LoadAttention(b.SelfAttention, n + ".attn", width);
            Vector(b.AttentionNormWeight, n + ".attn_ln.weight", new[] { width });
            Vector(b.AttentionNormBias, n + ".attn_ln.bias", new[] { width });
            if (b.CrossAttention is not null)
            {
                LoadAttention(b.CrossAttention, n + ".cross_attn", width);
                Vector(b.CrossNormWeight!, n + ".cross_attn_ln.weight", new[] { width });
                Vector(b.CrossNormBias!, n + ".cross_attn_ln.bias", new[] { width });
            }
            Linear(b.Mlp1, n + ".mlp.0.weight", width, 4 * width);
            Vector(b.Mlp1Bias, n + ".mlp.0.bias", new[] { 4 * width });
            Linear(b.Mlp2, n + ".mlp.2.weight", 4 * width, width);
            Vector(b.Mlp2Bias, n + ".mlp.2.bias", new[] { width });
            Vector(b.MlpNormWeight, n + ".mlp_ln.weight", new[] { width });
            Vector(b.MlpNormBias, n + ".mlp_ln.bias", new[] { width });
        }
        Conv1.LoadTorchWeights(read("encoder.conv1.weight", new[] { d.AudioState, d.Mels, 3 }), null, read("encoder.conv1.bias", new[] { d.AudioState }));
        Conv2.LoadTorchWeights(read("encoder.conv2.weight", new[] { d.AudioState, d.AudioState, 3 }), null, read("encoder.conv2.bias", new[] { d.AudioState }));
        Vector(AudioPositions, "encoder.positional_embedding", new[] { d.AudioContext, d.AudioState });
        for (int i = 0; i < EncoderBlocks.Count; i++) LoadBlock(EncoderBlocks[i], $"encoder.blocks.{i}", d.AudioState);
        Vector(PostNormWeight, "encoder.ln_post.weight", new[] { d.AudioState });
        Vector(PostNormBias, "encoder.ln_post.bias", new[] { d.AudioState });
        if (TokenEmbedding is null || TextPositions is null || NormWeight is null || NormBias is null) return;
        Vector(TokenEmbedding, "decoder.token_embedding.weight", new[] { d.Vocabulary, d.TextState });
        Vector(TextPositions, "decoder.positional_embedding", new[] { d.TextContext, d.TextState });
        for (int i = 0; i < DecoderBlocks.Count; i++) LoadBlock(DecoderBlocks[i], $"decoder.blocks.{i}", d.TextState);
        Vector(NormWeight, "decoder.ln.weight", new[] { d.TextState });
        Vector(NormBias, "decoder.ln.bias", new[] { d.TextState });
        if (OutputProjection is not null)
        {
            Linear(OutputProjection, "decoder.out_proj.weight", d.TextState, d.Vocabulary);
            Vector(OutputBias!, "decoder.out_proj.bias", new[] { d.Vocabulary });
        }
    }

    /// <summary>
    /// openai-whisper's initialization: PyTorch's defaults (uniform ±1/√fan-in for linear and convolution weights and
    /// biases, N(0, 1) token embeddings, LayerNorm at 1 and 0), the encoder's fixed sinusoidal positions, and an empty
    /// (zero) learned text-position table, which the release leaves uninitialized; the alignment heads every head of the
    /// decoder's second half.
    /// </summary>
    public void InitializeLikeReference(Random random)
    {
        var d = Dimensions;
        double Uniform(double bound) => bound * (2 * random.NextDouble() - 1);
        LoadTorchWeights((name, shape) =>
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            var values = new double[count];
            if (name == "encoder.positional_embedding") return Sinusoids(d.AudioContext, d.AudioState);
            if (name.EndsWith("ln.weight", StringComparison.Ordinal) || name.EndsWith("ln_post.weight", StringComparison.Ordinal))
                return Enumerable.Repeat(1.0, count).ToArray();
            if (name.EndsWith("ln.bias", StringComparison.Ordinal) || name.EndsWith("ln_post.bias", StringComparison.Ordinal) || name == "decoder.positional_embedding")
                return values;
            if (name == "decoder.token_embedding.weight") return TorchParametersGaussian(random, count);
            int fanIn = shape.Length >= 2 ? shape.Skip(1).Aggregate(1, (a, s) => a * s) : 0;
            if (shape.Length == 1)
            {
                // A bias: its layer's fan-in.
                fanIn = name.StartsWith("encoder.conv1", StringComparison.Ordinal) ? d.Mels * 3
                    : name.StartsWith("encoder.conv2", StringComparison.Ordinal) ? d.AudioState * 3
                    : name.Contains(".mlp.2.") ? 4 * (name.StartsWith("encoder", StringComparison.Ordinal) ? d.AudioState : d.TextState)
                    : name.StartsWith("encoder", StringComparison.Ordinal) ? d.AudioState : d.TextState;
            }
            double bound = 1.0 / Math.Sqrt(fanIn);
            for (int i = 0; i < count; i++) values[i] = Uniform(bound);
            return values;
        });
        AlignmentHeads = Enumerable.Range(d.TextLayers / 2, d.TextLayers - d.TextLayers / 2)
            .SelectMany(layer => Enumerable.Range(0, d.TextHeads).Select(head => (layer, head))).ToList();
    }

    private static double[] TorchParametersGaussian(Random random, int count)
    {
        var v = AiDotNet.TextToSpeech.CodecBased.TorchParameters.Gaussian<double>(random, count, 1.0);
        return v.ToArray();
    }

    /// <summary><c>sinusoids(length, channels)</c>: sines then cosines of position × exp(−log(10000) / (channels/2 − 1) · i).</summary>
    public static double[] Sinusoids(int length, int channels)
    {
        int half = channels / 2;
        double increment = Math.Log(10000) / (half - 1);
        var values = new double[length * channels];
        for (int t = 0; t < length; t++)
            for (int i = 0; i < half; i++)
            {
                double angle = t * Math.Exp(-increment * i);
                values[t * channels + i] = Math.Sin(angle);
                values[t * channels + half + i] = Math.Cos(angle);
            }
        return values;
    }

    /// <summary>The alignment heads stored as openai-whisper's base85-encoded, gzipped boolean mask
    /// (<c>_ALIGNMENT_HEADS[name]</c>).</summary>
    public void SetAlignmentHeads(string dump)
    {
        var d = Dimensions;
        var mask = Base85Gzip(dump);
        if (mask.Length != d.TextLayers * d.TextHeads) throw new ArgumentException("The alignment heads do not fit this model.", nameof(dump));
        AlignmentHeads = Enumerable.Range(0, mask.Length).Where(i => mask[i] != 0).Select(i => (i / d.TextHeads, i % d.TextHeads)).ToList();
    }

    // base64.b85decode then gzip.decompress.
    private static byte[] Base85Gzip(string text)
    {
        const string alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
        int padding = (5 - text.Length % 5) % 5;
        text += new string('~', padding);
        var bytes = new List<byte>();
        for (int i = 0; i < text.Length; i += 5)
        {
            uint value = 0;
            for (int j = 0; j < 5; j++) value = unchecked(value * 85 + (uint)alphabet.IndexOf(text[i + j]));
            bytes.Add((byte)(value >> 24));
            bytes.Add((byte)(value >> 16));
            bytes.Add((byte)(value >> 8));
            bytes.Add((byte)value);
        }
        bytes.RemoveRange(bytes.Count - padding, padding);
        using var input = new MemoryStream(bytes.ToArray());
        using var gzip = new System.IO.Compression.GZipStream(input, System.IO.Compression.CompressionMode.Decompress);
        using var output = new MemoryStream();
        gzip.CopyTo(output);
        return output.ToArray();
    }

    /// <summary>openai-whisper's alignment heads of its released models (<c>whisper._ALIGNMENT_HEADS</c>, MIT).</summary>
    public static readonly IReadOnlyDictionary<string, string> ReleasedAlignmentHeads = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["tiny.en"] = "ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
        ["tiny"] = "ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
        ["base.en"] = "ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
        ["base"] = "ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
        ["small.en"] = "ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
        ["small"] = "ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
        ["medium.en"] = "ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
        ["medium"] = "ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
        ["large-v1"] = "ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
        ["large-v2"] = "ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        ["large-v3"] = "ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
        ["large-v3-turbo"] = "ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
        ["turbo"] = "ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
    };
}
