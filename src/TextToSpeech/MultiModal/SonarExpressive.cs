using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>The sizes of SONAR Expressive's SpeechProp encoder and EnCodec-unit decoders (Duquenne et al. 2023 §3.1).</summary>
/// <remarks>The paper states the depths (12 encoder layers, 16 per decoder), the conditioning (SONAR's 1024-dimensional
/// embedding concatenated with the SpeechProp embedding) and the units (8 codebooks of 1024 at 25 Hz); widths, heads and
/// the SpeechProp embedding's size are not stated and default to SONAR's 1024 and 16.</remarks>
internal sealed record SonarExpressiveConfiguration
{
    public int Codebooks { get; init; } = 8;
    public int CodebookSize { get; init; } = 1024;
    public int Width { get; init; } = 1024;
    public int Heads { get; init; } = 16;
    public int FeedForward { get; init; } = 4096;
    public int EncoderLayers { get; init; } = 12;
    public int DecoderLayers { get; init; } = 16;

    /// <summary>The SONAR embedding's size (1024).</summary>
    public int SemanticDim { get; init; } = 1024;

    /// <summary>The SpeechProp embedding's size.</summary>
    public int SpeechPropDim { get; init; } = 1024;

    public double Dropout { get; init; } = 0.1;

    /// <summary>The common vocabulary: every codebook's units, then the AR decoder's start and end.</summary>
    public int Vocabulary => Codebooks * CodebookSize + 2;
    public int Bos => Codebooks * CodebookSize;
    public int Eos => Codebooks * CodebookSize + 1;
}

/// <summary>
/// SONAR Expressive (Duquenne et al. 2023, "SONAR EXPRESSIVE: Zero-shot Expressive Speech-to-Speech Translation"): a
/// SpeechProp encoder capturing what SONAR's semantic sentence embedding does not (prosody, expressive style), and two
/// decoders that predict EnCodec units from both embeddings — an autoregressive one for the first codebook and a
/// non-autoregressive one for each later codebook (VALL-E's split, Wang et al. 2023a). SpiritLM's paper (§3.2) reads its
/// style features from the SpeechProp encoder.
/// </summary>
/// <remarks>
/// <para><b>SpeechProp encoder</b>: a Transformer encoder over the sum of the codebooks' unit embeddings, mean-pooled.</para>
/// <para><b>Decoders</b>: Transformer decoders cross-attending to the two conditioning vectors (projected to the width), units
/// of all codebooks embedded from one common vocabulary. The NAR decoder reads the summed embeddings of codebooks 1 … n−1
/// and the stage n (adaptive layer norm, as VALL-E's NAR) and predicts codebook n; training samples n uniformly from 2 … 8.
/// Conditioning is replaced by zeros with probability 0.1 so the decoders also learn unconditionally, for classifier-free
/// guidance (scale 3) at inference with top-k 10.</para>
/// <para>The released system was never published; this is built from the paper's text. Pre-norm layers with GELU,
/// sinusoidal positions and the conditioning projections are filled where it is silent.</para>
/// </remarks>
internal sealed class SonarExpressive<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    internal sealed record Block(LayerNormalizationLayer<T> SelfNorm, DenseLayer<T>? SelfStage, DenseLayer<T> Q, DenseLayer<T> K, DenseLayer<T> V, DenseLayer<T> O,
        LayerNormalizationLayer<T>? CrossNorm, DenseLayer<T>? CrossStage, DenseLayer<T>? CrossQ, DenseLayer<T>? CrossK, DenseLayer<T>? CrossV, DenseLayer<T>? CrossO,
        LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T>? FeedForwardStage, DenseLayer<T> FeedForwardIn, DenseLayer<T> FeedForwardOut);

    internal sealed class Stack
    {
        public List<Block> Blocks { get; } = new();
        public LayerNormalizationLayer<T> FinalNorm { get; init; } = null!;
        public DenseLayer<T>? FinalStage { get; init; }
    }

    public SonarExpressive(IEngine engine, List<LayerBase<T>> encoderLayers, List<LayerBase<T>> decoderLayers, SonarExpressiveConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Width % c.Heads != 0) throw new ArgumentException($"The width {c.Width} must divide into {c.Heads} heads.");
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        TLayer Own<TLayer>(List<LayerBase<T>> list, TLayer layer) where TLayer : LayerBase<T> { list.Add(layer); return layer; }
        DenseLayer<T> Linear(List<LayerBase<T>> list, int inputs, int outputs)
        {
            var layer = Own(list, new DenseLayer<T>(outputs, (IActivationFunction<T>)new IdentityActivation<T>()));
            // torch nn.Linear: U(±1/√in) for the weight and the bias.
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(engine, layer, inputs, outputs, Enumerable.Range(0, inputs * outputs).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray());
            return layer;
        }
        LayerNormalizationLayer<T> Norm(List<LayerBase<T>> list) => Own(list, new LayerNormalizationLayer<T>(c.Width, 1e-5));
        DenseLayer<T>? Stage(List<LayerBase<T>> list, bool adaptive) => adaptive ? Linear(list, c.Width, 2 * c.Width) : null;
        Stack Build(List<LayerBase<T>> list, int depth, bool cross, bool adaptive)
        {
            var stack = new Stack { FinalNorm = Norm(list), FinalStage = Stage(list, adaptive) };
            for (int i = 0; i < depth; i++)
                stack.Blocks.Add(new Block(Norm(list), Stage(list, adaptive), Linear(list, c.Width, c.Width), Linear(list, c.Width, c.Width), Linear(list, c.Width, c.Width),
                    Linear(list, c.Width, c.Width), cross ? Norm(list) : null, cross ? Stage(list, adaptive) : null, cross ? Linear(list, c.Width, c.Width) : null,
                    cross ? Linear(list, c.Width, c.Width) : null, cross ? Linear(list, c.Width, c.Width) : null, cross ? Linear(list, c.Width, c.Width) : null,
                    Norm(list), Stage(list, adaptive), Linear(list, c.Width, c.FeedForward), Linear(list, c.FeedForward, c.Width)));
            return stack;
        }
        TiedEmbeddingLayer<T> Table(List<LayerBase<T>> list, int rows)
        {
            var table = Own(list, new TiedEmbeddingLayer<T>(rows, c.Width));
            table.Reinitialize(() => Normal() * 0.02);
            return table;
        }
        EncoderEmbedding = Table(encoderLayers, c.Codebooks * c.CodebookSize);
        Encoder = Build(encoderLayers, c.EncoderLayers, cross: false, adaptive: false);
        SpeechPropProjection = Linear(encoderLayers, c.Width, c.SpeechPropDim);

        DecoderEmbedding = Table(decoderLayers, c.Vocabulary);
        StageEmbedding = Table(decoderLayers, c.Codebooks);
        SemanticProjection = Linear(decoderLayers, c.SemanticDim, c.Width);
        PropProjection = Linear(decoderLayers, c.SpeechPropDim, c.Width);
        Autoregressive = Build(decoderLayers, c.DecoderLayers, cross: true, adaptive: false);
        NonAutoregressive = Build(decoderLayers, c.DecoderLayers, cross: true, adaptive: true);
        ArHead = Linear(decoderLayers, c.Width, c.Vocabulary);
        NarHead = Linear(decoderLayers, c.Width, c.Vocabulary);
        CrossAttentionLayers = Autoregressive.Blocks.Concat(NonAutoregressive.Blocks)
            .SelectMany(b => new LayerBase<T>?[] { b.CrossNorm, b.CrossStage, b.CrossQ, b.CrossK, b.CrossV, b.CrossO }).Where(l => l is not null)
            .Concat(new LayerBase<T>[] { SemanticProjection, PropProjection }).Cast<LayerBase<T>>().ToList();
    }

    public SonarExpressiveConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> EncoderEmbedding { get; }
    public Stack Encoder { get; }
    public DenseLayer<T> SpeechPropProjection { get; }
    public TiedEmbeddingLayer<T> DecoderEmbedding { get; }
    public TiedEmbeddingLayer<T> StageEmbedding { get; }
    public DenseLayer<T> SemanticProjection { get; }
    public DenseLayer<T> PropProjection { get; }
    public Stack Autoregressive { get; }
    public Stack NonAutoregressive { get; }
    public DenseLayer<T> ArHead { get; }
    public DenseLayer<T> NarHead { get; }

    /// <summary>The layers fine-tuning updates in the decoders (§4.1: only the cross-attention, with the conditioning
    /// projections).</summary>
    public List<LayerBase<T>> CrossAttentionLayers { get; }

    // ---------------------------------------------------------------- pieces

    private Tensor<T> Positions(int length)
    {
        int d = Configuration.Width, half = d / 2;
        var p = new Tensor<T>(new[] { length, d });
        for (int t = 0; t < length; t++)
            for (int i = 0; i < half; i++)
            {
                double angle = t / Math.Pow(10000, 2.0 * i / d);
                p[t, 2 * i] = NumOps.FromDouble(Math.Sin(angle));
                p[t, 2 * i + 1] = NumOps.FromDouble(Math.Cos(angle));
            }
        return p;
    }

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var t = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) t[i] = NumOps.FromDouble(ids[i]);
        return t;
    }

    private Tensor<T> Dropout(Tensor<T> x, Random? random) =>
        random is not null && Configuration.Dropout > 0 ? T5Seq2Seq<T>.Dropout(_engine, x, Configuration.Dropout, random) : x;

    // LayerNorm, or VALL-E's AdaptiveLayerNorm: weight, bias = Linear(stage).chunk(2); weight · norm(x) + bias.
    private Tensor<T> Normalize(LayerNormalizationLayer<T> norm, DenseLayer<T>? stageProjection, Tensor<T> x, Tensor<T>? stage)
    {
        var y = norm.Forward(x);
        if (stageProjection is null || stage is null) return y;
        int d = Configuration.Width, length = x.Shape[0];
        var ws = stageProjection.Forward(stage);                                                                  // [1, 2d]
        var weight = _engine.TensorBroadcastTo(_engine.TensorSlice(ws, new[] { 0, 0 }, new[] { 1, d }), new[] { length, d });
        var bias = _engine.TensorBroadcastTo(_engine.TensorSlice(ws, new[] { 0, d }, new[] { 1, d }), new[] { length, d });
        return _engine.TensorAdd(_engine.TensorMultiply(weight, y), bias);
    }

    private Tensor<T> Attention(DenseLayer<T> q, DenseLayer<T> k, DenseLayer<T> v, DenseLayer<T> o, Tensor<T> x, Tensor<T> memory, bool causal)
    {
        int length = x.Shape[0], keys = memory.Shape[0], heads = Configuration.Heads, e = Configuration.Width / heads;
        Tensor<T> Heads(DenseLayer<T> projection, Tensor<T> input, int n) =>
            _engine.TensorPermute(_engine.Reshape(projection.Forward(input), new[] { n, heads, e }), new[] { 1, 0, 2 });
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(Heads(q, x, length), _engine.TensorPermute(Heads(k, memory, keys), new[] { 0, 2, 1 })),
            NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        if (causal)
        {
            var mask = new Tensor<T>(new[] { 1, length, keys });
            for (int i = 0; i < length; i++)
                for (int j = i + 1; j < keys; j++) mask[0, i, j] = NumOps.FromDouble(-1e9);
            scores = _engine.TensorAdd(scores, _engine.TensorBroadcastTo(mask, new[] { heads, length, keys }));
        }
        var context = _engine.BatchMatMul(_engine.Softmax(scores, axis: 2), Heads(v, memory, keys));
        return o.Forward(_engine.Reshape(_engine.TensorPermute(context, new[] { 1, 0, 2 }), new[] { length, Configuration.Width }));
    }

    private Tensor<T> Run(Stack stack, Tensor<T> x, Tensor<T>? memory, bool causal, Tensor<T>? stage, Random? random)
    {
        foreach (var b in stack.Blocks)
        {
            var h = Normalize(b.SelfNorm, b.SelfStage, x, stage);
            x = _engine.TensorAdd(x, Dropout(Attention(b.Q, b.K, b.V, b.O, h, h, causal), random));
            if (b.CrossNorm is not null && memory is not null)
            {
                h = Normalize(b.CrossNorm, b.CrossStage, x, stage);
                x = _engine.TensorAdd(x, Dropout(Attention(b.CrossQ!, b.CrossK!, b.CrossV!, b.CrossO!, h, memory, false), random));
            }
            h = Normalize(b.FeedForwardNorm, b.FeedForwardStage, x, stage);
            x = _engine.TensorAdd(x, Dropout(b.FeedForwardOut.Forward(ActivationHelper.ExactGelu(b.FeedForwardIn.Forward(h), _engine)), random));
        }
        return Normalize(stack.FinalNorm, stack.FinalStage, x, stage);
    }

    // The summed embeddings [frames, width] of codes [codebooks, frames] (rows 0 … count−1), each codebook its own rows of
    // the common table.
    private Tensor<T> SumEmbeddings(TiedEmbeddingLayer<T> table, int[,] codes, int count)
    {
        int frames = codes.GetLength(1), size = Configuration.CodebookSize;
        Tensor<T>? sum = null;
        for (int q = 0; q < count; q++)
        {
            var ids = new int[frames];
            for (int f = 0; f < frames; f++)
            {
                if (codes[q, f] < 0 || codes[q, f] >= size) throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {size}).");
                ids[f] = q * size + codes[q, f];
            }
            var e = table.Forward(Ids(ids));
            sum = sum is null ? e : _engine.TensorAdd(sum, e);
        }
        return sum!;
    }

    // ---------------------------------------------------------------- SpeechProp

    /// <summary>The SpeechProp encoder's states <c>[frames, width]</c> of codes <c>[codebooks, frames]</c>.</summary>
    public Tensor<T> SpeechPropStates(int[,] codes, Random? random = null)
    {
        int frames = codes.GetLength(1);
        var x = _engine.TensorAdd(SumEmbeddings(EncoderEmbedding, codes, Configuration.Codebooks), Positions(frames));
        return Run(Encoder, Dropout(x, random), null, false, null, random);
    }

    /// <summary>The SpeechProp embedding <c>[1, speechprop]</c>: the encoder's states mean-pooled, projected.</summary>
    public Tensor<T> SpeechProp(int[,] codes, Random? random = null)
    {
        var states = SpeechPropStates(codes, random);
        var pooled = _engine.TensorMultiplyScalar(_engine.ReduceSum(states, new[] { 0 }, keepDims: true), NumOps.FromDouble(1.0 / codes.GetLength(1)));
        return SpeechPropProjection.Forward(pooled);
    }

    // ---------------------------------------------------------------- decoders

    // The two conditioning vectors as the decoders' memory [2, width]; zeros for an unconditional pass.
    private Tensor<T> Memory(Tensor<T>? semantic, Tensor<T>? speechProp)
    {
        var c = Configuration;
        var s = semantic is null ? new Tensor<T>(new[] { 1, c.SemanticDim }) : _engine.Reshape(semantic, new[] { 1, c.SemanticDim });
        var p = speechProp is null ? new Tensor<T>(new[] { 1, c.SpeechPropDim }) : _engine.Reshape(speechProp, new[] { 1, c.SpeechPropDim });
        return _engine.TensorConcatenate(new[] { SemanticProjection.Forward(s), PropProjection.Forward(p) }, 0);
    }

    /// <summary>The AR decoder's logits <c>[length, vocabulary]</c> after <c>[BOS] + first-codebook codes</c>.</summary>
    public Tensor<T> ArLogits(IReadOnlyList<int> firstCodebook, Tensor<T>? semantic, Tensor<T>? speechProp, Random? random = null)
    {
        var ids = new[] { Configuration.Bos }.Concat(firstCodebook).ToArray();
        var x = _engine.TensorAdd(DecoderEmbedding.Forward(Ids(ids)), Positions(ids.Length));
        return ArHead.Forward(Run(Autoregressive, Dropout(x, random), Memory(semantic, speechProp), true, null, random));
    }

    /// <summary>The NAR decoder's logits <c>[frames, vocabulary]</c> for codebook <paramref name="stage"/> (1-based, 2 … 8)
    /// from codebooks 1 … stage−1.</summary>
    public Tensor<T> NarLogits(int[,] codes, int stage, Tensor<T>? semantic, Tensor<T>? speechProp, Random? random = null)
    {
        if (stage < 2 || stage > Configuration.Codebooks) throw new ArgumentOutOfRangeException(nameof(stage));
        int frames = codes.GetLength(1);
        var x = _engine.TensorAdd(SumEmbeddings(DecoderEmbedding, codes, stage - 1), Positions(frames));
        var stageVector = StageEmbedding.Forward(Ids(new[] { stage - 1 }));
        return NarHead.Forward(Run(NonAutoregressive, Dropout(x, random), Memory(semantic, speechProp), false, stageVector, random));
    }

    private Tensor<T> CrossEntropy(Tensor<T> logits, IReadOnlyList<int> targets)
    {
        int count = targets.Count, vocabulary = Configuration.Vocabulary;
        var log = _engine.Reshape(_engine.TensorLogSoftmax(logits, axis: 1), new[] { count * vocabulary, 1 });
        var picks = new Tensor<int>(new[] { count }, new Vector<int>(Enumerable.Range(0, count).Select(j => j * vocabulary + targets[j]).ToArray()));
        return _engine.TensorMultiplyScalar(_engine.TensorNegate(_engine.ReduceSum(_engine.TensorIndexSelect(log, picks, 0), null, false)),
            NumOps.FromDouble(1.0 / count));
    }

    /// <summary>
    /// The training loss on codes <c>[codebooks, frames]</c> (§4.1): the AR cross-entropy of codebook 1 and its end plus the
    /// NAR cross-entropy of a codebook drawn from 2 … 8, conditioned on the semantic embedding and (when given) the
    /// SpeechProp embedding of a random crop of at least 10 frames (§4.1's random cropping); both conditions are zeroed with
    /// probability 0.1.
    /// </summary>
    public Tensor<T> Loss(int[,] codes, Tensor<T>? semantic, bool withSpeechProp, Random random)
    {
        var c = Configuration;
        int frames = codes.GetLength(1);
        Tensor<T>? prop = null;
        if (withSpeechProp)
        {
            int length = Math.Min(frames, Math.Max(10, random.Next(Math.Min(10, frames), frames + 1)));
            int start = random.Next(frames - length + 1);
            var crop = new int[c.Codebooks, length];
            for (int q = 0; q < c.Codebooks; q++)
                for (int f = 0; f < length; f++) crop[q, f] = codes[q, start + f];
            prop = SpeechProp(crop, random);
        }
        if (random.NextDouble() < 0.1) { semantic = null; prop = null; }
        var first = Enumerable.Range(0, frames).Select(f => codes[0, f]).ToArray();
        var arTargets = first.Concat(new[] { c.Eos }).ToArray();
        var loss = CrossEntropy(ArLogits(first, semantic, prop, random), arTargets);
        int stage = random.Next(2, c.Codebooks + 1);
        var narTargets = new int[frames];
        for (int f = 0; f < frames; f++) narTargets[f] = (stage - 1) * c.CodebookSize + codes[stage - 1, f];
        return _engine.TensorAdd(loss, CrossEntropy(NarLogits(codes, stage, semantic, prop, random), narTargets));
    }

    // Classifier-free guidance on logits: uncond + scale · (cond − uncond).
    private static double[] Guided(Tensor<T> conditional, Tensor<T> unconditional, int row, int from, int to, double scale)
    {
        var values = new double[to - from];
        for (int i = from; i < to; i++)
        {
            double u = NumOps.ToDouble(unconditional[row, i]);
            values[i - from] = u + scale * (NumOps.ToDouble(conditional[row, i]) - u);
        }
        return values;
    }

    private static int TopK(double[] logits, int k, Random random)
    {
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).Take(Math.Max(1, k)).ToArray();
        double max = logits[order[0]];
        var weights = order.Select(i => Math.Exp(logits[i] - max)).ToArray();
        double draw = random.NextDouble() * weights.Sum();
        for (int j = 0; j < order.Length; j++)
        {
            draw -= weights[j];
            if (draw <= 0) return order[j];
        }
        return order[^1];
    }

    /// <summary>
    /// Codes <c>[codebooks, frames]</c> from the conditioning (§5.3): the first codebook sampled autoregressively, each later
    /// one predicted by the NAR decoder, top-k sampling (k = 10) on classifier-free-guided logits (scale 3).
    /// </summary>
    public int[,] Generate(Tensor<T>? semantic, Tensor<T>? speechProp, int maxFrames, Random random, int topK = 10, double guidance = 3.0)
    {
        var c = Configuration;
        var first = new List<int>();
        while (first.Count < maxFrames)
        {
            var cond = ArLogits(first, semantic, speechProp);
            var uncond = ArLogits(first, null, null);
            int row = first.Count;
            // The first codebook's units and the end.
            var logits = Guided(cond, uncond, row, 0, c.CodebookSize, guidance).Concat(Guided(cond, uncond, row, c.Eos, c.Eos + 1, guidance)).ToArray();
            int next = TopK(logits, topK, random);
            if (next == c.CodebookSize) break;
            first.Add(next);
        }
        int frames = first.Count;
        var codes = new int[c.Codebooks, frames];
        for (int f = 0; f < frames; f++) codes[0, f] = first[f];
        for (int stage = 2; stage <= c.Codebooks && frames > 0; stage++)
        {
            var cond = NarLogits(codes, stage, semantic, speechProp);
            var uncond = NarLogits(codes, stage, null, null);
            int from = (stage - 1) * c.CodebookSize;
            for (int f = 0; f < frames; f++) codes[stage - 1, f] = TopK(Guided(cond, uncond, f, from, from + c.CodebookSize, guidance), topK, random);
        }
        return codes;
    }
}
