using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of VALL-E 2's codec language models (Chen et al. 2024, §3.2, §4.1.1).</summary>
/// <param name="TextTokens">Rows of the text embedding (the vocabulary, frame tokens included).</param>
/// <param name="AudioTokens">Codes per codebook (1024; the AR model adds the end token 1024).</param>
/// <param name="Codebooks">Codebooks (8).</param>
/// <param name="GroupSize">Codes per AR step (§3.1).</param>
/// <param name="ModelDim">Embedding width.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="Layers">Layers of each model.</param>
/// <param name="FeedForwardDim">Feed-forward width.</param>
/// <param name="Dropout">Dropout.</param>
/// <param name="MaxTextPositions">Rows of the learned text position embedding.</param>
/// <param name="MaxCodePositions">Rows of the learned code (and code-group) position embedding.</param>
internal sealed record VallE2Configuration(
    int TextTokens, int AudioTokens, int Codebooks, int GroupSize, int ModelDim, int Heads, int Layers, int FeedForwardDim,
    double Dropout, int MaxTextPositions, int MaxCodePositions);

/// <summary>
/// VALL-E 2's grouped AR and code-ID NAR codec language models (Chen et al. 2024, §3), built from VALL-E's pre-norm
/// Transformer layers (§4.1.1: "both the AR model and the NAR models employ the same Transformer architecture in
/// VALL-E").
/// </summary>
/// <remarks>
/// <para>
/// <b>AR</b> (§3.3.1): the text embeddings, then <c>&lt;eos&gt; &lt;bos&gt;</c>, then one group embedding per G
/// first-codebook codes (the G code embeddings concatenated and projected by <c>W_g</c>, Eq. 7–8), each sequence with
/// its own learned positions, run through a causal Transformer (Figure 2: every token attends to its left). A group
/// prediction layer maps each state to G states, and the code prediction layer — the code embedding, shared — gives each
/// of the next group's G codes; after the last group comes a group of end tokens.
/// </para>
/// <para>
/// <b>NAR</b> (§3.3.2): the text embeddings, <c>&lt;eos&gt;</c>, the summed code embeddings (all eight codebooks for the
/// acoustic condition, codebooks <c>0 … j − 1</c> after it, Eq. 12), <c>&lt;eos&gt;</c> and the code-ID embedding
/// <c>e_j</c> (Eq. 14), with learned positions, under full attention; codebook j's embedding, shared, predicts its codes.
/// </para>
/// <para>
/// The paper does not say where the separating <c>&lt;eos&gt;</c>, <c>&lt;bos&gt;</c> embeddings come from; they are the
/// text embedding's rows for those tokens, with positions continuing the sequence they close or open.
/// </para>
/// </remarks>
internal sealed class VallE2Core<T>
{
    private readonly IEngine _engine;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int EndText = 2, BeginCodes = 1;

    public VallE2Core(IEngine engine, List<LayerBase<T>> arLayers, List<LayerBase<T>> narLayers, VallE2Configuration configuration)
    {
        _engine = engine;
        var c = Configuration = configuration;
        int d = c.ModelDim;
        var stack = new VallETransformerConfiguration(d, c.Heads, c.Layers, c.FeedForwardDim, c.Dropout, AdaptiveNorm: false);

        ArTextEmbedding = Own(arLayers, new TiedEmbeddingLayer<T>(c.TextTokens, d));
        ArCodeEmbedding = Own(arLayers, new TiedEmbeddingLayer<T>(c.AudioTokens + 1, d));
        ArTextPositions = Own(arLayers, new TiedEmbeddingLayer<T>(c.MaxTextPositions, d));
        ArCodePositions = Own(arLayers, new TiedEmbeddingLayer<T>(c.MaxCodePositions, d));
        if (c.GroupSize > 1)
        {
            GroupEmbedding = Own(arLayers, new BiasFreeLinearLayer<T>(c.GroupSize * d, d));
            GroupPrediction = Own(arLayers, new BiasFreeLinearLayer<T>(d, c.GroupSize * d));
        }
        ArDecoder = new VallETransformer<T>(engine, arLayers, stack);

        NarTextEmbedding = Own(narLayers, new TiedEmbeddingLayer<T>(c.TextTokens, d));
        for (int j = 0; j < c.Codebooks; j++) NarCodeEmbeddings.Add(Own(narLayers, new TiedEmbeddingLayer<T>(c.AudioTokens, d)));
        NarCodeIds = Own(narLayers, new TiedEmbeddingLayer<T>(c.Codebooks - 1, d));
        NarTextPositions = Own(narLayers, new TiedEmbeddingLayer<T>(c.MaxTextPositions, d));
        NarCodePositions = Own(narLayers, new TiedEmbeddingLayer<T>(c.MaxCodePositions + 2, d));
        NarDecoder = new VallETransformer<T>(engine, narLayers, stack);
    }

    public VallE2Configuration Configuration { get; }
    public int EndToken => Configuration.AudioTokens;

    public TiedEmbeddingLayer<T> ArTextEmbedding { get; }
    public TiedEmbeddingLayer<T> ArCodeEmbedding { get; }
    public TiedEmbeddingLayer<T> ArTextPositions { get; }
    public TiedEmbeddingLayer<T> ArCodePositions { get; }
    public BiasFreeLinearLayer<T>? GroupEmbedding { get; }
    public BiasFreeLinearLayer<T>? GroupPrediction { get; }
    public VallETransformer<T> ArDecoder { get; }
    public TiedEmbeddingLayer<T> NarTextEmbedding { get; }
    public List<TiedEmbeddingLayer<T>> NarCodeEmbeddings { get; } = new();
    public TiedEmbeddingLayer<T> NarCodeIds { get; }
    public TiedEmbeddingLayer<T> NarTextPositions { get; }
    public TiedEmbeddingLayer<T> NarCodePositions { get; }
    public VallETransformer<T> NarDecoder { get; }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private static Tensor<T> Range(int start, int count) => Ids(Enumerable.Range(start, count).ToArray());

    /// <summary>PyTorch-style initialization: embeddings N(0, 1), linear maps uniform in ±1/√fan-in, the Transformers as
    /// VALL-E's (the paper states none).</summary>
    public void Initialize(Random random)
    {
        double Normal()
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        void Linear(BiasFreeLinearLayer<T> layer)
        {
            double bound = 1 / Math.Sqrt(layer.InputSize);
            var values = new Vector<T>(layer.InputSize * layer.OutputSize);
            for (int i = 0; i < values.Length; i++) values[i] = NumOps.FromDouble((2 * random.NextDouble() - 1) * bound);
            layer.SetParameters(values);
        }
        foreach (var table in new[] { ArTextEmbedding, ArCodeEmbedding, ArTextPositions, ArCodePositions, NarTextEmbedding,
                     NarCodeIds, NarTextPositions, NarCodePositions })
            table.Reinitialize(Normal);
        foreach (var table in NarCodeEmbeddings) table.Reinitialize(Normal);
        if (GroupEmbedding is not null) Linear(GroupEmbedding);
        if (GroupPrediction is not null) Linear(GroupPrediction);
        ArDecoder.InitializeLikePyTorch(random);
        NarDecoder.InitializeLikePyTorch(random);
    }

    // ---------------------------------------------------------------- AR

    private void CheckPositions(int text, int codes)
    {
        if (text > Configuration.MaxTextPositions)
            throw new ArgumentException($"{text} text positions exceed the {Configuration.MaxTextPositions} the model has.");
        if (codes > Configuration.MaxCodePositions)
            throw new ArgumentException($"{codes} code positions exceed the {Configuration.MaxCodePositions} the model has.");
    }

    /// <summary>
    /// The AR model's logits <c>[groups + 1, G, codes + 1]</c>: position g (the <c>&lt;bos&gt;</c>, then each group of
    /// <paramref name="codes"/>, whose length is a multiple of G) predicts the next group's G codes.
    /// </summary>
    public Tensor<T> ArLogits(IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training, Random random)
    {
        var c = Configuration;
        int g = c.GroupSize, d = c.ModelDim;
        if (codes.Count % g != 0) throw new ArgumentException($"The codes ({codes.Count}) are not whole groups of {g}.");
        int groups = codes.Count / g, textLength = text.Count + 1, codeLength = groups + 1;
        CheckPositions(textLength, codeLength);
        // Ex ‖ e<eos>, then e<bos> ‖ Eg, each with its own positions.
        var textIds = text.Concat(new[] { EndText }).ToArray();
        var x = _engine.TensorAdd(ArTextEmbedding.Forward(Ids(textIds)), ArTextPositions.Forward(Range(0, textLength)));
        var bos = ArTextEmbedding.Forward(Ids(new[] { BeginCodes }));                                         // [1, d]
        Tensor<T> groupEmbeddings;
        if (groups == 0)
        {
            groupEmbeddings = bos;
        }
        else
        {
            var codeEmbeddings = ArCodeEmbedding.Forward(Ids(codes));                                         // [T, d]
            var grouped = _engine.Reshape(codeEmbeddings, new[] { groups, g * d });
            var projected = GroupEmbedding is null ? grouped : GroupEmbedding.Forward(grouped);               // [T/G, d]
            groupEmbeddings = _engine.TensorConcatenate(new[] { bos, projected }, 0);
        }
        var y = _engine.TensorAdd(groupEmbeddings, ArCodePositions.Forward(Range(0, codeLength)));
        var xy = _engine.TensorConcatenate(new[] { x, y }, 0);
        int total = textLength + codeLength;
        var mask = new Tensor<T>(new[] { total, total });
        var blocked = NumOps.FromDouble(-1e9);
        for (int i = 0; i < total; i++)
            for (int j = i + 1; j < total; j++) mask[i, j] = blocked;
        var hidden = ArDecoder.Forward(xy, null, mask, training, random);
        var states = _engine.TensorSlice(hidden, new[] { textLength, 0 }, new[] { codeLength, d });           // [groups+1, d]
        var slots = GroupPrediction is null ? states : _engine.Reshape(GroupPrediction.Forward(states), new[] { codeLength * g, d });
        var logits = ArCodeEmbedding.Logits(slots);                                                           // [(groups+1)·G, V]
        return _engine.Reshape(logits, new[] { codeLength, g, c.AudioTokens + 1 });
    }

    /// <summary>The AR loss (Eq. 9–11): the summed cross-entropy of every group's codes and, after the last, a group of
    /// end tokens.</summary>
    public Tensor<T> ArLoss(IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training, Random random)
    {
        var logits = ArLogits(text, codes, training, random);
        int g = Configuration.GroupSize, steps = logits.Shape[0];
        var flat = _engine.Reshape(logits, new[] { steps * g, Configuration.AudioTokens + 1 });
        var targets = new int[steps * g];
        for (int i = 0; i < targets.Length; i++) targets[i] = i < codes.Count ? codes[i] : EndToken;
        return CrossEntropySum(flat, targets);
    }

    /// <summary>
    /// Continues the first codebook after <paramref name="prompt"/> (whole groups) group by group (Eq. 18–20) with
    /// repetition-aware sampling (Algorithm 1): each code by nucleus sampling at <paramref name="topP"/>, replaced by
    /// random sampling when its repetition ratio over the last <paramref name="window"/> codes exceeds
    /// <paramref name="threshold"/>. Ends at the first end token, or after <paramref name="maxNewCodes"/>.
    /// </summary>
    public List<int> ArGenerate(IReadOnlyList<int> text, IReadOnlyList<int> prompt, double topP, int window, double threshold,
        int maxNewCodes, Random random)
    {
        int g = Configuration.GroupSize;
        var codes = new List<int>(prompt);
        var generated = new List<int>();
        while (generated.Count < maxNewCodes)
        {
            var logits = ArLogits(text, codes, training: false, random);
            int last = logits.Shape[0] - 1, vocabulary = logits.Shape[2];
            for (int slot = 0; slot < g; slot++)
            {
                var row = new double[vocabulary];
                for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logits[last, slot, v]);
                var probabilities = Softmax(row);
                int code = Nucleus(probabilities, topP, random);
                // r = 1/K Σ_{k=0}^{K} 1[c_t' = c_{t'−k}] over the codes so far (the k = 0 term is the code itself).
                int matches = 1;
                for (int k = 1; k <= window && codes.Count - k >= 0; k++)
                    if (codes[codes.Count - k] == code) matches++;
                if ((double)matches / window > threshold) code = Categorical(probabilities, random);
                if (code == EndToken)
                    return Finish(generated);
                codes.Add(code);
                generated.Add(code);
                if (generated.Count >= maxNewCodes) break;
            }
        }
        return Finish(generated);
    }

    private static List<int> Finish(List<int> generated)
    {
        if (generated.Count == 0)
            throw new InvalidOperationException("VALL-E 2's AR model ended the utterance before its first code; a trained model does not.");
        return generated;
    }

    private static double[] Softmax(double[] logits)
    {
        double max = logits.Max(), total = 0;
        var p = new double[logits.Length];
        for (int i = 0; i < p.Length; i++) total += p[i] = Math.Exp(logits[i] - max);
        for (int i = 0; i < p.Length; i++) p[i] /= total;
        return p;
    }

    // Nucleus sampling: the smallest most-likely set whose mass reaches p (at least one code), renormalized.
    internal static int Nucleus(double[] probabilities, double topP, Random random)
    {
        var order = Enumerable.Range(0, probabilities.Length).OrderByDescending(i => probabilities[i]).ToArray();
        double mass = 0;
        int kept = 0;
        do
        {
            mass += probabilities[order[kept]];
            kept++;
        }
        while (kept < order.Length && mass < topP);
        double draw = random.NextDouble() * mass;
        for (int i = 0; i < kept; i++)
        {
            draw -= probabilities[order[i]];
            if (draw <= 0) return order[i];
        }
        return order[kept - 1];
    }

    internal static int Categorical(double[] probabilities, Random random)
    {
        double draw = random.NextDouble();
        for (int i = 0; i < probabilities.Length; i++)
        {
            draw -= probabilities[i];
            if (draw <= 0) return i;
        }
        return probabilities.Length - 1;
    }

    // ---------------------------------------------------------------- NAR

    /// <summary>
    /// The NAR logits <c>[frames, codes]</c> of codebook <paramref name="stage"/> (1 … 7) over the frames after the acoustic
    /// condition: text, condition <paramref name="prompt"/> <c>[T′, codebooks]</c> (all codebooks summed) and the target
    /// <paramref name="codes"/> <c>[frames, codebooks]</c> (codebooks 0 … stage − 1 summed).
    /// </summary>
    public Tensor<T> NarLogits(IReadOnlyList<int> text, int[,] prompt, int[,] codes, int stage, bool training, Random random)
    {
        var c = Configuration;
        int promptFrames = prompt.GetLength(0), frames = codes.GetLength(0), d = c.ModelDim;
        int textLength = text.Count + 1, codeLength = promptFrames + frames + 2;
        CheckPositions(textLength, codeLength - 2);
        Tensor<T> Sum(int[,] source, int levels)
        {
            int length = source.GetLength(0);
            Tensor<T>? sum = null;
            for (int j = 0; j < levels; j++)
            {
                var column = new int[length];
                for (int t = 0; t < length; t++) column[t] = source[t, j];
                var embedded = NarCodeEmbeddings[j].Forward(Ids(column));
                sum = sum is null ? embedded : _engine.TensorAdd(sum, embedded);
            }
            return sum!;
        }
        var textIds = text.Concat(new[] { EndText }).ToArray();
        var x = _engine.TensorAdd(NarTextEmbedding.Forward(Ids(textIds)), NarTextPositions.Forward(Range(0, textLength)));
        var parts = new List<Tensor<T>>();
        if (promptFrames > 0) parts.Add(Sum(prompt, c.Codebooks));
        parts.Add(Sum(codes, stage));
        parts.Add(NarTextEmbedding.Forward(Ids(new[] { EndText })));
        parts.Add(NarCodeIds.Forward(Ids(new[] { stage - 1 })));
        var y = _engine.TensorAdd(_engine.TensorConcatenate(parts.ToArray(), 0), NarCodePositions.Forward(Range(0, codeLength)));
        var hidden = NarDecoder.Forward(_engine.TensorConcatenate(new[] { x, y }, 0), null, null, training, random);
        var states = _engine.TensorSlice(hidden, new[] { textLength + promptFrames, 0 }, new[] { frames, d });
        return NarCodeEmbeddings[stage].Logits(states);
    }

    /// <summary>The NAR loss (Eq. 17): a uniformly drawn codebook j, the first <paramref name="promptFrames"/> frames as
    /// the acoustic condition, and the summed cross-entropy of codebook j after it.</summary>
    public Tensor<T> NarLoss(IReadOnlyList<int> text, int[,] codes, int promptFrames, bool training, Random random)
    {
        var c = Configuration;
        int frames = codes.GetLength(0), stage = 1 + random.Next(c.Codebooks - 1);
        var prompt = new int[promptFrames, c.Codebooks];
        var target = new int[frames - promptFrames, c.Codebooks];
        for (int t = 0; t < frames; t++)
            for (int j = 0; j < c.Codebooks; j++)
                if (t < promptFrames) prompt[t, j] = codes[t, j];
                else target[t - promptFrames, j] = codes[t, j];
        var logits = NarLogits(text, prompt, target, stage, training, random);
        var labels = new int[frames - promptFrames];
        for (int t = 0; t < labels.Length; t++) labels[t] = target[t, stage];
        return CrossEntropySum(logits, labels);
    }

    /// <summary>Fills codebooks 2 … 8 of the generated frames greedily, one at a time (Eq. 21).</summary>
    public int[,] NarGenerate(IReadOnlyList<int> text, int[,] prompt, int[] firstCodebook, Random random)
    {
        var c = Configuration;
        int frames = firstCodebook.Length;
        var codes = new int[frames, c.Codebooks];
        for (int t = 0; t < frames; t++) codes[t, 0] = firstCodebook[t];
        for (int stage = 1; stage < c.Codebooks; stage++)
        {
            var logits = NarLogits(text, prompt, codes, stage, training: false, random);
            for (int t = 0; t < frames; t++)
            {
                int best = 0;
                for (int v = 1; v < logits.Shape[1]; v++)
                    if (NumOps.ToDouble(logits[t, v]) > NumOps.ToDouble(logits[t, best])) best = v;
                codes[t, stage] = best;
            }
        }
        return codes;
    }

    private Tensor<T> CrossEntropySum(Tensor<T> logits, int[] targets)
    {
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var selection = new Tensor<T>(logits._shape);
        for (int t = 0; t < targets.Length; t++)
            if (targets[t] >= 0) selection[t, targets[t]] = NumOps.FromDouble(-1.0);
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, selection), new[] { 0, 1 }, keepDims: false);
    }
}
