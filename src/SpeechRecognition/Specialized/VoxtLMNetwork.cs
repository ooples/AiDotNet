using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.SpeechRecognition.Specialized;

/// <summary>
/// VoxtLM's networks (Maiti et al. 2024, §3.1; ESPnet's <c>egs2/voxtlm_v1</c> recipe): the speech tokenizer, k-means
/// clusters of HuBERT features, and ESPnet's <c>TransformerLM</c> over the joint speech-text vocabulary.
/// </summary>
/// <remarks>
/// <para>The language model is the recipe's <c>lm: transformer</c> with <c>pos_enc: null</c>: a token embedding, the
/// <c>linear</c> input layer (Linear, LayerNorm, dropout, ReLU), pre-norm Transformer layers (causal self-attention and a
/// ReLU feed-forward block, each a residual branch with dropout), a closing LayerNorm and a linear output layer. It has
/// no positional encoding; causal attention alone orders the tokens.</para>
/// <para>The vocabulary is laid out as ESPnet's token list with the paper's four task tokens: <c>&lt;blank&gt;</c>,
/// <c>&lt;unk&gt;</c>, <c>&lt;startspeech&gt;</c>, <c>&lt;generatetext&gt;</c>, <c>&lt;starttext&gt;</c>,
/// <c>&lt;generatespeech&gt;</c>, the speech units, the text tokens, then <c>&lt;sos/eos&gt;</c>.</para>
/// </remarks>
internal sealed class VoxtLMNetwork<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    public const int StartSpeech = 2, GenerateText = 3, StartText = 4, GenerateSpeech = 5, UnitOffset = 6;
    private readonly IEngine _engine;
    private readonly VoxtLMOptions _o;

    internal sealed record Block(LayerNormalizationLayer<T> AttentionNorm, DenseLayer<T> Query, DenseLayer<T> Key, DenseLayer<T> Value,
        DenseLayer<T> Output, LayerNormalizationLayer<T> FeedForwardNorm, DenseLayer<T> Intermediate, DenseLayer<T> FeedForwardOutput);

    public VoxtLMNetwork(IEngine engine, VoxtLMOptions options, List<LayerBase<T>> lmLayers, List<LayerBase<T>> hubertLayers, Random random)
    {
        _engine = engine;
        _o = options;
        if (options.ModelDim % options.NumAttentionHeads != 0)
            throw new ArgumentException($"The width {options.ModelDim} must divide into {options.NumAttentionHeads} heads.", nameof(options));
        if (options.NumSpeechUnits <= 0 || options.TextVocabSize <= 0) throw new ArgumentException("The speech and text vocabularies must be non-empty.", nameof(options));
        TLayer Own<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            lmLayers.Add(layer);
            return layer;
        }
        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        DenseLayer<T> Linear(int outputs) => Own(new DenseLayer<T>(outputs, identity));
        LayerNormalizationLayer<T> Norm(int width) => Own(new LayerNormalizationLayer<T>(width, 1e-12));
        int d = options.ModelDim;
        Embeddings = Own(new BiasFreeLinearLayer<T>(1, VocabSize * options.EmbeddingDim));
        InputLinear = Linear(d);
        InputNorm = Norm(d);
        for (int i = 0; i < options.NumLayers; i++)
            Blocks.Add(new Block(Norm(d), Linear(d), Linear(d), Linear(d), Linear(d), Norm(d), Linear(options.FeedForwardDim), Linear(d)));
        FinalNorm = Norm(d);
        Decoder = Linear(VocabSize);
        InitializeLikeReference(random);

        // GSLM's tokenizer as the recipe sets it (kmeans_feature: hubert_base/6): HuBERT Base read at a layer, then k-means.
        var hubert = HubertUnitExtractor<T>.MHubertBase(options.HubertLayer) with
        {
            HiddenSize = options.HubertWidth, Heads = options.HubertHeads, IntermediateSize = options.HubertIntermediate,
            ConvDims = Enumerable.Repeat(options.HubertConvChannels, 7).ToArray(),
            PositionalConvKernel = options.HubertPositionalKernel, PositionalConvGroups = options.HubertPositionalGroups,
        };
        Units = new HubertUnitExtractor<T>(engine, hubertLayers, hubert, options.HubertLayer, options.NumSpeechUnits);
        Units.Network.InitializeLikeReference(random);
        // Until a fitted k-means is loaded or fitted, centroids drawn in the space of HuBERT's layer-normalized features.
        var centroids = new double[options.NumSpeechUnits, options.HubertWidth];
        for (int k = 0; k < options.NumSpeechUnits; k++)
            for (int i = 0; i < options.HubertWidth; i++)
                centroids[k, i] = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        Units.Centroids = centroids;
    }

    public HubertUnitExtractor<T> Units { get; }
    public BiasFreeLinearLayer<T> Embeddings { get; }
    public DenseLayer<T> InputLinear { get; }
    public LayerNormalizationLayer<T> InputNorm { get; }
    public List<Block> Blocks { get; } = new();
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> Decoder { get; }

    /// <summary>The joint vocabulary: six special tokens, the units, the text tokens and <c>&lt;sos/eos&gt;</c>.</summary>
    public int VocabSize => UnitOffset + _o.NumSpeechUnits + _o.TextVocabSize + 1;
    public int TextOffset => UnitOffset + _o.NumSpeechUnits;
    public int EndToken => VocabSize - 1;

    // PyTorch's defaults, as the recipe leaves them: nn.Embedding N(0, 1), nn.Linear U(±1/√fan-in), LayerNorm (1, 0).
    private void InitializeLikeReference(Random random)
    {
        Embeddings.SetParameters(TorchParameters.Gaussian<T>(random, VocabSize * _o.EmbeddingDim, 1.0));
        void Linear(DenseLayer<T> layer, int inputs, int outputs)
        {
            double bound = 1.0 / Math.Sqrt(inputs);
            TorchParameters.Linear(_engine, layer, inputs, outputs,
                Enumerable.Range(0, inputs * outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
                Enumerable.Range(0, outputs).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        }
        int d = _o.ModelDim;
        Linear(InputLinear, _o.EmbeddingDim, d);
        foreach (var b in Blocks)
        {
            Linear(b.Query, d, d);
            Linear(b.Key, d, d);
            Linear(b.Value, d, d);
            Linear(b.Output, d, d);
            Linear(b.Intermediate, d, _o.FeedForwardDim);
            Linear(b.FeedForwardOutput, _o.FeedForwardDim, d);
        }
        Linear(Decoder, d, VocabSize);
    }

    /// <summary>The speech units of a 16 kHz recording, as vocabulary ids.</summary>
    public int[] SpeechTokens(IReadOnlyList<double> waveform) => Units.Units(waveform, deduplicate: false).Select(u => UnitOffset + u).ToArray();

    /// <summary>The ASR condition (Table 1): <c>&lt;startspeech&gt;</c>, the units, <c>&lt;generatetext&gt;</c>.</summary>
    public int[] AsrPrompt(IReadOnlyList<double> waveform) => new[] { StartSpeech }.Concat(SpeechTokens(waveform)).Concat(new[] { GenerateText }).ToArray();

    /// <summary>The logits <c>[L, vocabulary]</c> of ids <c>[L]</c>, each position predicting the next.</summary>
    public Tensor<T> Forward(IReadOnlyList<int> ids, bool training, Random random)
    {
        int length = ids.Count, d = _o.ModelDim;
        var index = new Tensor<int>(new[] { length });
        for (int i = 0; i < length; i++)
        {
            if (ids[i] < 0 || ids[i] >= VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Id {ids[i]} is outside [0, {VocabSize}).");
            index[i] = ids[i];
        }
        var table = _engine.Reshape(Embeddings.Weights, new[] { VocabSize, _o.EmbeddingDim });
        var x = _engine.TensorIndexSelect(table, index, 0);
        x = _engine.ReLU(Drop(InputNorm.Forward(InputLinear.Forward(x)), training, random));
        var mask = CausalMask(length);
        foreach (var b in Blocks) x = Layer(b, x, mask, training, random);
        return Decoder.Forward(FinalNorm.Forward(x));
    }

    // -1e9 above the diagonal, per head: each position attends to itself and the positions before it.
    private Tensor<T> CausalMask(int length)
    {
        var mask = new Tensor<T>(new[] { _o.NumAttentionHeads, length, length });
        var blocked = NumOps.FromDouble(-1e9);
        for (int h = 0; h < _o.NumAttentionHeads; h++)
            for (int i = 0; i < length; i++)
                for (int j = i + 1; j < length; j++) mask[h, i, j] = blocked;
        return mask;
    }

    private Tensor<T> Drop(Tensor<T> x, bool training, Random random)
    {
        double rate = _o.DropoutRate;
        if (!training || rate <= 0) return x;
        var mask = new Tensor<T>(x._shape);
        var keep = NumOps.FromDouble(1.0 / (1.0 - rate));
        for (int i = 0; i < mask.Length; i++) mask[i] = random.NextDouble() < rate ? NumOps.Zero : keep;
        return _engine.TensorMultiply(x, mask);
    }

    // ESPnet's EncoderLayer with normalize_before: x + dropout(attention(norm(x))), then x + dropout(ff(norm(x))).
    private Tensor<T> Layer(Block l, Tensor<T> x, Tensor<T> mask, bool training, Random random)
    {
        int length = x.Shape[0], heads = _o.NumAttentionHeads, e = _o.ModelDim / heads;
        var normed = l.AttentionNorm.Forward(x);
        Tensor<T> Split(DenseLayer<T> projection) =>
            _engine.TensorPermute(_engine.Reshape(projection.Forward(normed), new[] { length, heads, e }), new[] { 1, 0, 2 });
        var q = Split(l.Query);
        var k = Split(l.Key);
        var v = Split(l.Value);
        var scores = _engine.TensorMultiplyScalar(_engine.BatchMatMul(q, _engine.TensorPermute(k, new[] { 0, 2, 1 })), NumOps.FromDouble(1.0 / Math.Sqrt(e)));
        var weights = Drop(_engine.Softmax(_engine.TensorAdd(scores, mask), axis: 2), training, random);
        var context = _engine.Reshape(_engine.TensorPermute(_engine.BatchMatMul(weights, v), new[] { 1, 0, 2 }), new[] { length, _o.ModelDim });
        x = _engine.TensorAdd(x, Drop(l.Output.Forward(context), training, random));
        var hidden = Drop(_engine.ReLU(l.Intermediate.Forward(l.FeedForwardNorm.Forward(x))), training, random);
        return _engine.TensorAdd(x, Drop(l.FeedForwardOutput.Forward(hidden), training, random));
    }

    /// <summary>
    /// The recipe's loss (<c>lm_multitask</c>): the condition up to and including <c>&lt;generatetext&gt;</c> is masked,
    /// the transcript and <c>&lt;sos/eos&gt;</c> are predicted, with label smoothing (<c>lsm_weight</c>) as ESPnet's
    /// <c>LabelSmoothingLoss</c> computes it, summed over the tokens (<c>length_normalized_loss: false</c>).
    /// </summary>
    public Tensor<T> AsrLoss(int[] prompt, IReadOnlyList<int> transcript, bool training, Random random)
    {
        var target = transcript.Select(t => TextOffset + t).Concat(new[] { EndToken }).ToArray();
        var input = prompt.Concat(target.Take(target.Length - 1)).ToArray();
        var logits = Forward(input, training, random);
        int first = prompt.Length - 1, v = VocabSize;
        var rows = new Tensor<int>(new[] { target.Length }, new Vector<int>(Enumerable.Range(first, target.Length).ToArray()));
        var logProbabilities = _engine.TensorLogSoftmax(_engine.TensorIndexSelect(logits, rows, 0), axis: 1);
        double smoothing = _o.LabelSmoothing, other = smoothing / (v - 1), confidence = 1.0 - smoothing;
        var distribution = new Tensor<T>(new[] { target.Length, v });
        for (int r = 0; r < target.Length; r++)
            for (int c = 0; c < v; c++) distribution[r, c] = NumOps.FromDouble(c == target[r] ? confidence : other);
        // KL(q || p) = Σ q log q − Σ q log p; the first term is a constant.
        double entropy = (confidence > 0 ? confidence * Math.Log(confidence) : 0) + (other > 0 ? (v - 1) * other * Math.Log(other) : 0);
        var crossEntropy = _engine.TensorNegate(_engine.ReduceSum(_engine.TensorMultiply(logProbabilities, distribution), null));
        return _engine.TensorAddScalar(_engine.Reshape(crossEntropy, new[] { 1 }), NumOps.FromDouble(entropy * target.Length));
    }

    /// <summary>The logits <c>[1, vocabulary]</c> of the transcript's first token after the ASR condition.</summary>
    public Tensor<T> FirstLogits(int[] prompt) =>
        _engine.TensorSlice(Forward(prompt, training: false, random: Tensors.Helpers.RandomHelper.CreateSeededRandom(0)), new[] { prompt.Length - 1, 0 }, new[] { 1, VocabSize });

    /// <summary>The transcript's text-token indices by greedy decoding over the text tokens, until <c>&lt;sos/eos&gt;</c>.</summary>
    public int[] Greedy(int[] prompt, int maxTokens)
    {
        var sequence = new List<int>(prompt);
        var answer = new List<int>();
        for (int step = 0; step < maxTokens; step++)
        {
            var logits = Forward(sequence, training: false, random: Tensors.Helpers.RandomHelper.CreateSeededRandom(0));
            int last = sequence.Count - 1, best = EndToken;
            for (int c = TextOffset; c < EndToken; c++)
                if (NumOps.GreaterThan(logits[last, c], logits[last, best])) best = c;
            if (best == EndToken) break;
            answer.Add(best - TextOffset);
            sequence.Add(best);
        }
        return answer.ToArray();
    }

    /// <summary>
    /// Fits the speech tokenizer's k-means (Lloyd's algorithm from k-means++ seeds) on the HuBERT features of
    /// <paramref name="recordings"/>, as the recipe's stage 3a fits it.
    /// </summary>
    public void FitUnits(IEnumerable<IReadOnlyList<double>> recordings, int iterations, Random random)
    {
        var frames = new List<double[]>();
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            foreach (var recording in recordings)
            {
                var chunk = new Tensor<T>(new[] { recording.Count });
                for (int i = 0; i < recording.Count; i++) chunk[i] = NumOps.FromDouble(recording[i]);
                var features = Units.Network.Forward(chunk, Units.Layer)[Units.Layer];
                for (int f = 0; f < features.Shape[0]; f++)
                    frames.Add(Enumerable.Range(0, features.Shape[1]).Select(i => NumOps.ToDouble(features[f, i])).ToArray());
            }
        int k = Units.Centroids.GetLength(0), width = Units.Centroids.GetLength(1);
        if (frames.Count < k) throw new ArgumentException($"{frames.Count} HuBERT frames cannot fit {k} clusters.", nameof(recordings));
        static double Distance(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.Length; i++) { double t = a[i] - b[i]; s += t * t; } return s; }
        var centers = new List<double[]> { frames[random.Next(frames.Count)] };
        while (centers.Count < k)
        {
            var nearest = frames.Select(f => centers.Min(c => Distance(f, c))).ToArray();
            double total = nearest.Sum(), pick = random.NextDouble() * total;
            int chosen = 0;
            for (double running = 0; chosen < frames.Count - 1; chosen++) { running += nearest[chosen]; if (running >= pick) break; }
            centers.Add(frames[chosen]);
        }
        var assignment = new int[frames.Count];
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            for (int f = 0; f < frames.Count; f++)
            {
                int best = 0;
                for (int c = 1; c < k; c++) if (Distance(frames[f], centers[c]) < Distance(frames[f], centers[best])) best = c;
                assignment[f] = best;
            }
            for (int c = 0; c < k; c++)
            {
                var members = Enumerable.Range(0, frames.Count).Where(f => assignment[f] == c).ToList();
                if (members.Count == 0) continue;
                centers[c] = Enumerable.Range(0, width).Select(i => members.Average(f => frames[f][i])).ToArray();
            }
        }
        var fitted = new double[k, width];
        for (int c = 0; c < k; c++)
            for (int i = 0; i < width; i++) fitted[c, i] = centers[c][i];
        Units.Centroids = fitted;
    }
}
