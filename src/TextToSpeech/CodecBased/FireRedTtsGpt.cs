using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes and special tokens of FireRedTTS's text-to-speech language model (the release's <c>gpt</c> config).</summary>
/// <param name="Emotions">The instruction-tuning emotion table's rows (§4.2: neutral, happy, sad, angry); 0 for none.</param>
/// <param name="ParalinguisticLabels">The embedding-injection labels (§4.2, Table 1) plus the "none" row 0; 0 for none.</param>
internal sealed record FireRedTtsGptConfiguration(int ModelDim, int Heads, int Layers, int MaxTextTokens, int MaxAudioTokens, int TextTokens,
    int AudioTokens, int StartText, int StopText, int StartAudio, int StopAudio, int SpeakerDim, int ReferenceHidden, double Dropout,
    int Emotions, int ParalinguisticLabels);

/// <summary>How FireRedTTS samples semantic tokens (the release's <c>do_gpt_inference</c>: Hugging Face <c>generate</c>).</summary>
/// <param name="Candidates">Sequences sampled together (<c>num_return_sequences</c>).</param>
/// <param name="CandidateRank">The kept sequence's rank among them by length, shortest first (the release keeps 1, the
/// second shortest).</param>
/// <param name="SemanticOnly">Never emit the start token (the release does not constrain; a start token would fail in the
/// flow, as in the release).</param>
internal sealed record FireRedTtsDecoding(bool Sample, int TopK, double TopP, double Temperature, double RepetitionPenalty, int Candidates,
    int CandidateRank, int MaxTokens, bool SemanticOnly = false);

/// <summary>
/// FireRedTTS's decoder-only text-to-speech language model (Guo et al. 2024, §3.2; the release's
/// <c>fireredtts/modules/gpt/gpt.py</c>, XTTS's GPT): the speaker embedding through <c>reference_embedding</c> (Linear,
/// tanh, Linear) as one conditioning latent, the text <c>[BT, text, ET]</c> with learned text positions, the semantic tokens
/// <c>[BA, tokens…]</c> with learned audio positions, a GPT-2 stack, <c>final_norm</c>, and the text and audio heads.
/// </summary>
/// <remarks>
/// <para>Instruction tuning (§4.2), opt-in: an emotion embedding inserted after the speaker latent, and a paralinguistic
/// label embedding added to the text tokens it marks. The paper gives neither table's initialization; the emotion table
/// starts as <c>nn.Embedding</c>'s N(0, 1) (as the text table), the label table at zero so a pretrained model's behaviour
/// is unchanged until it trains.</para>
/// </remarks>
internal sealed class FireRedTtsGpt<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public FireRedTtsGpt(IEngine engine, List<LayerBase<T>> layers, FireRedTtsGptConfiguration configuration, Random random)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        int d = c.ModelDim;
        // nn.Embedding's default N(0, 1) for the token tables, LearnedPositionEmbeddings' N(0, 0.02), nn.Linear's default.
        TextEmbedding = IndexTtsOps.Embedding(layers, random, c.TextTokens, d, 1.0);
        AudioEmbedding = IndexTtsOps.Embedding(layers, random, c.AudioTokens, d, 1.0);
        Gpt = new IndexTtsGpt2<T>(engine, layers, random, d, c.Heads, c.Layers, c.Dropout);
        AudioPositions = IndexTtsOps.Embedding(layers, random, c.MaxAudioTokens + 2 + 1, d, 0.02);
        TextPositions = IndexTtsOps.Embedding(layers, random, c.MaxTextTokens + 2, d, 0.02);
        FinalNorm = IndexTtsOps.LayerNorm(layers, d);
        TextHead = IndexTtsOps.Linear(engine, layers, random, d, c.TextTokens);
        AudioHead = IndexTtsOps.Linear(engine, layers, random, d, c.AudioTokens);
        Reference1 = IndexTtsOps.Linear(engine, layers, random, c.SpeakerDim, c.ReferenceHidden);
        Reference2 = IndexTtsOps.Linear(engine, layers, random, c.ReferenceHidden, d);
        if (c.Emotions > 0) Emotion = IndexTtsOps.Embedding(layers, random, c.Emotions, d, 1.0);
        if (c.ParalinguisticLabels > 0)
        {
            Paralinguistic = new TiedEmbeddingLayer<T>(c.ParalinguisticLabels, d);
            layers.Add(Paralinguistic);
            Paralinguistic.Reinitialize(() => 0.0);
        }
    }

    public FireRedTtsGptConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> TextEmbedding { get; }
    public TiedEmbeddingLayer<T> AudioEmbedding { get; }
    public IndexTtsGpt2<T> Gpt { get; }
    public TiedEmbeddingLayer<T> AudioPositions { get; }
    public TiedEmbeddingLayer<T> TextPositions { get; }
    public LayerNormalizationLayer<T> FinalNorm { get; }
    public DenseLayer<T> TextHead { get; }
    public DenseLayer<T> AudioHead { get; }
    public DenseLayer<T> Reference1 { get; }
    public DenseLayer<T> Reference2 { get; }
    public TiedEmbeddingLayer<T>? Emotion { get; }
    public TiedEmbeddingLayer<T>? Paralinguistic { get; }

    /// <summary>The most semantic tokens one generation yields (<c>max_gen_mel_tokens</c>, the stop token included).</summary>
    public int MaxGeneratedTokens => Configuration.MaxAudioTokens - 1 - 2;

    /// <summary>The prompt rows <c>[1 or 2, d]</c>: <c>reference_embedding</c> of a speaker embedding <c>[SpeakerDim]</c>,
    /// then the emotion's row when one is given.</summary>
    public Tensor<T> Prefix(Tensor<T> speaker, int? emotion = null)
    {
        var c = Configuration;
        var x = _engine.Reshape(speaker, new[] { 1, c.SpeakerDim });
        var latent = Reference2.Forward(_engine.Tanh(Reference1.Forward(x)));
        if (emotion is null) return latent;
        var table = Emotion ?? throw new InvalidOperationException("This model has no emotion embedding; enable instruction tuning.");
        if (emotion < 0 || emotion >= c.Emotions) throw new ArgumentOutOfRangeException(nameof(emotion));
        return _engine.TensorConcatenate(new[] { latent, table.Forward(IndexTtsOps.Ids<T>(new[] { emotion.Value })) }, 0);
    }

    /// <summary>The text rows <c>[L + 2, d]</c>: <c>[BT, text…, ET]</c> embedded with text positions 0 …, plus each marked
    /// token's paralinguistic label embedding (label 0 is none; BT and ET carry none).</summary>
    public Tensor<T> TextRows(IReadOnlyList<int> text, IReadOnlyList<int>? labels = null)
    {
        var c = Configuration;
        var ids = new List<int>(text.Count + 2) { c.StartText };
        ids.AddRange(text);
        ids.Add(c.StopText);
        if (ids.Count > TextPositions.VocabularySize) throw new ArgumentException($"At most {c.MaxTextTokens} text tokens fit.", nameof(text));
        if (ids.Any(id => id < 0 || id >= c.TextTokens)) throw new ArgumentOutOfRangeException(nameof(text), $"Text ids lie in [0, {c.TextTokens}).");
        var rows = _engine.TensorAdd(TextEmbedding.Forward(IndexTtsOps.Ids<T>(ids)),
            TextPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(0, ids.Count).ToArray())));
        if (labels is null || labels.All(l => l == 0)) return rows;
        if (labels.Count != text.Count) throw new ArgumentException("One label per text token.", nameof(labels));
        var table = Paralinguistic ?? throw new InvalidOperationException("This model has no paralinguistic embedding; enable instruction tuning.");
        if (labels.Any(l => l < 0 || l >= c.ParalinguisticLabels)) throw new ArgumentOutOfRangeException(nameof(labels));
        var all = new List<int> { 0 };
        all.AddRange(labels);
        all.Add(0);
        return _engine.TensorAdd(rows, table.Forward(IndexTtsOps.Ids<T>(all)));
    }

    /// <summary>Audio rows <c>[n, d]</c> of tokens at audio positions <paramref name="start"/> ….</summary>
    public Tensor<T> AudioRows(IReadOnlyList<int> tokens, int start = 0)
    {
        if (start + tokens.Count > AudioPositions.VocabularySize)
            throw new ArgumentException($"At most {Configuration.MaxAudioTokens} semantic tokens fit.", nameof(tokens));
        return _engine.TensorAdd(AudioEmbedding.Forward(IndexTtsOps.Ids<T>(tokens)),
            AudioPositions.Forward(IndexTtsOps.Ids<T>(Enumerable.Range(start, tokens.Count).ToArray())));
    }

    /// <summary>
    /// XTTS's training sequences for one utterance whose semantic tokens cover its whole recording (the recording is
    /// <c>n · code_stride_len</c> samples long, so <c>code_lengths</c> is n + 3): the audio inputs
    /// <c>[BA, tokens, EA × 4]</c> and the targets the loss reads, <c>[tokens, EA × 4]</c> (<c>set_mel_padding</c> turns the
    /// three zeros the codes are padded with into EA; the last target is ignored, <c>mel_targets[l + 1:] = -1</c>), and
    /// the text targets <c>[text, ET]</c> (the last, ET again, ignored likewise). Each target is the next input's token, so
    /// the loss reads the first <c>targets.Length</c> rows of the logits.
    /// </summary>
    public (int[] AudioInputs, int[] AudioTargets, int[] TextTargets) XttsSequences(IReadOnlyList<int> text, IReadOnlyList<int> tokens)
    {
        var c = Configuration;
        var inputs = new List<int> { c.StartAudio };
        inputs.AddRange(tokens);
        inputs.AddRange(Enumerable.Repeat(c.StopAudio, 4));
        var targets = tokens.Concat(Enumerable.Repeat(c.StopAudio, 4)).ToArray();
        return (inputs.ToArray(), targets, text.Concat(new[] { c.StopText }).ToArray());
    }

    /// <summary>
    /// XTTS's teacher-forced <c>forward</c> for one utterance: the text logits <c>[L + 2, TextTokens]</c> over the text
    /// inputs <c>[BT, text, ET]</c> and the audio logits <c>[m, AudioTokens]</c> over <paramref name="audioInputs"/>, after
    /// the prompt rows.
    /// </summary>
    /// <param name="maskFinalKeys">XTTS's attention mask: <c>attn_mask_text[l + 1:]</c> and <c>attn_mask_mel[l + 1:]</c>
    /// hide the last text input (ET) and the last audio input from every query, as training does.</param>
    public (Tensor<T> Text, Tensor<T> Audio) Forward(Tensor<T> prefix, IReadOnlyList<int> text, IReadOnlyList<int>? labels,
        IReadOnlyList<int> audioInputs, bool training = false, Random? random = null, bool maskFinalKeys = false)
    {
        var textRows = TextRows(text, labels);
        var x = _engine.TensorConcatenate(new[] { prefix, textRows, AudioRows(audioInputs) }, 0);
        int length = x.Shape[0], d = Configuration.ModelDim, offset = prefix.Shape[0], textLength = textRows.Shape[0];
        bool[]? masked = null;
        if (maskFinalKeys)
        {
            masked = new bool[length];
            masked[offset + textLength - 1] = true;
            masked[length - 1] = true;
        }
        var hidden = _engine.Reshape(Gpt.Forward(_engine.Reshape(x, new[] { 1, length, d }), null, training, random, masked), new[] { length, d });
        var states = FinalNorm.Forward(_engine.TensorNarrow(hidden, 0, offset, length - offset));
        return (TextHead.Forward(_engine.TensorNarrow(states, 0, 0, textLength)),
            AudioHead.Forward(_engine.TensorNarrow(states, 0, textLength, audioInputs.Count)));
    }

    /// <summary>
    /// The release's <c>generate</c> through Hugging Face's sampling: <see cref="FireRedTtsDecoding.Candidates"/> sequences
    /// decoded together from <c>[prefix, BT text ET, BA]</c> (key-value cached), each step's logits through the repetition
    /// penalty (over every input id: the prompt's placeholder ids 1, BA and the tokens so far), temperature, top-k and
    /// top-p, then sampled (or the most likely token); a finished sequence continues with the stop token. Each sequence
    /// is returned up to its first stop token (whole when it has none).
    /// </summary>
    public List<int[]> Generate(Tensor<T> prefix, IReadOnlyList<int> text, IReadOnlyList<int>? labels, FireRedTtsDecoding decoding, Random random)
    {
        var c = Configuration;
        int d = c.ModelDim, n = decoding.Candidates;
        if (n < 1) throw new ArgumentOutOfRangeException(nameof(decoding), "At least one candidate.");
        var prompt = _engine.TensorConcatenate(new[] { prefix, TextRows(text, labels) }, 0);
        int promptLength = prompt.Shape[0];
        var first = _engine.TensorConcatenate(new[] { prompt, AudioRows(new[] { c.StartAudio }) }, 0);
        var batch = _engine.TensorTile(_engine.Reshape(first, new[] { 1, promptLength + 1, d }), new[] { n, 1, 1 });
        var cache = new IndexTtsGpt2Cache<T>();
        var hidden = Gpt.Forward(batch, cache);                                                           // [n, P + 1, d]
        var last = _engine.Reshape(_engine.TensorNarrow(hidden, 1, promptLength, 1), new[] { n, d });

        var sequences = Enumerable.Range(0, n).Select(_ => new List<int>()).ToList();
        var finished = new bool[n];
        // Hugging Face penalizes every id in input_ids: the prompt's placeholders (1) and BA, then each sampled token.
        var seen = Enumerable.Range(0, n).Select(_ => new HashSet<int> { 1, c.StartAudio }).ToList();
        int limit = Math.Min(decoding.MaxTokens, MaxGeneratedTokens);
        for (int step = 0; step < limit; step++)
        {
            var logits = AudioHead.Forward(FinalNorm.Forward(last));                                       // [n, V]
            var next = new int[n];
            for (int b = 0; b < n; b++)
            {
                var row = new double[c.AudioTokens];
                for (int v = 0; v < row.Length; v++) row[v] = NumOps.ToDouble(logits[b, v]);
                if (decoding.SemanticOnly) row[c.StartAudio] = double.NegativeInfinity;
                int token = SelectToken(row, seen[b], decoding, random);
                if (finished[b]) token = c.StopAudio;
                next[b] = token;
                sequences[b].Add(token);
                seen[b].Add(token);
                if (token == c.StopAudio) finished[b] = true;
            }
            if (finished.All(f => f)) break;
            // The k-th generated token sits at audio position k (BA is position 0).
            var rows = _engine.TensorAdd(AudioEmbedding.Forward(IndexTtsOps.Ids<T>(next)),
                _engine.TensorTile(AudioPositions.Forward(IndexTtsOps.Ids<T>(new[] { step + 1 })), new[] { n, 1 }));
            hidden = Gpt.Forward(_engine.Reshape(rows, new[] { n, 1, d }), cache);
            last = _engine.Reshape(hidden, new[] { n, d });
        }
        return sequences.Select(s =>
        {
            int stop = s.IndexOf(c.StopAudio);
            return (stop < 0 ? s : s.Take(stop)).ToArray();
        }).ToList();
    }

    /// <summary>The release's choice among the candidates: sorted by length (stably, shortest first), the one at
    /// <paramref name="rank"/> (clamped to the last).</summary>
    public static int[] Choose(List<int[]> candidates, int rank)
    {
        var sorted = candidates.Select((s, i) => (s, i)).OrderBy(p => p.s.Length).ThenBy(p => p.i).Select(p => p.s).ToList();
        return sorted[Math.Min(rank, sorted.Count - 1)];
    }

    /// <summary>
    /// One token from a step's logits after Hugging Face's processors (<c>RepetitionPenaltyLogitsProcessor</c>,
    /// <c>TemperatureLogitsWarper</c>, <c>TopKLogitsWarper</c>, <c>TopPLogitsWarper</c>, in that order): the processed scores
    /// are returned in <paramref name="row"/> (−∞ where filtered), and the token sampled from their softmax, or the
    /// arg-max without sampling.
    /// </summary>
    internal static int SelectToken(double[] row, ISet<int> seen, FireRedTtsDecoding decoding, Random random)
    {
        Process(row, seen, decoding);
        if (!decoding.Sample)
        {
            int best = 0;
            for (int v = 1; v < row.Length; v++) if (row[v] > row[best]) best = v;
            return best;
        }
        double max = row.Max(), sum = 0;
        var p = new double[row.Length];
        for (int v = 0; v < row.Length; v++) sum += p[v] = double.IsNegativeInfinity(row[v]) ? 0 : Math.Exp(row[v] - max);
        double u = random.NextDouble() * sum, acc = 0;
        int lastKept = -1;
        for (int v = 0; v < p.Length; v++)
        {
            if (p[v] == 0) continue;
            lastKept = v;
            acc += p[v];
            if (u < acc) return v;
        }
        return lastKept;
    }

    /// <summary>The logits processors in place (see <see cref="SelectToken"/>). Without sampling only the repetition
    /// penalty applies, as Hugging Face's greedy search uses no warpers.</summary>
    internal static void Process(double[] row, ISet<int> seen, FireRedTtsDecoding decoding)
    {
        double penalty = decoding.RepetitionPenalty;
        if (penalty != 1.0)
            foreach (int id in seen)
                if (id >= 0 && id < row.Length) row[id] = row[id] < 0 ? row[id] * penalty : row[id] / penalty;
        if (!decoding.Sample) return;
        if (decoding.Temperature != 1.0)
            for (int v = 0; v < row.Length; v++) row[v] /= decoding.Temperature;
        int k = Math.Min(Math.Max(decoding.TopK, 1), row.Length);
        if (decoding.TopK > 0 && k < row.Length)
        {
            double kth = row.OrderByDescending(x => x).ElementAt(k - 1);
            for (int v = 0; v < row.Length; v++) if (row[v] < kth) row[v] = double.NegativeInfinity;
        }
        if (decoding.TopP < 1.0)
        {
            // Sorted ascending (stably, as torch.sort does on CPU for these sizes), the cumulative softmax; tokens whose
            // cumulative probability is at most 1 − top_p are removed, the most likely always kept.
            var order = Enumerable.Range(0, row.Length).OrderBy(v => row[v]).ThenBy(v => v).ToArray();
            double max = row.Max(), sum = 0;
            var e = new double[row.Length];
            for (int i = 0; i < order.Length; i++) sum += e[i] = double.IsNegativeInfinity(row[order[i]]) ? 0 : Math.Exp(row[order[i]] - max);
            double cumulative = 0;
            for (int i = 0; i < order.Length - 1; i++)
            {
                cumulative += e[i] / sum;
                if (cumulative <= 1 - decoding.TopP) row[order[i]] = double.NegativeInfinity;
            }
        }
    }

    /// <summary>Loads the release's <c>fireredtts_gpt.pt</c> names (<c>model</c>'s state): <c>text_embedding</c>,
    /// <c>mel_embedding</c>, <c>gpt.h.*</c>, <c>gpt.ln_f</c>, the position tables, <c>final_norm</c>, the heads and
    /// <c>reference_embedding</c>. The checkpoint's text vocabulary has <paramref name="textRows"/> rows; a model with
    /// more (instruction tags) keeps the extra rows of the text table and head from its initialization.</summary>
    public void Load(Func<string, int[], double[]> read, int textRows)
    {
        var c = Configuration;
        int d = c.ModelDim;
        if (textRows < 1 || textRows > c.TextTokens) throw new ArgumentOutOfRangeException(nameof(textRows));
        TextEmbedding.LoadLeadingRows(read("text_embedding.weight", new[] { textRows, d }));
        AudioEmbedding.LoadTable(read("mel_embedding.weight", new[] { c.AudioTokens, d }));
        Gpt.Load(read, "gpt.");
        AudioPositions.LoadTable(read("mel_pos_embedding.emb.weight", new[] { c.MaxAudioTokens + 3, d }));
        TextPositions.LoadTable(read("text_pos_embedding.emb.weight", new[] { c.MaxTextTokens + 2, d }));
        IndexTtsOps.LoadNorm(_engine, read, FinalNorm, "final_norm", d);
        if (textRows == c.TextTokens)
        {
            IndexTtsOps.LoadLinear(_engine, read, TextHead, "text_head", d, c.TextTokens);
        }
        else
        {
            var weight = read("text_head.weight", new[] { textRows, d });
            var bias = read("text_head.bias", new[] { textRows });
            var w = TextHead.GetWeights();                                                                 // [in, out]
            var b = TextHead.GetBiases();
            for (int o = 0; o < textRows; o++)
            {
                for (int i = 0; i < d; i++) w[i, o] = NumOps.FromDouble(weight[o * d + i]);
                b[o] = NumOps.FromDouble(bias[o]);
            }
            _engine.InvalidatePersistentTensor(w);
            _engine.InvalidatePersistentTensor(b);
        }
        IndexTtsOps.LoadLinear(_engine, read, AudioHead, "mel_head", d, c.AudioTokens);
        IndexTtsOps.LoadLinear(_engine, read, Reference1, "reference_embedding.0", c.SpeakerDim, c.ReferenceHidden);
        IndexTtsOps.LoadLinear(_engine, read, Reference2, "reference_embedding.2", c.ReferenceHidden, d);
    }

    /// <summary>The release's text vocabulary rows (57,184 ranks, 156 special tokens and one spare row).</summary>
    internal const int ReleaseTextTokenCount = 57341;
}
