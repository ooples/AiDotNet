using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The sizes of VALL-E's two codec language models (Wang et al. 2023, §5.1).</summary>
/// <param name="TextTokens">Rows of the phoneme embeddings (the reference's <c>NUM_TEXT_TOKENS</c>, 512).</param>
/// <param name="AudioTokens">Codes per codebook (1024; the AR model adds the end token 1024).</param>
/// <param name="Codebooks">Codebooks (8).</param>
/// <param name="ModelDim">Embedding width (1024).</param>
/// <param name="Heads">Attention heads (16).</param>
/// <param name="Layers">Layers of each model (12).</param>
/// <param name="FeedForwardDim">Feed-forward width (4096).</param>
/// <param name="Dropout">Dropout (0.1).</param>
/// <param name="ShareEmbedding">Whether NAR head j shares its weights with acoustic embedding j + 1 (the reference's
/// <c>share_embedding</c>, on).</param>
/// <param name="Languages">Language-ID table rows (0: no language IDs, as in VALL-E; VALL-E X's languages).</param>
/// <param name="LanguagePlacement">Where the language embedding is added (VALL-E X).</param>
internal sealed record VallEConfiguration(
    int TextTokens, int AudioTokens, int Codebooks, int ModelDim, int Heads, int Layers, int FeedForwardDim,
    double Dropout, bool ShareEmbedding = true, int Languages = 0,
    VallELanguagePlacement LanguagePlacement = VallELanguagePlacement.AcousticTokens);

/// <summary>Where VALL-E X adds its language embedding.</summary>
public enum VallELanguagePlacement
{
    /// <summary>To the AR model's acoustic-token embeddings (Zhang et al. 2023, §3.3).</summary>
    AcousticTokens,

    /// <summary>To the phoneme embeddings of both the AR and the NAR model (Plachtaa/VALL-E-X).</summary>
    TextTokens,
}

/// <summary>
/// VALL-E's autoregressive and non-autoregressive codec language models (Wang et al. 2023, §4.2; lifeiteng/vall-e
/// <c>VALLE</c>), shared by the models of the VALL-E family.
/// </summary>
/// <remarks>
/// <para>
/// <b>AR</b> (§4.2.1): the phonemes <c>&lt;bos&gt; x &lt;eos&gt;</c> and the first codebook's codes, each embedded
/// with its own table and sinusoidal positions (each with a learned scale α), run through one causal Transformer in
/// which the phonemes see only phonemes and each code sees the phonemes and the codes before it; a bias-free head
/// predicts the next code or the end token 1024.
/// </para>
/// <para>
/// <b>NAR</b> (§4.2.2): for stage <c>i ∈ [2, 8]</c>, the embeddings of codebooks <c>1 … i − 1</c> are summed (Eq. 4),
/// an acoustic prompt with all eight codebooks summed precedes them, and a bidirectional Transformer whose layer norms
/// are adaptive to stage <c>i</c> (Eq. 5) predicts codebook <c>i</c>. Head <c>i</c> shares its weights with acoustic
/// embedding <c>i + 1</c> (the last head has its own).
/// </para>
/// <para>
/// <b>Training prompt</b> (§5.1): the NAR's prompt is a random 3-second segment of the same utterance, as the
/// reference's <c>prefix_mode 2</c> implements it: its codes are copied in front, and the segment's positions of the
/// predicted codebook are excluded from the loss. The reference shortens the segment to a quarter of an utterance
/// shorter than 12 seconds; the paper crops utterances to 10–20 seconds, so its prompts never needed that.
/// </para>
/// </remarks>
internal sealed class VallECore<T>
{
    private readonly IEngine _engine;

    public VallECore(IEngine engine, List<LayerBase<T>> arLayers, List<LayerBase<T>> narLayers, VallEConfiguration configuration)
    {
        _engine = engine;
        var c = Configuration = configuration;
        int d = c.ModelDim;
        var arStack = new VallETransformerConfiguration(d, c.Heads, c.Layers, c.FeedForwardDim, c.Dropout, AdaptiveNorm: false);
        var narStack = arStack with { AdaptiveNorm = true };

        ArTextEmbedding = Own(arLayers, new TiedEmbeddingLayer<T>(c.TextTokens, d));
        ArAudioEmbedding = Own(arLayers, new TiedEmbeddingLayer<T>(c.AudioTokens + 1, d));
        ArTextPosition = new VallEPositionalEncoding<T>(engine, arLayers, d, c.Dropout, learnScale: true);
        ArAudioPosition = new VallEPositionalEncoding<T>(engine, arLayers, d, c.Dropout, learnScale: true);
        ArDecoder = new VallETransformer<T>(engine, arLayers, arStack);
        ArPredict = Own(arLayers, new BiasFreeLinearLayer<T>(d, c.AudioTokens + 1));

        NarTextEmbedding = Own(narLayers, new TiedEmbeddingLayer<T>(c.TextTokens, d));
        for (int j = 0; j < c.Codebooks; j++)
            NarAudioEmbeddings.Add(Own(narLayers, new TiedEmbeddingLayer<T>(c.AudioTokens + (j == 0 ? 1 : 0), d)));
        NarTextPosition = new VallEPositionalEncoding<T>(engine, narLayers, d, 0.0, learnScale: false);
        NarAudioPosition = new VallEPositionalEncoding<T>(engine, narLayers, d, c.Dropout, learnScale: false);
        NarDecoder = new VallETransformer<T>(engine, narLayers, narStack);
        for (int j = 0; j < c.Codebooks - 1; j++)
            NarStageEmbeddings.Add(Own(narLayers, new TiedEmbeddingLayer<T>(1, d)));
        // Heads 1 … 6 share acoustic embeddings 2 … 7; the last head has its own weights (and so does every head
        // without sharing).
        for (int j = 0; j < c.Codebooks - 1; j++)
            NarOwnHeads.Add(c.ShareEmbedding && j < c.Codebooks - 2 ? null : Own(narLayers, new BiasFreeLinearLayer<T>(d, c.AudioTokens)));
        if (c.Languages > 0)
        {
            ArLanguageEmbedding = Own(arLayers, new TiedEmbeddingLayer<T>(c.Languages, d));
            if (c.LanguagePlacement == VallELanguagePlacement.TextTokens)
                NarLanguageEmbedding = Own(narLayers, new TiedEmbeddingLayer<T>(c.Languages, d));
        }
    }

    public VallEConfiguration Configuration { get; }

    /// <summary>The AR model's end token (1024).</summary>
    public int EndToken => Configuration.AudioTokens;

    public TiedEmbeddingLayer<T> ArTextEmbedding { get; }
    public TiedEmbeddingLayer<T> ArAudioEmbedding { get; }
    public VallEPositionalEncoding<T> ArTextPosition { get; }
    public VallEPositionalEncoding<T> ArAudioPosition { get; }
    public VallETransformer<T> ArDecoder { get; }
    public BiasFreeLinearLayer<T> ArPredict { get; }
    public TiedEmbeddingLayer<T> NarTextEmbedding { get; }
    public List<TiedEmbeddingLayer<T>> NarAudioEmbeddings { get; } = new();
    public VallEPositionalEncoding<T> NarTextPosition { get; }
    public VallEPositionalEncoding<T> NarAudioPosition { get; }
    public VallETransformer<T> NarDecoder { get; }
    public List<TiedEmbeddingLayer<T>> NarStageEmbeddings { get; } = new();
    public List<BiasFreeLinearLayer<T>?> NarOwnHeads { get; } = new();
    public TiedEmbeddingLayer<T>? ArLanguageEmbedding { get; }
    public TiedEmbeddingLayer<T>? NarLanguageEmbedding { get; }

    // Adds each position's language embedding, when the model has one and the caller gives the positions' languages.
    private Tensor<T> WithLanguage(Tensor<T> embedded, TiedEmbeddingLayer<T>? table, IReadOnlyList<int>? languages)
    {
        if (table is null) return embedded;
        if (languages is null) throw new ArgumentException("This model reads a language ID for every position.");
        if (languages.Count != embedded.Shape[0])
            throw new ArgumentException($"{languages.Count} language IDs for {embedded.Shape[0]} positions.");
        return _engine.TensorAdd(embedded, table.Forward(Ids(languages)));
    }

    private static TLayer Own<TLayer>(List<LayerBase<T>> layers, TLayer layer) where TLayer : LayerBase<T>
    {
        layers.Add(layer);
        return layer;
    }

    private static Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = ops.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>PyTorch's defaults, which the reference keeps: embeddings from N(0, 1), bias-free heads uniform in
    /// ±1/√fan-in, the Transformers as <see cref="VallETransformer{T}.InitializeLikePyTorch"/>, α at 1.</summary>
    public void InitializeLikePyTorch(Random random)
    {
        double Normal()
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        }
        var ops = MathHelper.GetNumericOperations<T>();
        void Head(BiasFreeLinearLayer<T> head)
        {
            double bound = 1 / Math.Sqrt(head.InputSize);
            var values = new Vector<T>(head.InputSize * head.OutputSize);
            for (int i = 0; i < values.Length; i++) values[i] = ops.FromDouble((2 * random.NextDouble() - 1) * bound);
            head.SetParameters(values);
        }
        ArTextEmbedding.Reinitialize(Normal);
        ArAudioEmbedding.Reinitialize(Normal);
        ArDecoder.InitializeLikePyTorch(random);
        Head(ArPredict);
        ArLanguageEmbedding?.Reinitialize(Normal);
        NarLanguageEmbedding?.Reinitialize(Normal);
        NarTextEmbedding.Reinitialize(Normal);
        foreach (var embedding in NarAudioEmbeddings) embedding.Reinitialize(Normal);
        NarDecoder.InitializeLikePyTorch(random);
        foreach (var stage in NarStageEmbeddings) stage.Reinitialize(Normal);
        foreach (var head in NarOwnHeads)
            if (head is not null) Head(head);
    }

    // ---------------------------------------------------------------- AR

    /// <summary>
    /// The AR model's logits <c>[codes, audioTokens + 1]</c> for each code position: phonemes <paramref name="text"/>
    /// (framed), first-codebook codes <paramref name="codes"/>; position <c>t</c> predicts the code after
    /// <c>codes[t]</c>.
    /// </summary>
    public Tensor<T> ArLogits(IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training, Random random,
        IReadOnlyList<int>? textLanguages = null, IReadOnlyList<int>? codeLanguages = null)
    {
        int textLength = text.Count, codeLength = codes.Count, total = textLength + codeLength;
        bool onText = Configuration.LanguagePlacement == VallELanguagePlacement.TextTokens;
        var x = ArTextPosition.Forward(WithLanguage(ArTextEmbedding.Forward(Ids(text)), onText ? ArLanguageEmbedding : null,
            textLanguages), training, random);
        var y = ArAudioPosition.Forward(WithLanguage(ArAudioEmbedding.Forward(Ids(codes)), onText ? null : ArLanguageEmbedding,
            codeLanguages), training, random);
        var xy = _engine.TensorConcatenate(new[] { x, y }, 0);
        // Phonemes attend to phonemes only; each code to the phonemes and the codes up to itself.
        var ops = MathHelper.GetNumericOperations<T>();
        var blocked = ops.FromDouble(-1e9);
        var mask = new Tensor<T>(new[] { total, total });
        for (int i = 0; i < total; i++)
            for (int j = 0; j < total; j++)
                if (i < textLength ? j >= textLength : j > i) mask[i, j] = blocked;
        var hidden = ArDecoder.Forward(xy, null, mask, training, random);
        var codeStates = _engine.TensorSlice(hidden, new[] { textLength, 0 }, new[] { codeLength, Configuration.ModelDim });
        return ArPredict.Forward(codeStates);
    }

    /// <summary>The AR loss (§4.2.1): the summed cross-entropy of each next code and, after the last, the end token,
    /// teacher-forced (the reference's <c>reduction="sum"</c>).</summary>
    public Tensor<T> ArLoss(IReadOnlyList<int> text, IReadOnlyList<int> codes, bool training, Random random,
        IReadOnlyList<int>? textLanguages = null, IReadOnlyList<int>? codeLanguages = null)
    {
        var logits = ArLogits(text, codes, training, random, textLanguages, codeLanguages);                                                // [T, V]
        var targets = new int[codes.Count];
        for (int t = 0; t < codes.Count; t++) targets[t] = t + 1 < codes.Count ? codes[t + 1] : EndToken;
        return CrossEntropySum(logits, targets);
    }

    /// <summary>
    /// Continues the first codebook after <paramref name="prompt"/> (§4.3): samples each next code (temperature, top
    /// k; 0 keeps every code), stopping when the most likely or the sampled code is the end token, or after
    /// <paramref name="maxNewCodes"/>. Returns only the new codes.
    /// </summary>
    public List<int> ArGenerate(IReadOnlyList<int> text, IReadOnlyList<int> prompt, double temperature, int topK,
        int maxNewCodes, Random random, IReadOnlyList<int>? textLanguages = null, int promptLanguage = 0, int targetLanguage = 0)
    {
        var codes = new List<int>(prompt);
        var generated = new List<int>();
        while (generated.Count <= maxNewCodes)
        {
            // The prompt's codes speak the prompt's language, the new ones the target's.
            var codeLanguages = ArLanguageEmbedding is null ? null
                : Enumerable.Range(0, codes.Count).Select(i => i < prompt.Count ? promptLanguage : targetLanguage).ToArray();
            var logits = ArLogits(text, codes, training: false, random, textLanguages, codeLanguages);
            int last = codes.Count - 1, vocabulary = logits.Shape[1];
            var row = new double[vocabulary];
            int best = 0;
            for (int v = 0; v < vocabulary; v++)
            {
                row[v] = MathHelper.GetNumericOperations<T>().ToDouble(logits[last, v]);
                if (row[v] > row[best]) best = v;
            }
            int sample = SampleTopK(row, temperature, topK, random);
            if (best == EndToken || sample == EndToken || generated.Count == maxNewCodes)
                break;
            codes.Add(sample);
            generated.Add(sample);
        }
        if (generated.Count == 0)
            throw new InvalidOperationException("VALL-E's AR model ended the utterance before its first code; a trained model does not.");
        return generated;
    }

    // topk_sampling: logits / temperature, keep the top k (k ≤ 0 keeps all), sample from the softmax.
    internal static int SampleTopK(double[] logits, double temperature, int topK, Random random)
    {
        int n = logits.Length;
        var scaled = new double[n];
        for (int v = 0; v < n; v++) scaled[v] = logits[v] / temperature;
        if (topK > 0 && topK < n)
        {
            var threshold = scaled.OrderByDescending(v => v).ElementAt(topK - 1);
            for (int v = 0; v < n; v++) if (scaled[v] < threshold) scaled[v] = double.NegativeInfinity;
        }
        double max = scaled.Max(), total = 0;
        for (int v = 0; v < n; v++) total += scaled[v] = Math.Exp(scaled[v] - max);
        double draw = random.NextDouble() * total;
        for (int v = 0; v < n; v++)
        {
            draw -= scaled[v];
            if (draw <= 0) return v;
        }
        return n - 1;
    }

    // ---------------------------------------------------------------- NAR

    /// <summary>
    /// The NAR model's logits <c>[frames, audioTokens]</c> for codebook <paramref name="stage"/> (1-based from the
    /// second codebook, 1 … 7) over the frames after the prompt: phonemes <paramref name="text"/>, the prompt's codes
    /// <paramref name="prompt"/> <c>[promptFrames, codebooks]</c> (all codebooks summed) and the target's codes
    /// <paramref name="codes"/> <c>[frames, codebooks]</c>, of which codebooks <c>0 … stage − 1</c> are summed.
    /// </summary>
    public Tensor<T> NarLogits(IReadOnlyList<int> text, int[,] prompt, int[,] codes, int stage, bool training, Random random,
        IReadOnlyList<int>? textLanguages = null)
    {
        var c = Configuration;
        int promptFrames = prompt.GetLength(0), frames = codes.GetLength(0), d = c.ModelDim;
        Tensor<T> Sum(int[,] source, int levels)
        {
            int length = source.GetLength(0);
            Tensor<T>? sum = null;
            for (int j = 0; j < levels; j++)
            {
                var column = new int[length];
                for (int t = 0; t < length; t++) column[t] = source[t, j];
                var embedded = NarAudioEmbeddings[j].Forward(Ids(column));
                sum = sum is null ? embedded : _engine.TensorAdd(sum, embedded);
            }
            return sum!;
        }
        var y = Sum(codes, stage);
        if (promptFrames > 0)
            y = _engine.TensorConcatenate(new[] { Sum(prompt, c.Codebooks), y }, 0);
        var x = NarTextPosition.Forward(WithLanguage(NarTextEmbedding.Forward(Ids(text)), NarLanguageEmbedding, textLanguages),
            training, random);
        var yPositioned = NarAudioPosition.Forward(y, training, random);
        var xy = _engine.TensorConcatenate(new[] { x, yPositioned }, 0);
        var stageEmbedding = NarStageEmbeddings[stage - 1].Forward(new Tensor<T>(new[] { 1 }));             // [1, d]
        var hidden = NarDecoder.Forward(xy, stageEmbedding, null, training, random);
        var targetStates = _engine.TensorSlice(hidden, new[] { text.Count + promptFrames, 0 }, new[] { frames, d });
        var ownHead = NarOwnHeads[stage - 1];
        return ownHead is not null ? ownHead.Forward(targetStates) : NarAudioEmbeddings[stage + 1].Logits(targetStates);
    }

    /// <summary>
    /// The NAR loss on one utterance (§4.2.2, §5.1): a uniformly drawn stage, a random 3-second prompt segment (at most
    /// a quarter of the utterance) copied in front, and the summed cross-entropy of the stage's codes outside the
    /// segment, scaled to the whole utterance as the reference scales it.
    /// </summary>
    public Tensor<T> NarLoss(IReadOnlyList<int> text, int[,] codes, int promptFrames, bool training, Random random)
    {
        var c = Configuration;
        int frames = codes.GetLength(0);
        int stage = 1 + random.Next(c.Codebooks - 1);
        int length = Math.Min(promptFrames, frames / 4);
        int start = random.Next(frames - length + 1);
        return NarLossAt(text, codes, stage, start, length, training, random);
    }

    /// <summary>The NAR loss for a given stage and prompt segment <c>[start, start + length)</c>.</summary>
    public Tensor<T> NarLossAt(IReadOnlyList<int> text, int[,] codes, int stage, int start, int length, bool training, Random random)
    {
        var c = Configuration;
        int frames = codes.GetLength(0);
        var prompt = new int[length, c.Codebooks];
        for (int t = 0; t < length; t++)
            for (int j = 0; j < c.Codebooks; j++) prompt[t, j] = codes[start + t, j];
        var logits = NarLogits(text, prompt, codes, stage, training, random);
        var targets = new int[frames];
        for (int t = 0; t < frames; t++) targets[t] = t >= start && t < start + length ? -1 : codes[t, stage];
        var loss = CrossEntropySum(logits, targets);
        // × total / (total − prefix): the reference's normalization of the excluded positions.
        return length == 0 ? loss : _engine.TensorMultiplyScalar(loss,
            MathHelper.GetNumericOperations<T>().FromDouble((double)frames / (frames - length)));
    }

    /// <summary>
    /// The NAR loss with an acoustic prompt from another utterance of the same speaker (VALL-E X, Eq. 2): a uniformly
    /// drawn stage, the prompt's codes (all codebooks) in front, and the summed cross-entropy of the stage's codes.
    /// </summary>
    public Tensor<T> NarLossWithPrompt(IReadOnlyList<int> text, int[,] prompt, int[,] codes, bool training, Random random,
        IReadOnlyList<int>? textLanguages = null)
    {
        int stage = 1 + random.Next(Configuration.Codebooks - 1);
        var logits = NarLogits(text, prompt, codes, stage, training, random, textLanguages);
        var targets = new int[codes.GetLength(0)];
        for (int t = 0; t < targets.Length; t++) targets[t] = codes[t, stage];
        return CrossEntropySum(logits, targets);
    }

    /// <summary>Fills codebooks 2 … 8 of the frames after the prompt, greedily, one stage at a time (§4.3).</summary>
    public int[,] NarGenerate(IReadOnlyList<int> text, int[,] prompt, int[] firstCodebook, Random random,
        IReadOnlyList<int>? textLanguages = null)
    {
        var c = Configuration;
        int frames = firstCodebook.Length;
        var codes = new int[frames, c.Codebooks];
        for (int t = 0; t < frames; t++) codes[t, 0] = firstCodebook[t];
        var ops = MathHelper.GetNumericOperations<T>();
        for (int stage = 1; stage < c.Codebooks; stage++)
        {
            var logits = NarLogits(text, prompt, codes, stage, training: false, random, textLanguages);
            for (int t = 0; t < frames; t++)
            {
                int best = 0;
                for (int v = 1; v < logits.Shape[1]; v++)
                    if (ops.ToDouble(logits[t, v]) > ops.ToDouble(logits[t, best])) best = v;
                codes[t, stage] = best;
            }
        }
        return codes;
    }

    // Σ −log softmax(logits)[target] over the rows whose target is not −1.
    private Tensor<T> CrossEntropySum(Tensor<T> logits, int[] targets)
    {
        var logProbabilities = _engine.TensorLogSoftmax(logits, axis: -1);
        var ops = MathHelper.GetNumericOperations<T>();
        var selection = new Tensor<T>(logits._shape);
        for (int t = 0; t < targets.Length; t++)
            if (targets[t] >= 0) selection[t, targets[t]] = ops.FromDouble(-1.0);
        return _engine.ReduceSum(_engine.TensorMultiply(logProbabilities, selection), new[] { 0, 1 }, keepDims: false);
    }

    // ---------------------------------------------------------------- weights

    /// <summary>Loads the reference's <c>VALLE</c> state dictionary.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        var c = Configuration;
        int d = c.ModelDim;
        ArTextEmbedding.LoadTable(read("ar_text_embedding.word_embeddings.weight", new[] { c.TextTokens, d }));
        ArAudioEmbedding.LoadTable(read("ar_audio_embedding.word_embeddings.weight", new[] { c.AudioTokens + 1, d }));
        ArTextPosition.Alpha!.LoadTable(read("ar_text_position.alpha", new[] { 1 }));
        ArAudioPosition.Alpha!.LoadTable(read("ar_audio_position.alpha", new[] { 1 }));
        ArDecoder.LoadTorchWeights(_engine, "ar_decoder", read);
        TorchParameters.Linear(ArPredict, d, c.AudioTokens + 1, read("ar_predict_layer.weight", new[] { c.AudioTokens + 1, d }));
        NarTextEmbedding.LoadTable(read("nar_text_embedding.word_embeddings.weight", new[] { c.TextTokens, d }));
        // The NAR's positional scales are frozen at 1 (requires_grad=False) but stored.
        foreach (var name in new[] { "nar_text_position.alpha", "nar_audio_position.alpha" })
            if (read(name, new[] { 1 })[0] != 1.0)
                throw new InvalidDataException($"'{name}' is not 1; the NAR model's positional scale is fixed.");
        for (int j = 0; j < c.Codebooks; j++)
            NarAudioEmbeddings[j].LoadTable(read($"nar_audio_embeddings.{j}.word_embeddings.weight",
                new[] { c.AudioTokens + (j == 0 ? 1 : 0), d }));
        NarDecoder.LoadTorchWeights(_engine, "nar_decoder", read);
        for (int j = 0; j < c.Codebooks - 1; j++)
        {
            NarStageEmbeddings[j].LoadTable(read($"nar_stage_embeddings.{j}.word_embeddings.weight", new[] { 1, d }));
            if (NarOwnHeads[j] is { } head)
                TorchParameters.Linear(head, d, c.AudioTokens, read($"nar_predict_layers.{j}.weight", new[] { c.AudioTokens, d }));
        }
        // Plachtaa/VALL-E-X's names for VALL-E X's language tables.
        ArLanguageEmbedding?.LoadTable(read("ar_language_embedding.word_embeddings.weight", new[] { c.Languages, d }));
        NarLanguageEmbedding?.LoadTable(read("nar_language_embedding.word_embeddings.weight", new[] { c.Languages, d }));
    }
}
