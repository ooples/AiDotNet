using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// CosyVoice 2's flow encoder (FunAudioLLM/CosyVoice <c>cosyvoice/transformer/upsample_encoder.py</c>
/// <c>UpsampleConformerEncoder</c>): a Conformer stack (<c>embed</c>, <c>encoders</c>) after a look-ahead convolution, a
/// 2× nearest-neighbour upsampling with a causal 5-tap convolution, and a second Conformer stack (<c>up_embed</c>, four
/// <c>up_encoders</c>) with the final LayerNorm. In streaming the stacks attend within chunks of
/// <c>StaticChunkSize</c> and twice that after the upsampling.
/// </summary>
/// <remarks>The reference fixes the look-ahead and upsampling convolutions at 512 channels; here they take the
/// encoder's width, which is 512 in every released configuration.</remarks>
internal sealed class UpsampleConformerEncoder<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <param name="configuration">The lower stack (its blocks, chunk size and dropout); the upper stack has four blocks.</param>
    /// <param name="lookahead">Frames the look-ahead convolution reads ahead (<c>pre_lookahead_len</c>, 3).</param>
    public UpsampleConformerEncoder(IEngine engine, List<LayerBase<T>> layers, WenetEncoderConfiguration configuration, int lookahead,
        Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        Lookahead = lookahead;
        int d = configuration.OutputSize;
        NormedConv1DLayer<T> Conv(int inputs, int outputs, int kernel)
        {
            var conv = new NormedConv1DLayer<T>(inputs, outputs, kernel, 1, 1, 1, 0, false, ConvolutionNormalization.None, useBias: true);
            layers.Add(conv);
            // nn.Conv1d's default initialization: weights and bias from U(±1/√(in·k)).
            double bound = 1.0 / Math.Sqrt(inputs * kernel);
            conv.LoadTorchWeights(Enumerable.Range(0, outputs * inputs * kernel).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray(),
                null, Enumerable.Range(0, outputs).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray());
            return conv;
        }
        Lower = new WenetEncoder<T>(engine, layers, configuration with { AfterNorm = false }, initialization);
        LookaheadConv1 = Conv(d, d, lookahead + 1);
        LookaheadConv2 = Conv(d, d, 3);
        UpConv = Conv(d, d, 5);
        Upper = new WenetEncoder<T>(engine, layers, configuration with { InputSize = d, Blocks = 4, AfterNorm = true }, initialization);
    }

    public WenetEncoderConfiguration Configuration { get; }
    public int Lookahead { get; }
    public WenetEncoder<T> Lower { get; }
    public NormedConv1DLayer<T> LookaheadConv1 { get; }
    public NormedConv1DLayer<T> LookaheadConv2 { get; }
    public NormedConv1DLayer<T> UpConv { get; }
    public WenetEncoder<T> Upper { get; }

    private Tensor<T> Pad(Tensor<T> x, int left, int right)
    {
        var parts = new List<Tensor<T>>();
        if (left > 0) parts.Add(new Tensor<T>(new[] { x.Shape[0], x.Shape[1], left }));
        parts.Add(x);
        if (right > 0) parts.Add(new Tensor<T>(new[] { x.Shape[0], x.Shape[1], right }));
        return parts.Count == 1 ? x : _engine.TensorConcatenate(parts.ToArray(), 2);
    }

    private Tensor<T> Channels(Tensor<T> x) => _engine.Reshape(_engine.TensorTranspose(x), new[] { 1, x.Shape[1], x.Shape[0] });

    private Tensor<T> Rows(Tensor<T> x) => _engine.TensorTranspose(_engine.Reshape(x, new[] { x.Shape[1], x.Shape[2] }));

    /// <summary>PreLookaheadLayer: the states padded by <see cref="Lookahead"/> frames on the right, a convolution reading
    /// them ahead, leaky ReLU, a causal 3-tap convolution, and the residual.</summary>
    internal Tensor<T> PreLookahead(Tensor<T> x)
    {
        var h = LookaheadConv1.Forward(Pad(Channels(x), 0, Lookahead));
        h = _engine.LeakyReLU(h, NumOps.FromDouble(0.01));
        h = LookaheadConv2.Forward(Pad(h, 2, 0));
        return _engine.TensorAdd(Rows(h), x);
    }

    /// <summary>Upsample1D: each frame repeated twice, padded by 4 frames on the left, a 5-tap convolution.</summary>
    internal Tensor<T> Upsample(Tensor<T> x)
    {
        int length = x.Shape[0], d = x.Shape[1];
        var index = new Tensor<int>(new[] { 2 * length });
        for (int i = 0; i < 2 * length; i++) index[i] = i / 2;
        var repeated = _engine.TensorIndexSelect(x, index, 0);
        return Rows(UpConv.Forward(Pad(Channels(repeated), 4, 0)));
    }

    /// <summary>The encoded states <c>[2L, OutputSize]</c> of inputs <c>[L, InputSize]</c>.</summary>
    public Tensor<T> Forward(Tensor<T> input, bool streaming, bool training, Random random)
    {
        int chunk = streaming ? Configuration.StaticChunkSize : 0;
        var (x, positions) = Lower.Embed(input, training, random);
        x = PreLookahead(x);
        x = Lower.RunBlocks(x, positions, chunk, training, random);
        var (y, upPositions) = Upper.Embed(Upsample(x), training, random);
        y = Upper.RunBlocks(y, upPositions, 2 * chunk, training, random);
        return (Upper.AfterNorm ?? throw new InvalidOperationException("The upper stack ends in its LayerNorm.")).Forward(y);
    }

    /// <summary>Loads an <c>UpsampleConformerEncoder</c> state: <c>embed</c>, <c>pre_lookahead_layer.conv1/2</c>,
    /// <c>encoders</c>, <c>up_layer.conv</c>, <c>up_embed</c>, <c>up_encoders</c> and <c>after_norm</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        int d = Configuration.OutputSize;
        void Conv(NormedConv1DLayer<T> conv, string name, int kernel) =>
            conv.LoadTorchWeights(read(name + ".weight", new[] { d, d, kernel }), null, read(name + ".bias", new[] { d }));
        Lower.LoadTorchWeights(read, prefix);
        Conv(LookaheadConv1, prefix + "pre_lookahead_layer.conv1", Lookahead + 1);
        Conv(LookaheadConv2, prefix + "pre_lookahead_layer.conv2", 3);
        Conv(UpConv, prefix + "up_layer.conv", 5);
        Upper.LoadTorchWeights(read, prefix, "up_embed", "up_encoders");
    }
}

/// <summary>The sizes of CosyVoice 2's language model (<c>Qwen2LM</c> over Qwen2.5-0.5B).</summary>
/// <param name="Qwen">The Qwen2 decoder (its token table reads the text; it has no head here).</param>
/// <param name="SpeechVocabulary">S3 speech tokens (6,561); the head adds end, a reserved id and fill.</param>
/// <param name="TextChunk">Bistream text tokens per chunk (<c>mix_ratio[0]</c>, 5).</param>
/// <param name="SpeechChunk">Bistream speech tokens per chunk (<c>mix_ratio[1]</c>, 15).</param>
internal sealed record CosyVoice2LmConfiguration(
    LlamaConfiguration Qwen, int SpeechVocabulary, int TextChunk = 5, int SpeechChunk = 15, double TopP = 0.8, int TopK = 25,
    int RepetitionWindow = 10, double RepetitionRatio = 0.1, double MinTokenTextRatio = 2, double MaxTokenTextRatio = 20);

/// <summary>
/// CosyVoice 2's text-to-speech-token language model (FunAudioLLM/CosyVoice <c>cosyvoice/llm/llm.py</c> <c>Qwen2LM</c>):
/// Qwen2 over <c>[sos, text…, task_id, speech…]</c>, its text embedded by Qwen's own table, its speech by a table of
/// <c>SpeechVocabulary + 3</c> rows, and a linear head over the speech tokens, end (<c>V</c>), a reserved id and fill
/// (<c>V + 2</c>). Unlike CosyVoice 1 it reads no x-vector.
/// </summary>
/// <remarks>
/// <para>Training (<c>prepare_lm_input_target</c>): the unistream sequence, or with probability ½ when the utterance has
/// more than <c>SpeechChunk / TextChunk</c> speech tokens per text token the bistream one — text and speech interleaved
/// in chunks of 5 and 15, each full text chunk's last position predicting the chunk's first speech token and each speech
/// chunk ending in fill, the remainder closed by task_id, the rest of the speech and end. The loss is the mean
/// cross-entropy over the scored positions.</para>
/// <para>Inference: the prompt's text precedes the text and its speech tokens follow task_id; end is forbidden until
/// <c>2·|text|</c> tokens and any of the three stop ids ends generation, drawn by repetition-aware sampling.</para>
/// </remarks>
internal sealed class CosyVoice2Lm<T>
{

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int Ignore = -1;
    private readonly IEngine _engine;

    public CosyVoice2Lm(IEngine engine, List<LayerBase<T>> layers, CosyVoice2LmConfiguration configuration, Random initialization)
    {
        _engine = engine;
        Configuration = configuration;
        var c = configuration;
        if (c.Qwen.VocabSize <= 0 || c.Qwen.IncludeHead)
            throw new ArgumentException("CosyVoice 2's Qwen2 reads text through its token table and has no head.", nameof(configuration));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T> { layers.Add(layer); return layer; }
        Func<double> normal = () => Math.Sqrt(-2.0 * Math.Log(1.0 - initialization.NextDouble())) * Math.Cos(2.0 * Math.PI * initialization.NextDouble());
        int d = c.Qwen.Hidden;
        LlmEmbedding = Add(new TiedEmbeddingLayer<T>(2, d));
        LlmEmbedding.Reinitialize(normal);
        Qwen = new LlamaNetwork<T>(engine, layers, c.Qwen);
        Qwen.InitializeLikeReference(initialization);
        double bound = 1.0 / Math.Sqrt(d);
        Head = Add(new DenseLayer<T>(c.SpeechVocabulary + 3, (IActivationFunction<T>)new IdentityActivation<T>()));
        TorchParameters.Linear(engine, Head, d, c.SpeechVocabulary + 3,
            Enumerable.Range(0, d * (c.SpeechVocabulary + 3)).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, c.SpeechVocabulary + 3).Select(_ => bound * (2 * initialization.NextDouble() - 1)).ToArray());
        SpeechEmbedding = Add(new TiedEmbeddingLayer<T>(c.SpeechVocabulary + 3, d));
        SpeechEmbedding.Reinitialize(normal);
    }

    public CosyVoice2LmConfiguration Configuration { get; }
    public TiedEmbeddingLayer<T> LlmEmbedding { get; }
    public LlamaNetwork<T> Qwen { get; }
    public DenseLayer<T> Head { get; }
    public TiedEmbeddingLayer<T> SpeechEmbedding { get; }

    /// <summary>The end-of-speech id.</summary>
    public int EndToken => Configuration.SpeechVocabulary;

    /// <summary>The bistream chunk's closing id.</summary>
    public int FillToken => Configuration.SpeechVocabulary + 2;

    private Tensor<T> TextEmbeddings(IReadOnlyList<int> text)
    {
        var ids = new int[1, text.Count];
        for (int i = 0; i < text.Count; i++) ids[0, i] = text[i];
        return _engine.Reshape(Qwen.Embed(ids), new[] { text.Count, Configuration.Qwen.Hidden });
    }

    private Tensor<T> Rows(TiedEmbeddingLayer<T> table, IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return table.Forward(tensor);
    }

    /// <summary>The head's logits <c>[L, SpeechVocabulary + 3]</c> of an input sequence <c>[L, Hidden]</c>.</summary>
    internal Tensor<T> LogitsOf(Tensor<T> input)
    {
        int length = input.Shape[0], d = Configuration.Qwen.Hidden;
        var (hidden, _) = Qwen.Forward(_engine.Reshape(input, new[] { 1, length, d }));
        return Head.Forward(_engine.Reshape(hidden, new[] { length, d }));
    }

    /// <summary>The training sequence and its targets (<c>prepare_lm_input_target</c> for one utterance): bistream with
    /// probability ½ when the utterance qualifies (<paramref name="random"/> draws it), unistream otherwise.</summary>
    internal (Tensor<T> Input, int[] Targets) TrainingSequence(IReadOnlyList<int> text, IReadOnlyList<int> speech, Random random) =>
        TrainingSequence(TextEmbeddings(text), speech, random);

    /// <summary>The speech embedding's rows for speech tokens (and the end and fill tokens after them).</summary>
    internal Tensor<T> SpeechRows(IReadOnlyList<int> ids) => Rows(SpeechEmbedding, ids);

    /// <summary>The LLM embedding's row for a special token: 0 sos, 1 the task id (MinMo's turn of speech).</summary>
    internal Tensor<T> SpecialRow(int id) => Rows(LlmEmbedding, new[] { id });

    /// <summary>
    /// The training sequence for text given as embeddings <c>[N, d]</c> (MinMo's semantic vectors): after
    /// <paramref name="prefix"/> (unscored rows after sos), bistream when <paramref name="bistream"/> says so (null: with
    /// probability ½ when the utterance qualifies, as CosyVoice 2 trains), unistream otherwise.
    /// </summary>
    internal (Tensor<T> Input, int[] Targets) TrainingSequence(Tensor<T> textEmbeddings, IReadOnlyList<int> speech, Random random,
        Tensor<T>? prefix = null, bool? bistream = null)
    {
        var c = Configuration;
        int n = textEmbeddings.Shape[0], d = textEmbeddings.Shape[1];
        Tensor<T> Slice(int start, int count) => _engine.TensorSlice(textEmbeddings, new[] { start, 0 }, new[] { count, d });
        var sos = Rows(LlmEmbedding, new[] { 0 });
        var task = Rows(LlmEmbedding, new[] { 1 });
        var parts = new List<Tensor<T>> { sos };
        var targets = new List<int> { Ignore };
        if (prefix is not null)
        {
            parts.Add(prefix);
            for (int i = 0; i < prefix.Shape[0]; i++) targets.Add(Ignore);
        }
        bool useBistream = bistream ?? (random.NextDouble() < 0.5 && (double)speech.Count / n > (double)c.SpeechChunk / c.TextChunk);
        if (useBistream)
        {
            int chunks = (int)Math.Ceiling((n + 1) / (double)c.TextChunk);
            for (int j = 0; j < chunks; j++)
            {
                int textStart = j * c.TextChunk, speechStart = j * c.SpeechChunk;
                int textCount = Math.Max(0, Math.Min(c.TextChunk, n - textStart));
                if (textCount == c.TextChunk)
                {
                    int speechCount = Math.Max(0, Math.Min(c.SpeechChunk, speech.Count - speechStart));
                    if (speechCount != c.SpeechChunk)
                        throw new InvalidOperationException("A full bistream text chunk needs a full speech chunk.");
                    for (int i = 0; i < c.TextChunk - 1; i++) targets.Add(Ignore);
                    for (int i = 0; i < c.SpeechChunk; i++) targets.Add(speech[speechStart + i]);
                    targets.Add(FillToken);
                    parts.Add(Slice(textStart, c.TextChunk));
                    parts.Add(Rows(SpeechEmbedding, speech.Skip(speechStart).Take(c.SpeechChunk).ToList()));
                }
                else
                {
                    var rest = speech.Skip(speechStart).ToList();
                    for (int i = 0; i < textCount; i++) targets.Add(Ignore);
                    targets.AddRange(rest);
                    targets.Add(EndToken);
                    if (textCount > 0) parts.Add(Slice(textStart, textCount));
                    parts.Add(task);
                    if (rest.Count > 0) parts.Add(Rows(SpeechEmbedding, rest));
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++) targets.Add(Ignore);
            targets.AddRange(speech);
            targets.Add(EndToken);
            if (n > 0) parts.Add(textEmbeddings);
            parts.Add(task);
            if (speech.Count > 0) parts.Add(Rows(SpeechEmbedding, speech));
        }
        return (_engine.TensorConcatenate(parts.ToArray(), 0), targets.ToArray());
    }

    /// <summary>The training loss: the mean cross-entropy of the scored positions of <see cref="TrainingSequence"/>.</summary>
    public Tensor<T> Loss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Random random) =>
        Loss(TrainingSequence(text, speech, random));

    /// <summary>The mean cross-entropy of the scored positions of a training sequence.</summary>
    internal Tensor<T> Loss((Tensor<T> Input, int[] Targets) sequence)
    {
        var (input, targets) = sequence;
        var logp = _engine.TensorLogSoftmax(LogitsOf(input), axis: -1);
        int scored = targets.Count(t => t != Ignore);
        var picks = new Tensor<T>(logp._shape);
        for (int p = 0; p < targets.Length; p++)
            if (targets[p] != Ignore) picks[p, targets[p]] = NumOps.FromDouble(-1.0 / scored);
        return _engine.ReduceSum(_engine.TensorMultiply(logp, picks), new[] { 0, 1 }, false);
    }

    /// <summary>Generates speech tokens (<c>Qwen2LM.inference</c>) for <paramref name="text"/> after an optional prompt.</summary>
    public List<int> Generate(IReadOnlyList<int> text, IReadOnlyList<int> promptText, IReadOnlyList<int> promptSpeech, Random random)
    {
        var c = Configuration;
        var allText = promptText.Concat(text).ToList();
        int minLength = (int)(text.Count * c.MinTokenTextRatio), maxLength = (int)(text.Count * c.MaxTokenTextRatio);
        var prefix = new List<Tensor<T>> { Rows(LlmEmbedding, new[] { 0 }), TextEmbeddings(allText), Rows(LlmEmbedding, new[] { 1 }) };
        if (promptSpeech.Count > 0) prefix.Add(Rows(SpeechEmbedding, promptSpeech));
        var input = _engine.TensorConcatenate(prefix.ToArray(), 0);
        var generated = new List<int>();
        for (int i = 0; i < maxLength; i++)
        {
            var logp = _engine.TensorLogSoftmax(LogitsOf(input), axis: -1);
            int last = logp.Shape[0] - 1, vocabulary = logp.Shape[1];
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = NumOps.ToDouble(logp[last, v]);
            if (i < minLength) row[EndToken] = double.NegativeInfinity;
            int next = CosyVoiceLm<T>.SampleRepetitionAware(row, generated, c.TopP, c.TopK, c.RepetitionWindow, c.RepetitionRatio, random);
            if (next >= c.SpeechVocabulary) break;
            generated.Add(next);
            input = _engine.TensorConcatenate(new[] { input, Rows(SpeechEmbedding, new[] { next }) }, 0);
        }
        return generated;
    }

    /// <summary>Loads <c>llm.pt</c>'s state: <c>llm_embedding</c>, <c>llm.model.model.*</c> (Qwen2), <c>llm_decoder</c>
    /// and <c>speech_embedding</c>.</summary>
    public void LoadTorchWeights(Func<string, int[], double[]> read, string prefix)
    {
        var c = Configuration;
        int d = c.Qwen.Hidden, outputs = c.SpeechVocabulary + 3;
        LlmEmbedding.LoadTable(read(prefix + "llm_embedding.weight", new[] { 2, d }));
        Qwen.LoadTorchWeights(read, prefix + "llm.model.model.");
        TorchParameters.Linear(_engine, Head, d, outputs, read(prefix + "llm_decoder.weight", new[] { outputs, d }), read(prefix + "llm_decoder.bias", new[] { outputs }));
        SpeechEmbedding.LoadTable(read(prefix + "speech_embedding.weight", new[] { outputs, d }));
    }
}
