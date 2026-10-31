using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// MinMo's networks and the paths through them (Chen et al. 2025 §3): the voice encoder and input projector turn speech
/// into the LLM's embeddings inside a ChatML prompt (FunASR LLM-ASR's layout); Qwen2.5 writes the answer; the output
/// projector maps each position's embedding concatenated with its last hidden state to a semantic vector; CosyVoice 2's
/// language model reads the user turn's vectors as a query and then the answer's five at a time, each followed by fifteen
/// speech tokens; its flow and vocoder make the waveform; the full duplex predictor reads the hidden states.
/// </summary>
internal sealed class MinMoCore<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private static readonly Lazy<QwenTokenizer> QwenText = new(() => new QwenTokenizer(Array.Empty<string>()));
    private const int EncoderRate = 16000;
    private readonly IEngine _engine;

    public MinMoCore(IEngine engine, MinMoOptions options, List<LayerBase<T>> encoderLayers, List<LayerBase<T>> inputProjectorLayers,
        List<LayerBase<T>> llmLayers, List<LayerBase<T>> outputProjectorLayers, List<LayerBase<T>> duplexLayers, int seed)
    {
        _engine = engine;
        Options = options;
        var o = options;
        if (o.HiddenDim % o.NumHeads != 0 || o.NumHeads % o.KeyValueHeads != 0)
            throw new ArgumentException("The LLM's width must divide into its heads, and its heads into the key-value heads.");
        if (o.VocabSize < QwenText.Value.VocabularySize)
            throw new ArgumentException($"VocabSize ({o.VocabSize}) must cover Qwen2.5's {QwenText.Value.VocabularySize} token ids.");
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        int encoderWidth;
        if (o.Encoder == MinMoVoiceEncoder.SenseVoiceLarge)
        {
            LargeEncoder = new OpenAiWhisperNetwork<T>(engine, encoderLayers,
                new WhisperDimensions(o.LargeEncoderMels, 1500, o.LargeEncoderWidth, o.LargeEncoderHeads, o.LargeEncoderLayers, 1, 1, 4, 1, 0)
                {
                    EncoderOnly = true, VariableAudioLength = true,
                }, random);
            LargeEncoder.InitializeLikeReference(random);
            encoderWidth = o.LargeEncoderWidth;
        }
        else
        {
            SmallEncoder = new SanmEncoder<T>(engine, encoderLayers, new SanmEncoderConfiguration(o.SmallEncoderMels, 7, 6, o.SmallEncoderWidth,
                o.SmallEncoderHeads, o.SmallEncoderFeedForward, o.SmallEncoderBlocks, o.SmallEncoderTpBlocks), random);
            encoderWidth = o.SmallEncoderWidth;
        }
        InputProjector = new MinMoInputProjector<T>(engine, inputProjectorLayers, encoderWidth, o.ProjectorDownsample, o.ProjectorFeedForward,
            o.HiddenDim, o.ProjectorLayers, o.ProjectorHeads, random);
        Llm = new LlamaNetwork<T>(engine, llmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.NumHeads, KeyValueHeads = o.KeyValueHeads,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, AttentionBias = true,
            LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
        });
        Llm.InitializeLikeReference(random);
        Voice = new CosyVoice2<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
            o.VoiceDecoder);
        int voiceWidth = o.VoiceDecoder.LLMDim;
        OutputProjector = new DenseLayer<T>(voiceWidth, (IActivationFunction<T>)new AiDotNet.ActivationFunctions.IdentityActivation<T>());
        double bound = 1.0 / Math.Sqrt(2 * o.HiddenDim);
        TorchParameters.Linear(engine, OutputProjector, 2 * o.HiddenDim, voiceWidth,
            Enumerable.Range(0, 2 * o.HiddenDim * voiceWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray(),
            Enumerable.Range(0, voiceWidth).Select(_ => bound * (2 * random.NextDouble() - 1)).ToArray());
        outputProjectorLayers.Add(OutputProjector);
        Duplex = new MinMoDuplexPredictor<T>(engine, duplexLayers, o.HiddenDim, o.DuplexWidth, o.DuplexHeads, o.DuplexFeedForward, random);
    }

    public MinMoOptions Options { get; }
    public OpenAiWhisperNetwork<T>? LargeEncoder { get; }
    public SanmEncoder<T>? SmallEncoder { get; }
    public MinMoInputProjector<T> InputProjector { get; }
    public LlamaNetwork<T> Llm { get; }
    public DenseLayer<T> OutputProjector { get; }
    public CosyVoice2<T> Voice { get; }
    public MinMoDuplexPredictor<T> Duplex { get; }

    public CosyVoice2Lm<T> VoiceLm => Voice.LanguageModel ?? throw new InvalidOperationException("The voice decoder exists in native mode only.");

    /// <summary>The LoRA branches' layers on the LLM's query and value projections.</summary>
    public IEnumerable<LayerBase<T>> LoraLayers => Llm.Layers
        .SelectMany(l => new[] { l.QueryLora, l.ValueLora })
        .Where(a => a is not null)
        .SelectMany(a => new LayerBase<T>[] { a!.A, a.B });

    public QwenTokenizer Tokenizer => QwenText.Value;

    private int ImEnd => Tokenizer.AddedTokenId("<|im_end|>");
    private int EndOfText => Tokenizer.AddedTokenId("<|endoftext|>");

    // ---------------------------------------------------------------- speech in

    /// <summary>The LLM-width embeddings <c>[n, width]</c> of speech at <paramref name="sampleRate"/> (resampled to 16 kHz
    /// for the encoder).</summary>
    public Tensor<T> SpeechEmbeddings(Tensor<T> audio, int sampleRate)
    {
        var wave = _engine.Reshape(audio, new[] { audio.Length });
        if (sampleRate != EncoderRate) wave = AudioHelper<T>.ResampleBandLimited(_engine, wave, sampleRate, EncoderRate);
        Tensor<T> encoded;
        if (LargeEncoder is not null)
        {
            var samples = new double[wave.Length];
            for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(wave.GetFlat(i));
            var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(samples, Options.LargeEncoderMels);
            int mels = mel.GetLength(0), frames = Math.Min(mel.GetLength(1), 3000);
            if (frames < 2) throw new ArgumentException("The recording is too short for the voice encoder.", nameof(audio));
            var melTensor = new Tensor<T>(new[] { mels, frames });
            for (int m = 0; m < mels; m++)
                for (int f = 0; f < frames; f++) melTensor[m, f] = Ops.FromDouble(mel[m, f]);
            var states = LargeEncoder.Encode(melTensor);
            encoded = _engine.Reshape(states, new[] { states.Shape[1], states.Shape[2] });
        }
        else encoded = SmallEncoder!.Encode(wave);
        return InputProjector.Forward(encoded);
    }

    // ---------------------------------------------------------------- the prompt

    /// <summary>A prompt laid out for the LLM: its embeddings <c>[L, width]</c>, its ids (−1 at speech positions), and the
    /// span of the user's message (the query the voice decoder reads).</summary>
    internal sealed record Prompt(Tensor<T> Embeddings, int[] Ids, int UserStart, int UserEnd);

    /// <summary>
    /// The ChatML prompt (FunASR LLM-ASR's <c>data_template</c>): <c>&lt;|im_start|&gt;system\n{system}&lt;|im_end|&gt;\n
    /// &lt;|im_start|&gt;user\n{instruction} [speech]&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</c>, the speech span
    /// replaced by its embeddings.
    /// </summary>
    public Prompt BuildPrompt(string instruction, Tensor<T>? speech, int sampleRate)
    {
        var head = Tokenizer.Encode($"<|im_start|>system\n{Options.SystemPrompt}<|im_end|>\n<|im_start|>user\n");
        var user = Tokenizer.Encode(speech is null ? instruction : instruction + " ");
        var tail = Tokenizer.Encode("<|im_end|>\n<|im_start|>assistant\n");
        var speechEmbeddings = speech is null ? null : SpeechEmbeddings(speech, sampleRate);
        int speechCount = speechEmbeddings?.Shape[0] ?? 0;
        var ids = head.Concat(user).Concat(Enumerable.Repeat(-1, speechCount)).Concat(tail).ToArray();
        var parts = new List<Tensor<T>> { TokenEmbeddings(head) };
        if (user.Length > 0) parts.Add(TokenEmbeddings(user));
        if (speechEmbeddings is not null) parts.Add(speechEmbeddings);
        parts.Add(TokenEmbeddings(tail));
        return new Prompt(_engine.TensorConcatenate(parts.ToArray(), 0), ids, head.Length, head.Length + user.Length + speechCount);
    }

    public Tensor<T> TokenEmbeddings(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        return _engine.Reshape(Llm.Embed(grid), new[] { ids.Count, Options.HiddenDim });
    }

    // ---------------------------------------------------------------- text out

    /// <summary>The answer's tokens (without the closing <c>&lt;|im_end|&gt;</c>), sampled with Qwen2.5-Instruct's settings,
    /// and the LLM's last hidden states of the whole sequence <c>[prompt + answer, width]</c>.</summary>
    public (int[] Answer, Tensor<T> Hidden, Tensor<T> Embeddings) Generate(Prompt prompt, Random random)
    {
        var o = Options;
        var cache = new LlamaCache<T>();
        var embeddings = new List<Tensor<T>> { prompt.Embeddings };
        var hidden = new List<Tensor<T>>();
        var answer = new List<int>();
        var input = prompt.Embeddings;
        var seen = new HashSet<int>(prompt.Ids.Where(i => i >= 0));
        bool pending = false;
        for (int step = 0; step < o.MaxNewTokens; step++)
        {
            pending = false;
            var (h, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], o.HiddenDim }), null, null, cache, lastLogitsOnly: true);
            hidden.Add(_engine.Reshape(h, new[] { input.Shape[0], o.HiddenDim }));
            var row = new double[o.VocabSize];
            for (int v = 0; v < row.Length; v++) row[v] = Ops.ToDouble(logits!.GetFlat(v));
            int next = Sample(row, seen, random);
            if (next == ImEnd || next == EndOfText) break;
            answer.Add(next);
            seen.Add(next);
            input = TokenEmbeddings(new[] { next });
            embeddings.Add(input);
            pending = true;
        }
        // The last answer token's own hidden state, when the length limit stopped generation before it was read.
        if (pending)
        {
            var (h, _) = Llm.Forward(_engine.Reshape(input, new[] { 1, 1, o.HiddenDim }), null, null, cache, lastLogitsOnly: true);
            hidden.Add(_engine.Reshape(h, new[] { 1, o.HiddenDim }));
        }
        return (answer.ToArray(), _engine.TensorConcatenate(hidden.ToArray(), 0), _engine.TensorConcatenate(embeddings.ToArray(), 0));
    }

    // Qwen2.5-Instruct's generation_config: repetition penalty on every id already present, temperature, top-k, top-p.
    private int Sample(double[] logits, HashSet<int> seen, Random random)
    {
        var o = Options;
        foreach (int id in seen)
            if (id < logits.Length) logits[id] = logits[id] > 0 ? logits[id] / o.RepetitionPenalty : logits[id] * o.RepetitionPenalty;
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        if (o.Temperature <= 0) return best;
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ThenBy(i => i)
            .Take(o.TopK > 0 ? Math.Min(o.TopK, logits.Length) : logits.Length).ToArray();
        double max = logits[order[0]] / o.Temperature;
        var weights = order.Select(i => Math.Exp(logits[i] / o.Temperature - max)).ToArray();
        double total = weights.Sum(), cumulative = 0;
        int keep = 0;
        // top-p keeps the smallest prefix whose mass reaches p (always the first).
        while (keep < order.Length)
        {
            cumulative += weights[keep] / total;
            keep++;
            if (cumulative >= o.TopP) break;
        }
        double kept = weights.Take(keep).Sum(), draw = random.NextDouble() * kept;
        for (int i = 0; i < keep; i++)
        {
            draw -= weights[i];
            if (draw <= 0) return order[i];
        }
        return order[keep - 1];
    }

    public string TextOf(IEnumerable<int> ids) => Tokenizer.Decode(ids);

    // ---------------------------------------------------------------- speech out

    /// <summary>The semantic vectors <c>[n, voice width]</c>: each position's input embedding and last hidden state
    /// concatenated, through the output projector.</summary>
    public Tensor<T> SemanticVectors(Tensor<T> embeddings, Tensor<T> hidden) =>
        OutputProjector.Forward(_engine.TensorConcatenate(new[] { embeddings, hidden }, 1));

    private Tensor<T> Rows(Tensor<T> x, int start, int count) => _engine.TensorSlice(x, new[] { start, 0 }, new[] { count, x.Shape[1] });

    /// <summary>
    /// The voice token LM's generation (§3.2): after sos and the query vectors, each full chunk of five answer vectors is
    /// followed by fifteen sampled speech tokens; the rest of the vectors and the turn-of-speech token follow, and tokens
    /// are sampled until the end of speech (within CosyVoice 2's token-to-text ratios).
    /// </summary>
    public int[] SpeechTokens(Tensor<T> query, Tensor<T> answer, Random random)
    {
        var lm = VoiceLm;
        var c = lm.Configuration;
        int n = answer.Shape[0], width = c.Qwen.Hidden;
        // The rows not yet read; the cache holds the keys and values of those already read.
        var pending = new List<Tensor<T>> { lm.SpecialRow(0), query };
        var cache = new LlamaCache<T>();
        var tokens = new List<int>();
        int Next(bool allowEnd)
        {
            var rows = pending.Count == 1 ? pending[0] : _engine.TensorConcatenate(pending.ToArray(), 0);
            pending.Clear();
            int length = rows.Shape[0];
            var (hidden, _) = lm.Qwen.Forward(_engine.Reshape(rows, new[] { 1, length, width }), null, null, cache, lastLogitsOnly: true);
            var last = _engine.TensorSlice(_engine.Reshape(hidden, new[] { length, width }), new[] { length - 1, 0 }, new[] { 1, width });
            var logp = _engine.TensorLogSoftmax(lm.Head.Forward(last), axis: -1);
            int vocabulary = logp.Shape[1];
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = Ops.ToDouble(logp[0, v]);
            for (int v = c.SpeechVocabulary + (allowEnd ? 1 : 0); v < vocabulary; v++) row[v] = double.NegativeInfinity;
            return CosyVoiceLm<T>.SampleRepetitionAware(row, tokens, c.TopP, c.TopK, c.RepetitionWindow, c.RepetitionRatio, random);
        }
        void Append(int token)
        {
            tokens.Add(token);
            pending.Add(lm.SpeechRows(new[] { token }));
        }
        int chunk = 0;
        for (; (chunk + 1) * c.TextChunk <= n; chunk++)
        {
            pending.Add(Rows(answer, chunk * c.TextChunk, c.TextChunk));
            for (int i = 0; i < c.SpeechChunk; i++) Append(Next(allowEnd: false));
        }
        int rest = n - chunk * c.TextChunk;
        if (rest > 0) pending.Add(Rows(answer, chunk * c.TextChunk, rest));
        pending.Add(lm.SpecialRow(1));
        int minimum = (int)(n * c.MinTokenTextRatio), maximum = Math.Max(minimum + 1, (int)(n * c.MaxTokenTextRatio));
        while (tokens.Count < maximum)
        {
            int token = Next(allowEnd: tokens.Count >= minimum);
            if (token >= c.SpeechVocabulary) break;
            Append(token);
        }
        return tokens.ToArray();
    }

    /// <summary>The waveform of speech tokens through CosyVoice 2's flow and vocoder (under the voice's
    /// <see cref="TtsModelBase{T}.Voice"/>).</summary>
    public Tensor<T> Waveform(IReadOnlyList<int> tokens, Random random)
    {
        if (tokens.Count == 0) return new Tensor<T>(new[] { 0 });
        return Voice.MelToAudio(Voice.TokensToMel(tokens, random), random);
    }

    // ---------------------------------------------------------------- objectives

    /// <summary>The speech-to-text loss: the cross-entropy of <c>{answer}&lt;|im_end|&gt;</c> after the prompt.</summary>
    public Tensor<T> SpeechToTextLoss(Prompt prompt, string answer)
    {
        var target = Tokenizer.Encode(answer).Concat(new[] { ImEnd }).ToArray();
        var input = _engine.TensorConcatenate(new[] { prompt.Embeddings, TokenEmbeddings(target) }, 0);
        int length = input.Shape[0];
        var (_, logits) = Llm.Forward(_engine.Reshape(input, new[] { 1, length, Options.HiddenDim }));
        var labels = new int[1, length];
        for (int i = 0; i < prompt.Ids.Length; i++) labels[0, i] = -100;
        for (int i = 0; i < target.Length; i++) labels[0, prompt.Ids.Length + i] = target[i];
        return Llm.CausalLmLoss(logits!, labels);
    }

    /// <summary>
    /// The voice decoder's loss (§3.4, stages 2 and 3): the LLM, frozen, reads the prompt and the answer; the output
    /// projector makes the user turn's query vectors and the answer's semantic vectors; the voice token LM is trained on
    /// them interleaved 5:15 with the answer recording's speech tokens (CosyVoice 2's S3 tokenizer at 25 Hz), unistream
    /// when the recording is too short for the ratio.
    /// </summary>
    public Tensor<T> VoiceLoss(Prompt prompt, string answer, Tensor<T> answerRecording, int sampleRate, Random random)
    {
        var target = Tokenizer.Encode(answer);
        if (target.Length == 0) throw new ArgumentException("The answer has no tokens.", nameof(answer));
        var answerEmbeddings = TokenEmbeddings(target);
        var input = _engine.TensorConcatenate(new[] { prompt.Embeddings, answerEmbeddings }, 0);
        Tensor<T> hidden;
        Tensor<T> embeddings;
        using (new NoGradScope<T>())
        {
            var (h, _) = Llm.Forward(_engine.Reshape(input, new[] { 1, input.Shape[0], Options.HiddenDim }), null, null, null, lastLogitsOnly: true);
            hidden = Detached(_engine.Reshape(h, new[] { input.Shape[0], Options.HiddenDim }));
            embeddings = Detached(input);
        }
        int[] speech;
        using (new NoGradScope<T>()) speech = Voice.SpeechTokens(answerRecording, sampleRate);
        var semantic = SemanticVectors(embeddings, hidden);
        var query = Rows(semantic, prompt.UserStart, prompt.UserEnd - prompt.UserStart);
        var answerVectors = Rows(semantic, prompt.Ids.Length, target.Length);
        var c = VoiceLm.Configuration;
        bool bistream = (double)speech.Length / target.Length > (double)c.SpeechChunk / c.TextChunk;
        return VoiceLm.Loss(VoiceLm.TrainingSequence(answerVectors, speech, random, query, bistream));
    }

    /// <summary>The duplex predictor's loss: the cross-entropy of the action labels (−1 unlabelled) over the speech
    /// positions of the prompt, the LLM frozen.</summary>
    public Tensor<T> DuplexLoss(Prompt prompt, IReadOnlyList<int> labels)
    {
        Tensor<T> hidden;
        using (new NoGradScope<T>())
        {
            var (h, _) = Llm.Forward(_engine.Reshape(prompt.Embeddings, new[] { 1, prompt.Ids.Length, Options.HiddenDim }), null, null, null, lastLogitsOnly: true);
            hidden = Detached(_engine.Reshape(h, new[] { prompt.Ids.Length, Options.HiddenDim }));
        }
        var logits = Duplex.Forward(hidden);
        var speechPositions = Enumerable.Range(0, prompt.Ids.Length).Where(i => prompt.Ids[i] == -1).ToArray();
        if (labels.Count != speechPositions.Length)
            throw new ArgumentException($"One label per speech position: {speechPositions.Length} expected, {labels.Count} given.", nameof(labels));
        int actions = logits.Shape[1], scored = labels.Count(l => l >= 0);
        if (scored == 0) throw new ArgumentException("No position is labelled.", nameof(labels));
        var logp = _engine.TensorLogSoftmax(logits, axis: -1);
        var picks = new Tensor<T>(logp._shape);
        for (int k = 0; k < labels.Count; k++)
        {
            if (labels[k] < 0) continue;
            if (labels[k] >= actions) throw new ArgumentOutOfRangeException(nameof(labels), $"Labels lie in [0, {actions}).");
            picks[speechPositions[k], labels[k]] = Ops.FromDouble(-1.0 / scored);
        }
        return _engine.ReduceSum(_engine.TensorMultiply(logp, picks), new[] { 0, 1 }, false);
    }

    /// <summary>The duplex predictor's action at each speech position of the prompt.</summary>
    public MinMoDuplexAction[] DuplexActions(Prompt prompt)
    {
        var (h, _) = Llm.Forward(_engine.Reshape(prompt.Embeddings, new[] { 1, prompt.Ids.Length, Options.HiddenDim }), null, null, null, lastLogitsOnly: true);
        var logits = Duplex.Forward(_engine.Reshape(h, new[] { prompt.Ids.Length, Options.HiddenDim }));
        var actions = new List<MinMoDuplexAction>();
        for (int i = 0; i < prompt.Ids.Length; i++)
        {
            if (prompt.Ids[i] != -1) continue;
            int best = 0;
            for (int a = 1; a < logits.Shape[1]; a++) if (Ops.ToDouble(logits[i, a]) > Ops.ToDouble(logits[i, best])) best = a;
            actions.Add((MinMoDuplexAction)best);
        }
        return actions.ToArray();
    }

    private static Tensor<T> Detached(Tensor<T> x) => new(x.Shape.ToArray(), x.ToVector());
}
