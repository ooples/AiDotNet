using AiDotNet.Audio.Features;
using AiDotNet.Audio.Foundations;
using AiDotNet.Audio.Speaker;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A speaker Step-Audio's TTS speaks as (the release's <c>speakers_info</c> entry): the name, the prompt
/// recording's transcript and dual-codebook ids, its 22.05 kHz mel <c>[frames, 80]</c> and CAM++ x-vector.</summary>
public sealed class StepAudioSpeaker<T>
{
    public required string Name { get; init; }
    public required string PromptText { get; init; }
    public required int[] PromptCode { get; init; }
    public required Tensor<T> PromptMel { get; init; }
    public required Tensor<T> Embedding { get; init; }
}

/// <summary>
/// One CosyVoice-300M-25Hz decoder (Step-Audio's fork): the flow from dual-codebook tokens (1,024 linguistic then the
/// semantic codes, 5,120 ids, at 41.667 Hz) to an
/// 80-band mel with the prompt's mel as condition and the x-vector, and HiFT at 22.05 kHz.
/// </summary>
internal sealed class StepAudioDecoder<T>
{
    public StepAudioDecoder(IEngine engine, StepAudioOptions o, List<LayerBase<T>> flowLayers, List<LayerBase<T>> hiftLayers, Random random)
    {
        Flow = new CosyVoiceFlow<T>(engine, flowLayers, new CosyVoiceFlowConfiguration(
            VocabSize: o.LinguisticCodes + o.SemanticCodes, InputSize: o.FlowWidth, OutputSize: o.MelChannelCount, SpeakerDim: o.SpeakerEmbeddingDimension,
            InputFrameRate: 41.667, SampleRate: o.SampleRate, HopSize: 256,
            Encoder: new WenetEncoderConfiguration(o.FlowWidth, o.FlowWidth, o.FlowHeads, o.FlowFeedForward, o.FlowEncoderLayers,
                WenetInputLayer.Linear, Swish: true, Conformer: true, Dropout: 0.01, PositionalDropout: 0.01, AttentionDropout: 0.01),
            RegulatorStages: 4, DecoderChannels: (int[])o.DecoderChannels.Clone(), DecoderHeads: o.DecoderHeads, DecoderHeadDim: o.DecoderHeadDim,
            DecoderBlocksPerLevel: o.DecoderBlocks, DecoderMidBlocks: o.DecoderMidBlocks, SigmaMin: 1e-6, TrainingCfgRate: 0.2,
            InferenceCfgRate: 0.7, Timesteps: o.FlowTimesteps, Regulator: CosyVoiceRegulator.LinearWhole), random);
        int stages = o.HiftUpsampleRates.Length;
        Hift = new HiftGenerator<T>(engine, hiftLayers, new HiftConfiguration(InChannels: o.MelChannelCount, BaseChannels: o.HiftBaseChannels,
            Harmonics: o.HiftHarmonics, SampleRate: o.SampleRate, NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 10,
            UpsampleRates: (int[])o.HiftUpsampleRates.Clone(), UpsampleKernels: (int[])o.HiftUpsampleKernels.Clone(), FftSize: 16, HopSize: 4,
            ResblockKernels: new[] { 3, 7, 11 }, ResblockDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            SourceKernels: (int[])o.HiftSourceKernels.Clone(), SourceDilations: Enumerable.Range(0, stages).Select(_ => new[] { 1, 3, 5 }).ToArray(),
            LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: o.HiftBaseChannels), random);
    }

    public CosyVoiceFlow<T> Flow { get; }
    public HiftGenerator<T> Hift { get; }
}

/// <summary>
/// Step-Audio's networks and pipelines (Huang et al. 2025; stepfun-ai/Step-Audio): the dual-codebook tokenizer, the
/// Step-1 chat and TTS models, and the CosyVoice decoders.
/// </summary>
internal sealed class StepAudioCore<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private const int TokenizerRate = 16000, ChunkSamples = 30 * 16000;
    private readonly IEngine _engine;
    private StepAudioTokens? _tokens;
    private KaldiFbank<T>? _speakerFbank;
    private DifferentiableMel<T>? _mel;

    public StepAudioCore(IEngine engine, StepAudioOptions options, StepAudioLayers<T> layers, int seed)
    {
        _engine = engine;
        Options = options;
        var o = options;
        if (o.LinguisticCodes != StepAudioTokens.LinguisticCodes)
            throw new ArgumentException($"The release's token layout puts the semantic codes after {StepAudioTokens.LinguisticCodes} linguistic ones.");
        if (StepAudioTokens.AudioBase + o.LinguisticCodes + o.SemanticCodes > o.VocabSize)
            throw new ArgumentException("The audio tokens must fit in the vocabulary.");
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        LlamaConfiguration Step1(int width, int heads, int groups, int count, int intermediate) => new()
        {
            VocabSize = o.VocabSize, Hidden = width, Layers = count, Heads = heads, KeyValueHeads = groups, Intermediate = intermediate,
            RmsEpsilon = o.RmsEpsilon, SqrtAlibi = true,
        };
        if (o.ChatLayers > 0)
        {
            Chat = new LlamaNetwork<T>(engine, layers.Chat, Step1(o.HiddenDim, o.NumHeads, o.ChatGroups, o.ChatLayers, o.ChatIntermediate));
            Chat.InitializeLikeReference(random);
        }
        Tts = new LlamaNetwork<T>(engine, layers.Tts, Step1(o.TtsWidth, o.TtsHeads, o.TtsGroups, o.TtsLayers, o.TtsIntermediate));
        Tts.InitializeLikeReference(random);
        Linguistic = new StreamingParaformerEncoder<T>(engine, layers.Tokenizer, new SanmEncoderConfiguration(o.ParaformerMels, 7, 6,
            o.ParaformerWidth, o.ParaformerHeads, o.ParaformerFeedForward, o.ParaformerBlocks, 0, 11, 1e-12), random)
        {
            Dither = o.LinguisticDither,
        };
        Centroids = TorchParameters.Gaussian<T>(random, o.LinguisticCodes * o.ParaformerWidth, 1.0).ToArray().Select(v => Ops.ToDouble(v)).ToArray();
        Semantic = new S3Tokenizer<T>(engine, layers.Tokenizer, new S3TokenizerConfiguration(State: o.SemanticWidth, Heads: o.SemanticHeads,
            Layers: o.SemanticLayers, CodebookSize: o.SemanticCodes), random);
        Speaker = new CamPlusPlusEncoder<T>(engine, layers.Tokenizer, new CamPlusPlusConfiguration(EmbeddingSize: o.SpeakerEmbeddingDimension,
            GrowthRate: o.SpeakerEncoderGrowthRate, InitialChannels: o.SpeakerEncoderChannels), random);
        Speaker.SetTrainingMode(false);
        Common = new StepAudioDecoder<T>(engine, o, layers.Decoder, layers.Decoder, random);
        if (o.IncludeMusicDecoder) Music = new StepAudioDecoder<T>(engine, o, layers.MusicDecoder, layers.MusicDecoder, random);
    }

    public StepAudioOptions Options { get; }
    public LlamaNetwork<T>? Chat { get; }
    public LlamaNetwork<T> Tts { get; }
    public StreamingParaformerEncoder<T> Linguistic { get; }

    /// <summary>The linguistic tokenizer's k-means centroids <c>[codes · width]</c> (<c>linguistic_tokenizer.npy</c>).</summary>
    public double[] Centroids { get; set; }

    public S3Tokenizer<T> Semantic { get; }
    public CamPlusPlusEncoder<T> Speaker { get; }
    public StepAudioDecoder<T> Common { get; }
    public StepAudioDecoder<T>? Music { get; }

    public LlamaNetwork<T> ChatModel => Chat ?? throw new InvalidOperationException("The chat model was left out (ChatLayers = 0).");

    public StepAudioTokens Tokens
    {
        get
        {
            if (_tokens is not null) return _tokens;
            string path = Options.TokenizerPath ?? throw new InvalidOperationException("Text needs Step-1's tokenizer: set StepAudioOptions.TokenizerPath.");
            if (!File.Exists(path)) throw new FileNotFoundException($"Tokenizer not found: {path}", path);
            var model = new SentencePieceModel(File.ReadAllBytes(path));
            if (model.Count > Options.VocabSize) throw new InvalidOperationException($"The tokenizer has {model.Count} pieces; the vocabulary is {Options.VocabSize}.");
            return _tokens = new StepAudioTokens(model);
        }
    }

    public void ResetTokenizer() => _tokens = null;

    // ---------------------------------------------------------------- audio to tokens

    private static double[] Samples(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(audio.GetFlat(i));
        return samples;
    }

    private Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    private double[] Resample(Tensor<T> audio, int from, int to) =>
        from == to ? Samples(audio) : Samples(AudioHelper<T>.ResampleBandLimited(_engine, _engine.Reshape(audio, new[] { audio.Length }), from, to));

    /// <summary>
    /// The dual-codebook tokens of a recording (tokenizer.py's <c>wav2token</c> as <c>__call__</c> runs it: resampled to
    /// 16 kHz, its peak scaled to 0.999 — not below 0.01 — and not trimmed): the linguistic codes, the semantic codes, and
    /// their interleaved ids (65536 + code, two linguistic then three semantic).
    /// </summary>
    public (int[] Ids, int[] Linguistic, int[] Semantic) AudioTokens(Tensor<T> audio, int sampleRate)
    {
        var x = Resample(audio, sampleRate, TokenizerRate);
        double peak = x.Length == 0 ? 0 : x.Max(Math.Abs);
        double gain = 0.999 / Math.Max(peak, 0.01);
        for (int i = 0; i < x.Length; i++) x[i] *= gain;
        int[] linguistic, semantic;
        using (new NoGradScope<T>())
        {
            linguistic = LinguisticCodes(x);
            semantic = SemanticCodes(x);
        }
        return (StepAudioTokens.AudioIds(linguistic, semantic), linguistic, semantic);
    }

    /// <summary>The linguistic codes of 16 kHz audio: the streaming Paraformer encoder's frames, each the nearest
    /// centroid's index (Euclidean, <c>torch.cdist</c>).</summary>
    public int[] LinguisticCodes(double[] audio)
    {
        var frames = Linguistic.Encode(audio);
        int count = frames.Shape[0], width = frames.Shape[1], codes = Centroids.Length / width;
        var result = new int[count];
        for (int t = 0; t < count; t++)
        {
            double best = double.PositiveInfinity;
            for (int k = 0; k < codes; k++)
            {
                double d = 0;
                for (int j = 0; j < width; j++)
                {
                    double diff = Ops.ToDouble(frames[t, j]) - Centroids[k * width + j];
                    d += diff * diff;
                }
                if (d < best) { best = d; result[t] = k; }
            }
        }
        return result;
    }

    /// <summary>The semantic codes of 16 kHz audio (tokenizer.py's <c>get_vq06_code</c>): S3 v1 over 30 s chunks of the
    /// 128-band Whisper log-mel, chunks under 480 samples skipped.</summary>
    public int[] SemanticCodes(double[] audio)
    {
        var codes = new List<int>();
        for (int start = 0; start < audio.Length; start += ChunkSamples)
        {
            int count = Math.Min(ChunkSamples, audio.Length - start);
            if (count < 480) continue;
            var mel = S3Tokenizer<T>.LogMel(new ArraySegment<double>(audio, start, count), Semantic.Configuration.Mels);
            int mels = mel.GetLength(0), frames = mel.GetLength(1);
            var tensor = new Tensor<T>(new[] { mels, frames });
            for (int m = 0; m < mels; m++)
                for (int f = 0; f < frames; f++) tensor[m, f] = Ops.FromDouble(mel[m, f]);
            codes.AddRange(Semantic.Tokenize(tensor));
        }
        return codes.ToArray();
    }

    // ---------------------------------------------------------------- speakers

    /// <summary>
    /// A speaker from a prompt recording and its transcript (tts.py's <c>preprocess_prompt_wav</c>): the recording's
    /// dual-codebook ids, its mel at 22.05 kHz (the decoder's <c>feat_extractor</c>: 1024-point FFT, hop 256, 80 bands to 8
    /// kHz, not centred) and its CAM++ x-vector (16 kHz Kaldi filterbanks minus their mean).
    /// </summary>
    public StepAudioSpeaker<T> CreateSpeaker(Tensor<T> recording, int sampleRate, string transcript, string name)
    {
        using var _ = new NoGradScope<T>();
        var (ids, _, _) = AudioTokens(recording, sampleRate);
        var at22 = Wave(Resample(recording, sampleRate, Options.SampleRate));
        _mel ??= new DifferentiableMel<T>(_engine, Options.SampleRate, 1024, 256, 1024, Options.MelChannelCount, 0, 8000);
        var mel = _mel.Forward(at22);                                                                    // [frames, mels]
        var at16 = Wave(Resample(recording, sampleRate, TokenizerRate));
        _speakerFbank ??= new KaldiFbank<T>(_engine, TokenizerRate, 80, WindowType.Povey);
        var features = _speakerFbank.Forward(at16);
        features = _engine.TensorSubtract(features, _engine.TensorBroadcastTo(_engine.ReduceMean(features, new[] { 0 }, true), features.Shape.ToArray()));
        var embedding = _engine.Reshape(Speaker.Forward(_engine.Reshape(features, new[] { 1, features.Shape[0], features.Shape[1] })),
            new[] { Options.SpeakerEmbeddingDimension });
        return new StepAudioSpeaker<T> { Name = name, PromptText = transcript, PromptCode = ids, PromptMel = mel, Embedding = embedding };
    }

    // ---------------------------------------------------------------- generation

    /// <summary>
    /// The next token (transformers 4.48's sampling as the release calls <c>generate</c>): the repetition-aware processor
    /// when given (the last token barred once it fills more than the ratio of the window), then the temperature, top-k 50
    /// (the default generation config), top-p, and a draw.
    /// </summary>
    public int Sample(double[] logits, IReadOnlyList<int> sequence, double temperature, double topP, Random random, bool repetitionAware)
    {
        var o = Options;
        if (repetitionAware && sequence.Count >= o.RepetitionWindow)
        {
            int last = sequence[sequence.Count - 1], repeats = 0;
            for (int i = sequence.Count - o.RepetitionWindow; i < sequence.Count; i++) if (sequence[i] == last) repeats++;
            if ((double)repeats / o.RepetitionWindow > o.RepetitionRatio) logits[last] = double.NegativeInfinity;
        }
        if (temperature <= 0)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            return best;
        }
        var order = Enumerable.Range(0, logits.Length).Where(i => !double.IsNegativeInfinity(logits[i]))
            .OrderByDescending(i => logits[i]).ThenBy(i => i).Take(50).ToArray();
        double max = logits[order[0]] / temperature;
        var weights = order.Select(i => Math.Exp(logits[i] / temperature - max)).ToArray();
        double total = weights.Sum(), cumulative = 0;
        int keep = order.Length;
        if (topP < 1)
        {
            // top-p keeps the tokens whose cumulative probability (in descending order) stays within p, and always the first.
            keep = 0;
            while (keep < order.Length)
            {
                cumulative += weights[keep] / total;
                keep++;
                if (cumulative >= topP) break;
            }
        }
        double kept = weights.Take(keep).Sum(), draw = random.NextDouble() * kept;
        for (int i = 0; i < keep; i++)
        {
            draw -= weights[i];
            if (draw <= 0) return order[i];
        }
        return order[keep - 1];
    }

    /// <summary>Tokens a Step-1 model writes after <paramref name="prompt"/>, until &lt;|EOT|&gt; (not included) or the limits,
    /// with the KV cache.</summary>
    public List<int> Generate(LlamaNetwork<T> model, IReadOnlyList<int> prompt, int maxNew, double temperature, double topP, Random random,
        bool repetitionAware)
    {
        var cache = new LlamaCache<T>();
        var sequence = new List<int>(prompt);
        var output = new List<int>();
        int[] pending = prompt.ToArray();
        int width = model.Configuration.Hidden, vocabulary = model.Configuration.VocabSize;
        using var _ = new NoGradScope<T>();
        for (int step = 0; step < maxNew; step++)
        {
            var grid = new int[1, pending.Length];
            for (int i = 0; i < pending.Length; i++) grid[0, i] = pending[i];
            var (_, logits) = model.Forward(model.Embed(grid), null, null, cache, lastLogitsOnly: true);
            var row = new double[vocabulary];
            for (int v = 0; v < vocabulary; v++) row[v] = Ops.ToDouble(logits!.GetFlat(v));
            int next = Sample(row, sequence, temperature, topP, random, repetitionAware);
            if (next == StepAudioTokens.Eos) break;
            output.Add(next);
            sequence.Add(next);
            pending = new[] { next };
        }
        return output;
    }

    /// <summary>
    /// The TTS model's speech for text (tts.py's <c>__call__</c>): the two-turn prompt with the speaker, audio tokens
    /// sampled at the TTS temperature through the repetition-aware processor, then the decoder (the music one for RAP and
    /// humming tags): the flow from those ids − 65536 (negative ones clamped to 0) after the speaker's, its mel as the
    /// condition, then HiFT.
    /// </summary>
    public Tensor<T> Speak(string text, StepAudioSpeaker<T> speaker, Random random) =>
        Speak(text, Tokens.TtsPrompt(text, speaker.PromptText, speaker.Name, speaker.PromptCode), speaker, random);

    /// <summary>The speech for text given as token ids (the human turn holds exactly these ids).</summary>
    public Tensor<T> Speak(IReadOnlyList<int> textIds, StepAudioSpeaker<T> speaker, Random random)
    {
        string text = Tokens.Decode(textIds);
        return Speak(text, Tokens.TtsPrompt(text, textIds, speaker.PromptText, speaker.Name, speaker.PromptCode), speaker, random);
    }

    private Tensor<T> Speak(string text, int[] prompt, StepAudioSpeaker<T> speaker, Random random)
    {
        var o = Options;
        var output = Generate(Tts, prompt, Math.Min(o.TtsMaxNewTokens, Math.Max(1, o.TtsMaxLength - prompt.Length)), o.TtsTemperature, 1.0, random, repetitionAware: true);
        if (output.Count == 0) return new Tensor<T>(new[] { 0 });
        var decoder = StepAudioTokens.IsMusic(text) && Music is not null ? Music : Common;
        int vocabulary = decoder.Flow.Configuration.VocabSize;
        int[] Codes(IEnumerable<int> ids) => ids.Select(id => Math.Min(vocabulary - 1, Math.Max(0, id - StepAudioTokens.AudioBase))).ToArray();
        using var _ = new NoGradScope<T>();
        var tokens = Codes(output);
        var promptTokens = Codes(speaker.PromptCode);
        int frames = speaker.PromptMel.Shape[0] + decoder.Flow.MelFrames(tokens.Length);
        var noise = new Tensor<T>(new[] { 1, o.MelChannelCount, frames });
        for (int i = 0; i < noise.Length; i++)
            noise.SetFlat(i, Ops.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble())));
        var mel = decoder.Flow.Inference(tokens, promptTokens, speaker.PromptMel, speaker.Embedding, noise);
        var audio = decoder.Hift.Forward(mel, random);
        return _engine.Reshape(audio, new[] { audio.Length });
    }

    /// <summary>The chat model's answer to conversation turns (stepaudio.py's <c>__call__</c>), sampled at its temperature
    /// and top-p.</summary>
    public string Answer(IReadOnlyList<(string Role, string Content)> turns, Random random)
    {
        var o = Options;
        var output = Generate(ChatModel, Tokens.ChatPrompt(turns), o.ChatMaxNewTokens, o.ChatTemperature, o.ChatTopP, random, repetitionAware: false);
        return Tokens.Decode(output);
    }
}

/// <summary>Step-Audio's layer lists, by part.</summary>
internal sealed class StepAudioLayers<T>
{
    public List<LayerBase<T>> Chat { get; } = new();
    public List<LayerBase<T>> Tts { get; } = new();
    public List<LayerBase<T>> Tokenizer { get; } = new();
    public List<LayerBase<T>> Decoder { get; } = new();
    public List<LayerBase<T>> MusicDecoder { get; } = new();
    public List<LayerBase<T>> Reward { get; } = new();
    public List<LayerBase<T>> Critic { get; } = new();
    public List<LayerBase<T>> Reference { get; } = new();
}
