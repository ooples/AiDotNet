using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// What both KaniTTS releases share: an LFM2 language model writing NVIDIA NanoCodec codes after the release's prompt, the
/// codec, the speech extraction, the training sequence and its loss, and loading. Kani 2's frame-level positions,
/// learnable rotary scale and speaker embedding are options (<see cref="KaniTTSOptions"/>) that <see cref="KaniTTS2{T}"/>
/// turns on.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class KaniTTSBase<T> : TtsModelBase<T>, ICodecTts<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private readonly KaniTTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private Lfm2Network<T>? _lm;
    private NanoCodec<T>? _codec;
    private ByteLevelBpeTokenizer? _tokenizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly List<KaniExample> _pending = new();

    /// <summary>A training sequence: its ids, labels and (KaniTTS 2) speaker embedding.</summary>
    internal sealed record KaniExample(int[] Ids, int[] Labels, Tensor<T>? Speaker);

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _encoderLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>The options, for KaniTTS 2.</summary>
    protected KaniTTSOptions KaniOptions => _options;

    /// <summary>Creates an ONNX-backed model for inference.</summary>
    protected KaniTTSBase(NeuralNetworkArchitecture<T> architecture, string modelPath, KaniTTSOptions options)
        : base(architecture)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable model.</summary>
    protected KaniTTSBase(NeuralNetworkArchitecture<T> architecture, KaniTTSOptions options)
        : base(architecture)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        int hop = o.CodecDownsampleRates.Aggregate(1, (a, b) => a * b);
        if (hop != o.CodecUpsampleRates.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("The codec's decoder must upsample by its encoder's hop.");
        if (o.VocabSize < Layout.RequiredVocabulary)
            throw new ArgumentException($"The vocabulary ({o.VocabSize}) must hold the text tokens, ten control tokens and {o.Codebooks} codebooks of {CodebookSize} codes.");
        if (o.GradientAccumulationSteps < 1) throw new ArgumentOutOfRangeException(nameof(o.GradientAccumulationSteps), "At least one sample makes a step.");
        base.SampleRate = o.SampleRate;
        base.HopSize = hop;
        base.HiddenDim = o.HiddenDim;
        // The Trainer clips the gradient norm to max_grad_norm, which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodecLevels.Aggregate(1, (a, l) => a * l);

    /// <summary>Codec frames per second, rounded down (NanoCodec's 22,050 / 1764 is 12.5).</summary>
    public int CodecFrameRate => _options.SampleRate / HopSize;

    // The release's token layout: the text tokenizer's ids, then the control tokens, then each codebook's codes.
    private KaniLayout Layout => new(_options.TextVocabulary, _options.EndOfText, _options.Codebooks, CodebookSize);
    private int StartOfSpeech => Layout.StartOfSpeech;
    private int EndOfSpeech => Layout.EndOfSpeech;

    /// <summary>The speaker a multi-speaker checkpoint speaks as (<c>speaker_id</c>, such as "andrew"); null for none.
    /// Unset, a voice's <see cref="TtsVoice{T}.SpeakerId"/> picks one of <see cref="KaniTTSOptions.SpeakerNames"/>.</summary>
    public string? Speaker { get; set; }

    /// <inheritdoc />
    /// <remarks>KaniTTS trains on recordings: the transcript as the text tokenizer's ids (<see cref="TtsTrainingSample{T}.Tokens"/>,
    /// from <see cref="TtsModelBase{T}.TextToTokens"/>) and the recording at 22.05 kHz (<see cref="TtsTrainingSample{T}.Audio"/>)
    /// or its NanoCodec codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>, <c>[frames, 4]</c>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.Codebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => CodebookSize;

    /// <summary>Whether the release's networks exist (native mode without caller-supplied layers).</summary>
    protected bool HasPaperLayers => _lm is not null;
    private protected InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal Lfm2Network<T> LanguageModel => _lm ?? throw NotNative();
    internal NanoCodec<T> Codec => _codec ?? throw NotNative();

    /// <summary>The text tokenizer, read from the checkpoint's <c>tokenizer.json</c> by <see cref="LoadReferenceWeights"/>
    /// or <see cref="LoadTokenizer"/> (LFM2's tokenizer comes under its own license, so none is built in).</summary>
    internal ByteLevelBpeTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            if (_options.TokenizerJson is { } json) return SetTokenizer(json);
            throw new InvalidOperationException(
                "KaniTTS's tokenizer comes with its weights: call LoadReferenceWeights(directory) or LoadTokenizer(tokenizer.json) first.");
        }
    }

    // The options keep the tokenizer's JSON, so a clone or a deserialized model reads the same tokenizer.
    private ByteLevelBpeTokenizer SetTokenizer(string json)
    {
        var tokenizer = ByteLevelBpeTokenizer.FromJson(json);
        // LFM2's tokenizer names every id of the model (its added tokens cover the audio codes).
        if (tokenizer.VocabularySize > _options.VocabSize)
            throw new InvalidDataException($"The tokenizer has {tokenizer.VocabularySize} tokens; the model's vocabulary is {_options.VocabSize}.");
        _options.TokenizerJson = json;
        return _tokenizer = tokenizer;
    }

    /// <summary>Reads the text tokenizer from a Hugging Face <c>tokenizer.json</c>.</summary>
    public void LoadTokenizer(string path) => SetTokenizer(File.ReadAllText(path));

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.TextVocabulary) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _lm = new Lfm2Network<T>(Engine, _lmLayers, new Lfm2Configuration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, AttentionLayers = (bool[])o.AttentionLayers.Clone(), Heads = o.Heads,
            KeyValueHeads = o.KeyValueHeads, Intermediate = o.Intermediate, AutoAdjustFeedForward = o.AutoAdjustFeedForward,
            FeedForwardMultiplier = o.FeedForwardMultiplier, FeedForwardMultipleOf = o.FeedForwardMultipleOf, NormEpsilon = o.NormEpsilon,
            RopeTheta = o.RopeTheta, ConvKernel = o.ConvKernel, PadTokenId = 0, TieWordEmbeddings = true,
            LoraRank = o.UseLora ? o.LoraRank : 0, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
            LoraTargets = (string[])o.LoraTargets.Clone(), RankStabilizedLora = o.RankStabilizedLora,
            LearnableRope = o.LearnableRope, RopeAlphaMin = o.RopeAlphaMin, RopeAlphaMax = o.RopeAlphaMax,
            SpeakerEmbeddingDim = o.SpeakerEmbeddingDim,
        });
        _lm.InitializeLikeReference(random);
        _codec = new NanoCodec<T>(Engine, new NanoCodecConfiguration
        {
            SampleRate = o.SampleRate, EncoderChannels = o.CodecEncoderChannels, DownsampleRates = (int[])o.CodecDownsampleRates.Clone(),
            EncodedDim = o.CodecEncodedDim, DecoderChannels = o.CodecDecoderChannels, UpsampleRates = (int[])o.CodecUpsampleRates.Clone(),
            QuantizerGroups = o.Codebooks, QuantizerLevels = (int[])o.CodecLevels.Clone(),
        }, _encoderLayers, _decoderLayers);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_encoderLayers);
        ComponentLayers.AddRange(_decoderLayers);
    }

    // ---------------------------------------------------------------- the codec

    private int[,] CodecEncode(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        return Codec.Encode(audio);
    }

    private Tensor<T> CodecDecode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        return Codec.Decode(codes);
    }

    /// <inheritdoc />
    /// <remarks>NanoCodec's codes <c>[4, frames]</c> of a 22.05 kHz recording (<c>AudioCodecModel.encode</c>: zero-padded
    /// to whole frames of 1764 samples).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = CodecEncode(audio);
        var tokens = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int t = 0; t < codes.GetLength(1); t++) tokens[q, t] = NumOps.FromDouble(codes[q, t]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio of NanoCodec codes <c>[4, frames]</c> (<c>AudioCodecModel.decode</c>).</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        return CodecDecode(Codes(tokens));
    }

    private int[,] Codes(Tensor<T> tokens)
    {
        if (tokens.Rank != 2 || tokens.Shape[0] != _options.Codebooks)
            throw new ArgumentException($"Codes are [{_options.Codebooks}, frames].", nameof(tokens));
        var codes = new int[tokens.Shape[0], tokens.Shape[1]];
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int t = 0; t < codes.GetLength(1); t++) codes[q, t] = (int)Math.Round(NumOps.ToDouble(tokens[q, t]));
        return codes;
    }

    // A training sample's codes, [frames, codebooks], as [codebooks, frames].
    private int[,] SampleCodes(Tensor<T> tokens)
    {
        if (tokens.Rank != 2 || tokens.Shape[1] != _options.Codebooks)
            throw new ArgumentException($"A sample's codec tokens are [frames, {_options.Codebooks}].", nameof(tokens));
        var codes = new int[tokens.Shape[1], tokens.Shape[0]];
        for (int t = 0; t < tokens.Shape[0]; t++)
            for (int q = 0; q < tokens.Shape[1]; q++) codes[q, t] = (int)Math.Round(NumOps.ToDouble(tokens[t, q]));
        return codes;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's ids in the text tokenizer, without its start token (synthesis builds the prompt from their text).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private string TextOf(Tensor<T> ids) =>
        Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))), skipSpecial: true);

    /// <summary>The prefix synthesis puts before the text ("prefix: text"): <see cref="Speaker"/>, or the voice's
    /// speaker's name; KaniTTS 2 puts a language tag there.</summary>
    protected virtual string? SynthesisPrefix() => SpeakerFor(Voice?.SpeakerId);

    /// <summary>The prefix a training sample's text takes: its speaker's name (lowercased by the pipeline).</summary>
    protected virtual string? TrainingPrefix(TtsTrainingSample<T> sample) => SpeakerFor(sample.SpeakerId);

    /// <summary>The speaker embedding synthesis inserts after the first token (KaniTTS 2), null for none.</summary>
    protected virtual Tensor<T>? SynthesisSpeakerEmbedding() => null;

    /// <summary>The speaker embedding a training sample's sequence carries (KaniTTS 2), null for none.</summary>
    protected virtual Tensor<T>? TrainingSpeakerEmbedding(TtsTrainingSample<T> sample) => null;

    // A speaker id picks a name of a multi-speaker checkpoint; a single-speaker checkpoint has none and ignores it.
    private string? SpeakerFor(int? speakerId)
    {
        var names = _options.SpeakerNames;
        if (speakerId is null || names.Length == 0) return Speaker;
        if (speakerId < 0 || speakerId >= names.Length)
            throw new ArgumentOutOfRangeException(nameof(speakerId), $"Speaker {speakerId} is not one of the checkpoint's {names.Length} speakers.");
        return names[speakerId.Value];
    }

    /// <summary>The release's prompt (<c>KaniModel.get_input_ids</c>): start-of-human, the tokenizer's ids of
    /// "speaker: text" (or the text) with its start token, end-of-text, end-of-human.</summary>
    internal int[] PromptIds(string text, string? speaker) => Layout.PromptIds(Tokenizer, text, speaker);

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's ids (<see cref="PreprocessText"/>); the audio of the speech the model
    /// writes after the release's prompt for it (with <see cref="Speaker"/>, or the voice's speaker).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.TextVocabulary))
            throw new ArgumentOutOfRangeException(nameof(input), $"Text ids lie in [0, {_options.TextVocabulary}).");
        var prompt = PromptIds(TextOf(input), SynthesisPrefix());
        var generated = Generate(prompt, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), SynthesisSpeakerEmbedding());
        return AudioOf(prompt.Concat(generated).ToList());
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // Hugging Face generate() as the kani-tts package calls it: the repetition penalty over every token so far, then
    // temperature, top-k and top-p, a draw (or the argmax at temperature 0), until end-of-speech or the token limit. With
    // a speaker embedding it is inserted after the first prompt token; with frame-level positions each fed token's
    // position follows KaniTTS2ForCausalLM.prepare_inputs_for_generation: a non-audio token at the cache's length, the
    // first audio token there too, and each later frame's four tokens AudioStep further on.
    internal List<int> Generate(IReadOnlyList<int> prompt, Random random, Tensor<T>? speaker = null)
    {
        var o = _options;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var lm = LanguageModel;
            lm.Training = false;
            var ids = new List<int>(prompt);
            var generated = new List<int>();
            var cache = new Lfm2Cache<T>();
            var embeddings = speaker is null ? lm.Embed(Row(prompt)) : lm.EmbedWithSpeaker(Row(prompt), speaker);
            // Prefill: sequential positions with a speaker; the prompt's frame-level positions (sequential, as it holds no
            // audio) without one.
            var prefill = o.FrameLevelPositions && speaker is null ? FramePositions(prompt) : null;
            var logits = lm.Forward(embeddings, cache, lastLogitsOnly: true, prefill).Logits;
            int audioGenerated = 0;
            double? framePosition = null;
            for (int step = 0; step < o.MaxNewTokens; step++)
            {
                var values = logits.ToArray();
                var scores = new double[values.Length];
                for (int i = 0; i < scores.Length; i++) scores[i] = NumOps.ToDouble(values[i]);
                int next;
                if (o.Temperature > 0)
                    next = OuteTTSSampler.Sample(OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, ids.Count, o.Temperature, o.TopK, o.TopP, 0), random);
                else
                {
                    var processed = OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, ids.Count, 1.0, 0, 1.0, 0);
                    next = Array.IndexOf(processed, processed.Max());
                }
                ids.Add(next);
                generated.Add(next);
                if (next == EndOfSpeech) break;
                double[]? position = null;
                if (o.FrameLevelPositions)
                {
                    if (next < Layout.AudioTokensStart) position = new double[] { cache.Length };
                    else
                    {
                        framePosition ??= cache.Length;
                        if (audioGenerated % o.Codebooks == 0 && audioGenerated > 0) framePosition += o.AudioStep;
                        position = new[] { framePosition.Value };
                        audioGenerated++;
                    }
                }
                // With a speaker, generate()'s cache position runs one behind the cache (the inserted slot is not counted),
                // and the causal mask it builds hides the new token's own key.
                int lag = speaker is not null && o.ReleaseSpeakerCachePosition ? 1 : 0;
                logits = lm.Forward(lm.Embed(Row(new[] { next })), cache, lastLogitsOnly: true, position, lag).Logits;
            }
            return generated;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>KaniTTS 2's <c>compute_frame_level_positions</c>: each non-audio token one step after the last, each audio
    /// frame's tokens sharing one position <see cref="KaniTTSOptions.AudioStep"/> after the previous frame's.</summary>
    internal double[] FramePositions(IReadOnlyList<int> ids)
    {
        var positions = new double[ids.Count];
        int text = 0, audio = 0;
        for (int i = 0; i < ids.Count; i++)
        {
            positions[i] = text + audio / _options.Codebooks * _options.AudioStep;
            if (ids[i] >= Layout.AudioTokensStart) audio++;
            else text++;
        }
        return positions;
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    /// <summary>The audio of a generated sequence (prompt and continuation), by <see cref="KaniTTSOptions.LenientSpeechSpans"/>.</summary>
    internal Tensor<T> AudioOf(IReadOnlyList<int> sequence) => _options.LenientSpeechSpans ? StreamedAudio(sequence) : SpanAudio(sequence);

    // NemoAudioPlayer.get_waveform: the codes between the first start-of-speech and the first end-of-speech.
    private Tensor<T> SpanAudio(IReadOnlyList<int> sequence) => CodecDecode(Layout.SpanCodes(sequence));

    private Tensor<T>? DecodeFrames(IReadOnlyList<int> tokens, int frames) =>
        Layout.Codes(tokens, frames) is { } codes ? CodecDecode(codes) : null;

    // The basic example's StreamingAudioWriter: tokens after start-of-speech decoded in windows of new frames with
    // lookback context, a window with tokens that are not codes skipped, the rest decoded at end-of-speech.
    private Tensor<T> StreamedAudio(IReadOnlyList<int> sequence)
    {
        int chunk = _options.StreamingChunkFrames, lookback = _options.StreamingLookbackFrames, q = _options.Codebooks;
        var all = new List<int>();
        var pieces = new List<Tensor<T>>();
        bool inside = false, ended = false;
        int decoded = 0;
        (Tensor<T>? Audio, int Frames, int Skip) Window()
        {
            int startFrame = Math.Max(0, decoded - lookback);
            var tokens = all.Skip(startFrame * q).ToList();
            int frames = tokens.Count / q;
            if (frames == 0) return (null, 0, 0);
            var audio = DecodeFrames(tokens, frames);
            return (audio, frames, Math.Min(decoded, lookback));
        }
        foreach (int token in sequence)
        {
            if (token == StartOfSpeech)
            {
                inside = true;
                ended = false;
                continue;
            }
            if (token == EndOfSpeech)
            {
                if (ended) continue;
                if (all.Count / q - decoded >= 1)
                {
                    var (audio, frames, skip) = Window();
                    if (audio is not null)
                    {
                        int perFrame = audio.Length / frames;
                        pieces.Add(Engine.TensorSlice(audio, new[] { skip * perFrame }, new[] { audio.Length - skip * perFrame }));
                    }
                }
                inside = false;
                ended = true;
                continue;
            }
            if (!inside || ended) continue;
            all.Add(token);
            if (all.Count / q - decoded >= chunk)
            {
                var (audio, frames, skip) = Window();
                if (audio is null) continue;
                int perFrame = audio.Length / frames;
                int from = skip * perFrame, take = Math.Min(chunk * perFrame, audio.Length - from);
                pieces.Add(Engine.TensorSlice(audio, new[] { from }, new[] { take }));
                decoded += chunk;
            }
        }
        return pieces.Count == 0 ? new Tensor<T>(new[] { 0 }) : Engine.TensorConcatenate(pieces.ToArray(), 0);
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads a released Kani 1 checkpoint directory: <c>model.safetensors</c> (transformers' <c>Lfm2ForCausalLM</c>),
    /// <c>tokenizer.json</c> and, for a multi-speaker checkpoint, <c>config.json</c>'s speaker list. LoRA adapters the
    /// checkpoint lacks start as peft starts them (A uniform in ±1/√inputs, B zero), which leaves the model unchanged.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string tokenizer = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(tokenizer)) LoadTokenizer(tokenizer);
        string config = Path.Combine(directory, "config.json");
        if (File.Exists(config)) ReadCheckpointConfig(JObject.Parse(File.ReadAllText(config)));
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(directory, "*.safetensors"))
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        LoadLanguageModel(name => weights.TryGetValue(name, out var v) ? v : null);
    }

    /// <summary>Reads what the checkpoint's <c>config.json</c> adds to the options: Kani 1's speaker list.</summary>
    protected virtual void ReadCheckpointConfig(JObject config)
    {
        if (config["speaker_settings"]?["speaker_list"] is JArray speakers) _options.SpeakerNames = speakers.Select(s => (string)s!).ToArray();
    }

    /// <summary>Loads the language model's <c>Lfm2ForCausalLM</c> state dict; LoRA adapters it lacks start as peft starts
    /// them (A uniform in ±1/√inputs, B zero).</summary>
    internal void LoadLanguageModel(Func<string, double[]?> read)
    {
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        LanguageModel.LoadTorchWeights((name, shape) =>
        {
            if (read(name) is { } v) return v;
            if (!name.Contains(".lora_")) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            var values = new double[shape.Aggregate(1, (a, s) => a * s)];
            if (name.EndsWith("lora_A.weight", StringComparison.Ordinal))
            {
                double bound = 1.0 / Math.Sqrt(shape[1]);
                for (int i = 0; i < values.Length; i++) values[i] = bound * (2 * random.NextDouble() - 1);
            }
            return values;
        });
    }

    /// <summary>Loads NanoCodec from NeMo's <c>.nemo</c> archive (<c>nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo</c>) or the
    /// <c>model_weights.ckpt</c> inside it; the discriminators' tensors are not needed.</summary>
    public void LoadCodecWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string checkpoint = path;
        string? extracted = null;
        if (path.EndsWith(".nemo", StringComparison.OrdinalIgnoreCase))
        {
            extracted = Path.Combine(Path.GetTempPath(), $"kanitts-{Guid.NewGuid():N}.ckpt");
            if (!TarArchiveReader.TryExtract(path, "model_weights.ckpt", extracted))
                throw new InvalidDataException("The .nemo archive has no model_weights.ckpt.");
            checkpoint = extracted;
        }
        try
        {
            var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint))
                if (entry.Key.StartsWith("audio_encoder.", StringComparison.Ordinal) || entry.Key.StartsWith("audio_decoder.", StringComparison.Ordinal))
                    state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
            LoadCodec(name => state.TryGetValue(name, out var v) ? v : null, state.Keys);
        }
        finally
        {
            if (extracted is not null && File.Exists(extracted)) File.Delete(extracted);
        }
    }

    internal void LoadCodec(Func<string, double[]?> read, IEnumerable<string> names)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        Codec.LoadNemoWeights(name =>
        {
            var v = read(name);
            if (v is not null) used.Add(name);
            return v;
        });
        var unused = names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The codec checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>
    /// The release's training example (KaniTTS-Finetune-pipeline's <c>TrainDataPreProcessor</c>): start-of-human, the
    /// tokenizer's ids of "speaker: text" (the speaker lowercased) with its start token, end-of-text, end-of-human,
    /// start-of-AI, start-of-speech, the recording's NanoCodec frames (consecutive repeats kept once) as codes offset by
    /// codebook, end-of-speech, end-of-AI; the causal language-model loss on every token. Every
    /// <see cref="KaniTTSOptions.GradientAccumulationSteps"/>-th sample makes an optimizer step on the loss averaged over
    /// all the tokens since the last step, as the Trainer accumulates; the other samples return their own loss.
    /// </remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var example = TrainingExample(sample);
        _pending.Add(example);
        if (_pending.Count < _options.GradientAccumulationSteps)
        {
            using var _ = new NoGradScope<T>();
            return Loss(new[] { example })[0];
        }
        return Step(sample.Tokens);
    }

    /// <summary>Makes an optimizer step on the samples accumulated since the last one, if any (as the Trainer does with
    /// the remainder at an epoch's end); returns their loss, or zero when none were waiting.</summary>
    public T CompleteTrainingStep()
    {
        ThrowIfDisposed();
        if (_pending.Count == 0) return NumOps.Zero;
        var key = new Tensor<T>(new[] { 1 });
        return Step(key);
    }

    private T Step(Tensor<T> key)
    {
        var batch = _pending.ToList();
        _pending.Clear();
        var lm = LanguageModel;
        try
        {
            lm.Training = true;
            return TrainWithCustomObjective(key, key, (_, _) => Loss(batch), Optimizer());
        }
        finally
        {
            lm.Training = false;
        }
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            LanguageModel.Training = false;
            return Loss(new[] { TrainingExample(sample) })[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private readonly Dictionary<Tensor<T>, (string Transcript, string? Prefix, object? SpeakerKey, KaniExample Example)> _examples =
        new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    // The release's input_ids (and labels, the same) for a sample, and its speaker embedding (KaniTTS 2).
    internal KaniExample TrainingExample(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var source = sample.CodecTokens ?? sample.Audio
            ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio or CodecTokens.", nameof(sample));
        string transcript = TextOf(sample.Tokens);
        string? prefix = TrainingPrefix(sample);
        object? speakerKey = sample.SpeakerReference ?? (object?)sample.SpeakerId;
        if (_examples.TryGetValue(source, out var cached) && cached.Transcript == transcript && cached.Prefix == prefix
            && Equals(cached.SpeakerKey, speakerKey))
            return cached.Example;
        var codes = sample.CodecTokens is not null ? SampleCodes(sample.CodecTokens) : CodecEncode(sample.Audio!);
        var ids = ExampleIds(transcript, prefix, codes);
        var example = new KaniExample(ids, ids, TrainingSpeakerEmbedding(sample));
        if (_examples.Count > 64) _examples.Clear();
        _examples[source] = (transcript, prefix, speakerKey, example);
        return example;
    }

    internal int[] ExampleIds(string transcript, string? prefix, int[,] codes) =>
        Layout.TrainingIds(Tokenizer, transcript, prefix, codes, _options.RemoveRepeatedFrames);

    // The loss over several examples, averaged over all their scored tokens (the Trainer's num_items_in_batch). A speaker
    // embedding goes in after the first token with no label of its own (KaniTTS 2's DataCollator), and frame-level
    // positions shift by one after it.
    internal Tensor<T> Loss(IReadOnlyList<KaniExample> examples)
    {
        var lm = LanguageModel;
        var rows = examples.Select(e =>
        {
            var labels = e.Speaker is null ? e.Labels : new[] { e.Labels[0], -100 }.Concat(e.Labels.Skip(1)).ToArray();
            double[]? positions = null;
            if (_options.FrameLevelPositions)
            {
                var basis = FramePositions(e.Ids);
                positions = e.Speaker is null ? basis : new[] { basis[0], basis[0] + 1 }.Concat(basis.Skip(1).Select(p => p + 1)).ToArray();
            }
            return (Ids: e.Ids, Labels: labels, Speaker: e.Speaker, Positions: positions);
        }).ToList();
        int total = rows.Sum(r => r.Labels.Skip(1).Count(l => l != -100));
        Tensor<T>? loss = null;
        foreach (var (ids, labels, speaker, positions) in rows)
        {
            int scored = labels.Skip(1).Count(l => l != -100);
            if (scored == 0) continue;
            var embeddings = speaker is null ? lm.Embed(Row(ids)) : lm.EmbedWithSpeaker(Row(ids), speaker);
            var logits = lm.Forward(embeddings, null, false, positions).Logits;
            var part = Engine.TensorMultiplyScalar(lm.CausalLmLoss(logits, Row(labels)), NumOps.FromDouble(scored / (double)total));
            loss = loss is null ? part : Engine.TensorAdd(loss, part);
        }
        return loss ?? throw new ArgumentException("The examples score no tokens.", nameof(examples));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradNorm > 0, MaxGradientNorm = o.MaxGradNorm,
        };
        if (!o.UseLora)
        {
            var lm = LanguageModel;
            var alphas = new HashSet<LayerBase<T>>(lm.Layers.Where(l => l.RopeAlpha is not null).Select(l => (LayerBase<T>)l.RopeAlpha!));
            // Kani 2's LearnableRoPETrainer decays every parameter but the rotary scales, which learn at
            // RopeAlphaLearningRateRatio times the rate; Kani 1's Trainer exempts biases and norms
            // (get_decay_parameter_names).
            var exempt = o.LearnableRope ? alphas
                : new HashSet<LayerBase<T>>(lm.Layers.SelectMany(l => new LayerBase<T>?[]
                {
                    l.OperatorNorm, l.FeedForwardNorm, l.QueryNorm, l.KeyNorm, l.ConvKernelBias, l.ConvIn?.Bias, l.ConvOut?.Bias,
                }).Where(x => x is not null)!.Cast<LayerBase<T>>()) { lm.FinalNorm };
            var parameters = new List<Tensor<T>>();
            var decay = new List<T>();
            var rates = new List<T>();
            foreach (var layer in _lmLayers.OfType<BiasFreeLinearLayer<T>>())
            {
                parameters.Add(layer.Weights);
                var d = exempt.Contains(layer) ? NumOps.Zero : NumOps.One;
                var r = alphas.Contains(layer) ? NumOps.FromDouble(o.RopeAlphaLearningRateRatio) : NumOps.One;
                for (int i = 0; i < layer.Weights.Length; i++)
                {
                    decay.Add(d);
                    rates.Add(r);
                }
            }
            options.WeightDecayMask = new Vector<T>(decay.ToArray());
            if (alphas.Count > 0) options.LearningRateMask = new Vector<T>(rates.ToArray());
            options.MaskParameters = parameters;
        }
        if (o.TrainingSteps > 0)
        {
            int warmup = o.WarmupSteps > 0 ? o.WarmupSteps : (int)Math.Ceiling(o.WarmupRatio * o.TrainingSteps);
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, warmup, o.TrainingSteps, 0.0, LinearWarmupScheduler.DecayMode.Cosine);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the LoRA adapters (or, without LoRA, the language model); the codec is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = _options.UseLora
            ? new HashSet<Tensor<T>>(LanguageModel.LoraLayers.Select(l => l.Weights), Helpers.TensorReferenceComparer<Tensor<T>>.Instance)
            : new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
                Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? $"{GetType().Name.Split('`')[0]}-Native" : $"{GetType().Name.Split('`')[0]}-ONNX",
            Description = "KaniTTS: text-to-speech as LFM2 language modeling over NanoCodec codes (nineninesix.ai)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = GetType().Name.Split('`')[0];
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(KaniTTSBase<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
