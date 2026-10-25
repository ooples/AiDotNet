using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Whisper;
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
using AiDotNet.TextToSpeech.StyleEmotion;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// OuteTTS 1.0 (OuteAI): text-to-speech as plain language modeling — a Llama 3.2 or Qwen3 language model reads the text
/// and continues it with each word's duration, energy, spectral-centroid and pitch tokens and its DAC codes, which a DAC
/// decoder turns into 24 kHz audio. A voice is a short recording's words with their codes (a speaker profile), timed by
/// Whisper.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>OuteTTS has no paper; its release (edwko/OuteTTS 1.0, OuteAI/Llama-OuteTTS-1.0-1B and OuteTTS-1.0-0.6B) is the
/// reference: the prompt format, text normalization and chunking, sampling, the DAC interface and speaker profiles are
/// ported from it, with Whisper's transcription and word timing from openai-whisper.</para>
/// <para><b>For Beginners:</b> OuteTTS treats speech as text a language model can write: after the sentence to speak it
/// writes, word by word, little tokens that describe the sound, and a codec turns them into audio. Give it a few seconds of
/// someone speaking and it continues in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new OuteTTS&lt;float&gt;(architecture, new OuteTTSOptions());
/// model.LoadReferenceWeights("Llama-OuteTTS-1.0-1B");          // model.safetensors and tokenizer.json
/// model.LoadCodecWeights("weights_24khz_1.5kbps_v1.0.pth");      // ibm-research/DAC.speech.v1.0
/// model.Speaker = OuteTTSSpeaker.DefaultEnglishFemale();
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("OuteTTS 1.0", "https://github.com/edwko/OuteTTS", Year = 2025, Authors = "OuteAI")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "The release states no training recipe; its training guide (examples/training/OuteTTS-0.3/train.md) fine-tunes "
             + "with any language-model trainer and names TRL's SFTTrainer, whose SFTConfig defaults are AdamW at 2e-5 "
             + "(betas 0.9, 0.999, eps 1e-8, no weight decay), linear decay without warm-up, gradient norm clipped to 1.0, "
             + "and the loss on the completion only.")]
public partial class OuteTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly OuteTTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private LlamaNetwork<T>? _lm;
    private DacEncoder<T>? _codecEncoder;
    private FactorizedVectorQuantizerLayer<T>? _quantizer;
    private DacDecoder<T>? _codecDecoder;
    private OpenAiWhisperNetwork<T>? _aligner;
    private OpenAiWhisperTranscriber<T>? _transcriber;
    private ByteLevelBpeTokenizer? _tokenizer;
    private Dictionary<int, int>? _c1, _c2;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private (Tensor<T> Audio, OuteTTSSpeaker Speaker)? _voiceSpeaker;

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();
    private readonly List<LayerBase<T>> _alignerLayers = new();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed OuteTTS for inference.</summary>
    public OuteTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, OuteTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OuteTTSOptions();
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable OuteTTS.</summary>
    public OuteTTS(NeuralNetworkArchitecture<T> architecture, OuteTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OuteTTSOptions();
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.CodecEncoderRates.Aggregate(1, (a, b) => a * b);
        base.HiddenDim = o.HiddenDim;
        if (o.CodecEncoderRates.Aggregate(1, (a, b) => a * b) != o.CodecDecoderRates.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("The codec's decoder must upsample by its encoder's hop.");
        // TRL's SFTTrainer clips the gradient norm to 1.0 (max_grad_norm), which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.CodecCodebooks;

    /// <inheritdoc />
    public int CodebookSize => _options.CodecCodebookSize;

    /// <summary>Codec frames per second (75).</summary>
    public int CodecFrameRate => _options.SampleRate / HopSize;

    /// <summary>The speaker profile synthesis continues (null: no speaker, as the release allows). A voice with a
    /// <see cref="TtsVoice{T}.ReferenceAudio"/> builds one with <see cref="CreateSpeaker"/> when this is unset.</summary>
    public OuteTTSSpeaker? Speaker { get; set; }

    /// <inheritdoc />
    /// <remarks>OuteTTS trains on recordings: the recording (24 kHz) and its transcript as the language model's token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>, from <see cref="TtsModelBase{T}.TextToTokens"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _lm is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal LlamaNetwork<T> LanguageModel => _lm ?? throw NotNative();
    internal OpenAiWhisperTranscriber<T> Transcriber => _transcriber ?? throw NotNative();
    internal OpenAiWhisperNetwork<T> Aligner => _aligner ?? throw NotNative();

    /// <summary>The language model's tokenizer: built in for the Qwen3 backbone, read by <see cref="LoadReferenceWeights"/>
    /// (or <see cref="LoadTokenizer"/>) for Llama.</summary>
    internal ByteLevelBpeTokenizer Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            if (_options.Backbone == OuteTTSBackbone.Qwen3_0_6B) return SetTokenizer(OuteTTSTokenizers.Qwen3);
            throw new InvalidOperationException("Llama-OuteTTS's tokenizer comes with its weights: call LoadReferenceWeights(directory) or " +
                "LoadTokenizer(tokenizer.json) first.");
        }
    }

    private ByteLevelBpeTokenizer SetTokenizer(ByteLevelBpeTokenizer tokenizer)
    {
        if (tokenizer.VocabularySize > _options.VocabSize)
            throw new InvalidDataException($"The tokenizer has {tokenizer.VocabularySize} tokens; the model's vocabulary is {_options.VocabSize}.");
        _tokenizer = tokenizer;
        _c1 = new Dictionary<int, int>();
        _c2 = new Dictionary<int, int>();
        for (int i = 0; i <= 1024; i++)
        {
            _c1[tokenizer.Encode($"<|c1_{i}|>")[0]] = i;
            _c2[tokenizer.Encode($"<|c2_{i}|>")[0]] = i;
        }
        return tokenizer;
    }

    /// <summary>Reads the language model's tokenizer from a Hugging Face <c>tokenizer.json</c>.</summary>
    public void LoadTokenizer(string path) => SetTokenizer(OuteTTSTokenizers.FromFile(path));

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

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
        _lm = new LlamaNetwork<T>(Engine, _lmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.Heads, KeyValueHeads = o.KeyValueHeads, HeadDim = o.HeadDim,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, RopeScalingFactor = o.RopeScalingFactor,
            RopeLowFrequencyFactor = o.RopeLowFrequencyFactor, RopeHighFrequencyFactor = o.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = o.RopeOriginalMaxPositions, QueryKeyNorm = o.QueryKeyNorm, TieWordEmbeddings = true,
        });
        _lm.InitializeLikeReference(random);
        _codecEncoder = new DacEncoder<T>(Engine, o.CodecEncoderDim, o.CodecEncoderRates, o.CodecLatentDim, _codecLayers);
        _quantizer = new FactorizedVectorQuantizerLayer<T>(o.CodecLatentDim, o.CodecCodebooks, o.CodecCodebookSize, o.CodecCodebookDim);
        _codecLayers.Add(_quantizer);
        _codecDecoder = new DacDecoder<T>(Engine, o.CodecLatentDim, o.CodecDecoderDim, o.CodecDecoderRates, _codecLayers);
        _aligner = new OpenAiWhisperNetwork<T>(Engine, _alignerLayers, new WhisperDimensions(o.AlignerMels, 1500, o.AlignerWidth, o.AlignerHeads,
            o.AlignerEncoderLayers, 51866, 448, o.AlignerWidth, o.AlignerHeads, o.AlignerDecoderLayers), random);
        _transcriber = new OpenAiWhisperTranscriber<T>(Engine, _aligner);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
        ComponentLayers.AddRange(_alignerLayers);
    }

    // ---------------------------------------------------------------- the codec

    private int[,] CodecEncode(Tensor<T> chunk)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _codecLayers) layer.SetTrainingMode(false);
        return _quantizer!.Encode(_codecEncoder!.Forward(chunk), _options.CodecCodebooks);
    }

    private Tensor<T> CodecDecode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        foreach (var layer in _codecLayers) layer.SetTrainingMode(false);
        return _codecDecoder!.Forward(_quantizer!.Decode(codes));
    }

    /// <inheritdoc />
    /// <remarks>The DAC codes <c>[codebooks, frames]</c> of a 24 kHz recording, as the release encodes it (loudness to −18
    /// LUFS, 5-second windows).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = OuteTTSAudio.Encode<T>(CodecEncode, HopSize, Doubles(audio), _options.SampleRate);
        var tokens = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int t = 0; t < codes.GetLength(1); t++) tokens[q, t] = NumOps.FromDouble(codes[q, t]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio of DAC codes <c>[codebooks, frames]</c>, as the release decodes them (2048-frame chunks with
    /// fades, loudness to −18 LUFS).</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        int q = tokens.Shape[0], frames = tokens.Shape[1];
        var codes = new int[q, frames];
        for (int i = 0; i < q; i++)
            for (int t = 0; t < frames; t++) codes[i, t] = (int)Math.Round(NumOps.ToDouble(tokens[i, t]));
        return Wave(OuteTTSAudio.Decode<T>(CodecDecode, codes, _options.SampleRate));
    }

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    // ---------------------------------------------------------------- speaker profiles

    /// <summary>
    /// A speaker profile from a recording at the model's sample rate (the release's <c>create_speaker</c>): Whisper
    /// transcribes it with word timestamps (or, given <paramref name="transcript"/>, times that transcript's words), DAC
    /// encodes it, and each word takes its frames, duration and energy, spectral-centroid and pitch features.
    /// </summary>
    /// <remarks>Whisper reads the recording resampled to 16 kHz by torchaudio's sinc resampler; the release has ffmpeg
    /// load the file at 16 kHz. With a transcript, its words are timed by Whisper's alignment of those tokens rather than
    /// transcribed (the release always transcribes, ignoring a transcript).</remarks>
    public OuteTTSSpeaker CreateSpeaker(Tensor<T> audio, string? transcript = null)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var samples = Doubles(audio);
        double seconds = samples.Length / (double)_options.SampleRate;
        if (seconds > _options.MaxSpeakerSeconds)
            throw new ArgumentException($"Speaker audio is longer than {_options.MaxSpeakerSeconds} seconds. Use a shorter clip.", nameof(audio));
        var whisperAudio = _options.SampleRate == OpenAiWhisperTranscriber<T>.SampleRate ? samples
            : Doubles(new TorchaudioResample<T>(Engine, _options.SampleRate, OpenAiWhisperTranscriber<T>.SampleRate).Forward(audio));
        string text;
        var words = new List<(string Word, double Start, double End)>();
        if (transcript is null)
        {
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            var result = Transcriber.TranscribeAudio(whisperAudio, _options.AlignerTemperatures, true, null, random);
            text = OuteTTSPrompts.NormalizeTranscript(result.Text);
            foreach (var segment in result.Segments)
                foreach (var w in segment.Words) words.Add((w.Word.Trim(), w.Start, w.End));
        }
        else
        {
            text = OuteTTSPrompts.NormalizeTranscript(transcript);
            words.AddRange(TimeTranscript(whisperAudio, transcript));
        }
        return SpeakerFromWords(samples, text, words);
    }

    // The transcript's words timed by Whisper's alignment over the recording's first window.
    private IEnumerable<(string Word, double Start, double End)> TimeTranscript(double[] audio, string transcript)
    {
        var t = Transcriber;
        var tokens = t.Encode(" " + transcript.Trim()).ToList();
        if (tokens.Count == 0) return Array.Empty<(string, double, double)>();
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(audio, Aligner.Dimensions.Mels, OpenAiWhisperTranscriber<T>.ChunkSamples);
        int content = mel.GetLength(1) - OpenAiWhisperTranscriber<T>.ChunkFrames;
        int frames = Math.Min(content, OpenAiWhisperTranscriber<T>.ChunkFrames);
        var window = new Tensor<T>(new[] { mel.GetLength(0), OpenAiWhisperTranscriber<T>.ChunkFrames });
        for (int m = 0; m < mel.GetLength(0); m++)
            for (int f = 0; f < frames; f++) window[m, f] = NumOps.FromDouble(mel[m, f]);
        using var _ = new NoGradScope<T>();
        var language = Transcriber.DetectLanguage(Aligner.Encode(window)).OrderByDescending(p => p.Value).First().Key;
        var segment = new WhisperTranscriptSegment
        {
            Seek = 0, Start = 0, End = frames / (double)OpenAiWhisperTranscriber<T>.FramesPerSecond, Tokens = tokens,
        };
        t.AddWordTimestamps(new List<WhisperTranscriptSegment> { segment }, language, window, frames, 0.0);
        return segment.Words.Select(w => (w.Word.Trim(), w.Start, w.End));
    }

    /// <summary>The release's <c>create_speaker_from_dict</c>: the recording's codes split among its timed words (the
    /// first word reaching 20 frames earlier and the last 20 later), each word's features from its own span.</summary>
    internal OuteTTSSpeaker SpeakerFromWords(double[] audio, string text, IReadOnlyList<(string Word, double Start, double End)> words)
    {
        int sr = _options.SampleRate;
        var codes = OuteTTSAudio.Encode<T>(CodecEncode, HopSize, audio, sr);
        int frames = codes.GetLength(1);
        const int tps = 75, maxExtension = 20;
        var speaker = new OuteTTSSpeaker { Text = text, GlobalFeatures = OuteTTSAudio.Features(audio, sr) };
        int? start = null;
        for (int i = 0; i < words.Count; i++)
        {
            var (word, wordStart, wordEnd) = words[i];
            if (start is null) start = Math.Max(0, (int)(wordStart * tps) - maxExtension);
            int end = i == words.Count - 1 ? Math.Min(frames, (int)(wordEnd * tps) + maxExtension) : (int)(wordEnd * tps);
            int from = Math.Min(start.Value, frames), to = Math.Max(from, Math.Min(end, frames));
            var c1 = Enumerable.Range(from, to - from).Select(t => codes[0, t]).ToArray();
            var c2 = Enumerable.Range(from, to - from).Select(t => codes[1, t]).ToArray();
            int a = Math.Min(audio.Length, (int)(wordStart * sr)), b = Math.Min(audio.Length, Math.Max(a, (int)(wordEnd * sr)));
            var wordAudio = new double[b - a];
            Array.Copy(audio, a, wordAudio, 0, b - a);
            start = end;
            speaker.Words.Add(new OuteTTSWord
            {
                Word = word, Duration = AiDotNet.Audio.Whisper.OpenAiWhisperTranscriber<T>.PythonRound(c1.Length / (double)tps, 2), C1 = c1, C2 = c2,
                Features = OuteTTSAudio.Features(wordAudio, sr),
            });
        }
        return speaker;
    }

    private OuteTTSSpeaker? VoiceSpeaker()
    {
        if (Speaker is not null) return Speaker;
        var reference = Voice?.ReferenceAudio;
        if (reference is null) return null;
        if (_voiceSpeaker is { } cached && ReferenceEquals(cached.Audio, reference)) return cached.Speaker;
        var speaker = CreateSpeaker(reference);
        _voiceSpeaker = (reference, speaker);
        return speaker;
    }

    // ---------------------------------------------------------------- synthesis

    private int EosToken => Tokenizer.TokenId("<|im_end|>");

    /// <inheritdoc />
    /// <remarks>The text's token ids in the language model's tokenizer (synthesis builds the prompt from their text).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private string TextOf(Tensor<T> ids) =>
        Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))), skipSpecial: true);

    /// <inheritdoc />
    /// <remarks>The release's chunked generation: the text in chunks of 10–30 words, each chunk's prompt continued until
    /// the end token, every chunk's codes decoded together.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var chunks = _options.Chunked ? OuteTTSPrompts.ChunkText(text, _options.ChunkMinWords, _options.ChunkMaxWords) : new List<string> { text };
        var speaker = VoiceSpeaker();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var generated = new List<int>();
        foreach (var chunk in chunks)
            generated.AddRange(Generate(Tokenizer.Encode(OuteTTSPrompts.CompletionPrompt(chunk, speaker)), random));
        return AudioOf(generated);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's token ids (<see cref="PreprocessText"/>); the waveform of the codes
    /// the model writes after its completion prompt with the current speaker (in one pass), empty when it writes none.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.VocabSize))
            throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        var prompt = Tokenizer.Encode(OuteTTSPrompts.CompletionPrompt(TextOf(input), VoiceSpeaker()));
        return AudioOf(Generate(prompt, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed)));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // HF generate() with the release's sampling: the windowed repetition penalty, temperature, top-k, top-p and min-p, until
    // the end token or the length limit; the generated tokens (the end token included).
    internal List<int> Generate(IReadOnlyList<int> prompt, Random random)
    {
        var o = _options;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var lm = LanguageModel;
            var ids = new List<int>(prompt);
            var generated = new List<int>();
            var cache = new LlamaCache<T>();
            var logits = lm.Forward(lm.Embed(Row(prompt)), null, null, cache, lastLogitsOnly: true).Logits!;
            int eos = EosToken;
            while (ids.Count < o.MaxLength)
            {
                var values = logits.ToArray();
                var scores = new double[values.Length];
                for (int i = 0; i < scores.Length; i++) scores[i] = NumOps.ToDouble(values[i]);
                int next;
                if (o.Temperature > 0)
                    next = OuteTTSSampler.Sample(OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, o.RepetitionRange, o.Temperature, o.TopK, o.TopP, o.MinP), random);
                else
                {
                    var processed = OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, o.RepetitionRange, 1.0, 0, 1.0, 0);
                    next = Array.IndexOf(processed, processed.Max());
                }
                ids.Add(next);
                generated.Add(next);
                if (next == eos) break;
                logits = lm.Forward(lm.Embed(Row(new[] { next })), null, null, cache, lastLogitsOnly: true).Logits!;
            }
            return generated;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    // extract_audio_from_tokens then the codec: each codebook's tokens in order, cut to the shorter.
    private Tensor<T> AudioOf(IReadOnlyList<int> tokens)
    {
        _ = Tokenizer;
        var c1 = tokens.Where(_c1!.ContainsKey).Select(t => _c1![t]).ToList();
        var c2 = tokens.Where(_c2!.ContainsKey).Select(t => _c2![t]).ToList();
        int frames = Math.Min(c1.Count, c2.Count);
        if (frames == 0) return new Tensor<T>(new[] { 0 });
        var codes = new int[2, frames];
        for (int t = 0; t < frames; t++)
        {
            codes[0, t] = c1[t];
            codes[1, t] = c2[t];
        }
        return Wave(OuteTTSAudio.Decode<T>(CodecDecode, codes, _options.SampleRate));
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads a released OuteTTS 1.0 model directory: <c>model.safetensors</c> (the Hugging Face <c>LlamaForCausalLM</c> or
    /// <c>Qwen3ForCausalLM</c> state dict) and, for Llama, its <c>tokenizer.json</c>.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string tokenizer = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(tokenizer)) LoadTokenizer(tokenizer);
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(directory, "*.safetensors"))
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        LanguageModel.LoadTorchWeights((name, shape) => weights.TryGetValue(name, out var v) ? v
            : throw new InvalidDataException($"The checkpoint has no tensor '{name}'."));
    }

    /// <summary>Loads the codec: descript-audio-codec's checkpoint (ibm-research/DAC.speech.v1.0's
    /// <c>weights_24khz_1.5kbps_v1.0.pth</c>).</summary>
    public void LoadCodecWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            string name = entry.Key.StartsWith("state_dict.", StringComparison.Ordinal) ? entry.Key.Substring("state_dict.".Length) : entry.Key;
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        LoadCodec(name => state.TryGetValue(name, out var v) ? v : null, state.Keys);
    }

    internal void LoadCodec(Func<string, double[]?> read, IEnumerable<string> names)
    {
        var o = _options;
        var used = AiDotNet.Audio.Effects.DAC<T>.LoadDescriptModules(Engine, _codecEncoder!, _codecDecoder!, _quantizer!, o.CodecEncoderRates.Length,
            o.CodecDecoderRates.Length, o.CodecCodebooks, o.CodecCodebookSize, o.CodecCodebookDim, read);
        var unused = names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The codec checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    /// <summary>Loads Whisper for speaker profiles: an openai-whisper checkpoint (its <c>model_state_dict</c>, such as
    /// <c>large-v3-turbo.pt</c>) and the alignment heads of the named release (<c>turbo</c> by default).</summary>
    public void LoadAlignerWeights(string path, string alignmentHeads = "turbo")
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            string name = entry.Key.StartsWith("model_state_dict.", StringComparison.Ordinal) ? entry.Key.Substring("model_state_dict.".Length) : entry.Key;
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        Aligner.LoadTorchWeights((name, shape) => state.TryGetValue(name, out var v) ? v : throw new InvalidDataException($"The checkpoint has no tensor '{name}'."));
        if (!OpenAiWhisperNetwork<T>.ReleasedAlignmentHeads.TryGetValue(alignmentHeads, out var dump))
            throw new ArgumentException($"Unknown Whisper release '{alignmentHeads}'.", nameof(alignmentHeads));
        Aligner.SetAlignmentHeads(dump);
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
    /// <remarks>A speaker profile of the recording with its transcript (<see cref="TtsTrainingSample{T}.Tokens"/>, the words
    /// timed by Whisper's alignment), the release's training prompt of it, and the causal language-model loss on the
    /// completion after <c>&lt;|audio_start|&gt;</c>, as TRL's SFTTrainer computes it on prompt–completion data.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var (ids, labels) = TrainingExample(sample);
        var loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(ids, labels), Optimizer());
        return loss;
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
            var (ids, labels) = TrainingExample(sample);
            return Loss(ids, labels)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Training prompts by recording (Whisper's alignment and the codec run once per recording and transcript).
    private readonly Dictionary<Tensor<T>, (string Transcript, int[] Ids, int[] Labels)> _examples = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    // The prompt and its labels: the prompt part (through "<|audio_start|>\n") is not scored.
    internal (int[] Ids, int[] Labels) TrainingExample(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording; set Audio.", nameof(sample));
        string transcript = TextOf(sample.Tokens);
        if (_examples.TryGetValue(audio, out var cached) && cached.Transcript == transcript) return (cached.Ids, cached.Labels);
        var speaker = CreateSpeaker(audio, transcript);
        string prompt = OuteTTSPrompts.TrainingPrompt(speaker);
        string header = prompt.Substring(0, prompt.IndexOf(OuteTTSPrompts.AudioStart, StringComparison.Ordinal) + OuteTTSPrompts.AudioStart.Length + 1);
        var promptIds = Tokenizer.Encode(header);
        var ids = Tokenizer.Encode(prompt);
        var labels = ids.Select((id, i) => i < promptIds.Length ? -100 : id).ToArray();
        if (_examples.Count > 64) _examples.Clear();
        _examples[audio] = (transcript, ids, labels);
        return (ids, labels);
    }

    internal Tensor<T> Loss(int[] ids, int[] labels)
    {
        var lm = LanguageModel;
        var row = Row(ids);
        var labelRow = new int[1, labels.Length];
        for (int i = 0; i < labels.Length; i++) labelRow[0, i] = labels[i];
        var (_, logits) = lm.ForwardWithLogits(lm.Embed(row));
        return lm.CausalLmLoss(logits, labelRow);
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
        if (o.TrainingSteps > 0)
        {
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, 0, o.TrainingSteps, 0.0, LinearWarmupScheduler.DecayMode.Linear);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the language model; the codec and Whisper are frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
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
            Name = _useNativeMode ? "OuteTTS-Native" : "OuteTTS-ONNX",
            Description = "OuteTTS 1.0: text-to-speech as language modeling over DAC codes (OuteAI, 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "OuteTTS";
        metadata.AdditionalInfo["Backbone"] = _options.Backbone.ToString();
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OuteTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
