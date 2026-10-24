using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Audio.Speaker;
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
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which CosyVoice network a training step trains (the recipe trains <c>llm</c>, then <c>flow</c>).</summary>
public enum CosyVoiceTrainingStage
{
    /// <summary>The text-to-speech-token language model (<c>TransformerLM</c>): next-token cross-entropy.</summary>
    LanguageModel,

    /// <summary>The speech-token-to-mel flow (<c>MaskedDiffWithXvec</c>): conditional flow matching.</summary>
    Flow,
}

/// <summary>
/// CosyVoice: multilingual TTS with supervised semantic tokens and conditional flow matching.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer Based on Supervised Semantic Tokens" (Du et al., 2024)</item>
/// <item>Code: FunAudioLLM/CosyVoice (Apache-2.0), CosyVoice-300M's <c>cosyvoice.yaml</c>.</item></list></para>
/// <para>Text is tokenized by Whisper's multilingual tiktoken encoding. A language model (a Conformer text encoder and
/// a causal Transformer) generates S3 speech tokens after <c>[sos, x-vector, text, task_id]</c>, by repetition-aware
/// sampling. A flow model encodes the tokens with a Conformer, stretches them to mel frames and samples an 80-band mel
/// by conditional flow matching with classifier-free guidance, conditioned on the x-vector and a prompt's mel. HiFT turns
/// the mel into 22.05 kHz audio. The S3 tokenizer (<c>speech_tokenizer_v1</c>) and CAM++ provide a prompt's speech
/// tokens and x-vector; they are frozen, as in the release.</para>
/// <para>Voices (<see cref="CreateVoice"/>): zero-shot, from a recording and its transcript (the LM continues the
/// prompt's text and tokens); cross-lingual, from a recording alone (only the flow reads the prompt); or a speaker
/// embedding alone (<c>sft</c>). Without a voice the x-vector is zero.</para>
/// <para><b>For Beginners:</b> CosyVoice is a multilingual text-to-speech model that can clone voices
/// without any training on the target speaker (zero-shot). It works by first converting text into
/// "semantic tokens" using a language model, then using conditional flow matching to generate
/// high-quality mel-spectrograms from those tokens.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new CosyVoice&lt;float&gt;(architecture, new CosyVoiceOptions());
/// model.LoadPretrainedWeights("pretrained_models/CosyVoice-300M");
/// model.Voice = model.CreateVoice(promptAudio, 16000, "The transcript of the prompt.");
/// var audio = model.Synthesize("Hello from CosyVoice.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "CosyVoice: A Scalable Multilingual Zero-Shot Text-to-Speech Synthesizer Based on Supervised Semantic Tokens",
    "https://arxiv.org/abs/2407.05407",
    Year = 2024,
    Authors = "Du et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Component = "language model", LearningRate = 1e-3, WarmupSteps = 2500,
    Schedule = LearningRateSchedulerType.NoamHoldAnnealing, HoldFraction = 0, DecayRate = 0.5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice examples/libritts/cosyvoice/conf/cosyvoice.yaml train_conf: Adam at 1e-3 under "
        + "WarmupLR (2,500 warmup steps, then inverse-square-root decay: NoamHoldAnnealing with no hold and power 0.5), "
        + "gradient clipping at 5 and 2-step accumulation; run.sh trains llm and flow with it in turn.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "flow", LearningRate = 1e-3, WarmupSteps = 2500,
    Schedule = LearningRateSchedulerType.NoamHoldAnnealing, HoldFraction = 0, DecayRate = 0.5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "FunAudioLLM/CosyVoice examples/libritts/cosyvoice/conf/cosyvoice.yaml train_conf, the flow's run of the "
        + "same recipe as the language model's.")]
public partial class CosyVoice<T> : TtsModelBase<T>, ICodecTts<T>
{
    // S3 and CAM++ read 16 kHz audio (load_wav(prompt_wav, 16000)).
    private const int PromptSampleRate = 16000;

    private readonly CosyVoiceOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private readonly Dictionary<CosyVoiceTrainingStage, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _stageOptimizers = new();
    private readonly Random _trainingRandom;
    private readonly TiktokenEncoding _text;
    private bool _useNativeMode;
    private bool _disposed;

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _flowLayers = new();
    private readonly List<LayerBase<T>> _hiftLayers = new();
    private readonly List<LayerBase<T>> _speechTokenizerLayers = new();
    private readonly List<LayerBase<T>> _speakerEncoderLayers = new();
    private CosyVoiceLm<T>? _lm;
    private CosyVoiceFlow<T>? _flow;
    private HiftGenerator<T>? _hift;
    private S3Tokenizer<T>? _speechTokenizer;
    private CamPlusPlusEncoder<T>? _speakerEncoder;
    private KaldiFbank<T>? _fbank;
    private DifferentiableMel<T>? _mel;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public CosyVoice(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        CosyVoiceOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new CosyVoiceOptions();
        _useNativeMode = false;
        _text = WhisperTiktoken.Multilingual(_options.TextLanguages);
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">An optimizer for both stages instead of the recipe's per-stage Adam.</param>
    public CosyVoice(
        NeuralNetworkArchitecture<T> architecture,
        CosyVoiceOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new CosyVoiceOptions();
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _text = WhisperTiktoken.Multilingual(_options.TextLanguages);
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.NumCodebooks != 1)
            throw new ArgumentException("CosyVoice speaks one stream of S3 tokens; NumCodebooks must be 1.", nameof(o.NumCodebooks));
        // The text vocabulary is the tokenizer's, as text_token_size is.
        o.VocabSize = _text.VocabularySize;
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.LLMDim;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => _options.NumCodebooks;
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <summary>The network a training step trains; the recipe trains the language model, then the flow.</summary>
    public CosyVoiceTrainingStage CurrentStage { get; set; } = CosyVoiceTrainingStage.LanguageModel;

    /// <inheritdoc />
    /// <remarks>The S3 tokens of the utterance and the speaker: its x-vector, or a recording CAM++ embeds.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens | TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => 1;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.CodebookSize;

    private bool HasPaperLayers => _lm is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_text.VocabularySize) : base.GetInputDomain(inputShape);

    private CosyVoiceLm<T> Lm => _lm ?? throw new InvalidOperationException("CosyVoice's language model exists in native mode only.");
    private CosyVoiceFlow<T> Flow => _flow ?? throw new InvalidOperationException("CosyVoice's flow exists in native mode only.");
    private HiftGenerator<T> Hift => _hift ?? throw new InvalidOperationException("CosyVoice's vocoder exists in native mode only.");
    private S3Tokenizer<T> SpeechTokenizer => _speechTokenizer ?? throw new InvalidOperationException("CosyVoice's speech tokenizer exists in native mode only.");
    private CamPlusPlusEncoder<T> SpeakerEncoder => _speakerEncoder ?? throw new InvalidOperationException("CosyVoice's speaker encoder exists in native mode only.");

    // The networks, for tests.
    internal CosyVoiceLm<T>? LanguageModel => _lm;
    internal CosyVoiceFlow<T>? FlowModel => _flow;
    internal HiftGenerator<T>? Vocoder => _hift;
    internal IReadOnlyList<LayerBase<T>> LanguageModelLayers => _lmLayers;
    internal IReadOnlyList<LayerBase<T>> FlowLayers => _flowLayers;
    internal IReadOnlyList<LayerBase<T>> FrozenLayers => _hiftLayers.Concat(_speechTokenizerLayers).Concat(_speakerEncoderLayers).ToList();

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        // TransformerLM: a Conformer text encoder (linear input) and a Transformer (linear_legacy input), both causal
        // (static_chunk_size 1) with ESPnet relative positions; no attention dropout.
        _lm = new CosyVoiceLm<T>(Engine, _lmLayers, new CosyVoiceLmConfiguration(
            TextVocabulary: _text.VocabularySize, SpeechVocabulary: o.CodebookSize, TextEncoderInputSize: o.TextEncoderDim,
            LlmSize: o.LLMDim, SpeakerDim: o.SpeakerEmbeddingDim,
            TextEncoder: new WenetEncoderConfiguration(o.TextEncoderDim, o.LLMDim, o.NumHeads, o.LmFeedForwardDim, o.NumEncoderLayers,
                WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: 1, Dropout: o.DropoutRate,
                PositionalDropout: o.DropoutRate),
            Llm: new WenetEncoderConfiguration(o.LLMDim, o.LLMDim, o.NumHeads, o.LmFeedForwardDim, o.NumLLMLayers,
                WenetInputLayer.LinearLegacy, Swish: false, Conformer: false, StaticChunkSize: 1, Dropout: o.DropoutRate,
                PositionalDropout: o.DropoutRate),
            TopP: o.TopP, TopK: o.TopK, RepetitionWindow: o.RepetitionWindow, RepetitionRatio: o.RepetitionRatio), random);

        // MaskedDiffWithXvec: a Conformer encoder, InterpolateRegulator's four stages and ConditionalCFM's U-Net.
        _flow = new CosyVoiceFlow<T>(Engine, _flowLayers, new CosyVoiceFlowConfiguration(
            VocabSize: o.CodebookSize, InputSize: o.FlowDim, OutputSize: o.MelChannels, SpeakerDim: o.SpeakerEmbeddingDim,
            InputFrameRate: o.CodecFrameRate, SampleRate: o.SampleRate, HopSize: o.HopSize,
            Encoder: new WenetEncoderConfiguration(o.FlowDim, o.FlowDim, o.FlowHeads, o.FlowFeedForwardDim, o.FlowEncoderLayers,
                WenetInputLayer.Linear, Swish: true, Conformer: true, Dropout: o.DropoutRate, PositionalDropout: o.DropoutRate,
                AttentionDropout: o.FlowAttentionDropoutRate),
            RegulatorStages: 4, DecoderChannels: (int[])o.DecoderChannels.Clone(), DecoderHeads: o.DecoderHeads,
            DecoderHeadDim: o.DecoderHeadDim, DecoderBlocksPerLevel: o.DecoderBlocks, DecoderMidBlocks: o.DecoderMidBlocks,
            SigmaMin: 1e-6, TrainingCfgRate: o.TrainingCfgRate, InferenceCfgRate: o.InferenceCfgRate, Timesteps: o.FlowTimesteps,
            Loss: o.FlowLoss), random);

        // HiFTGenerator with ConvRNNF0Predictor: upsampling 8·8 and an iSTFT of 16 with hop 4, 256 samples per frame.
        var hift = new HiftConfiguration(InChannels: o.MelChannels, BaseChannels: o.HiftBaseChannels, Harmonics: o.HiftHarmonics,
            SampleRate: o.SampleRate, NsfAlpha: 0.1, NsfSigma: 0.003, VoicedThreshold: 10, UpsampleRates: new[] { 8, 8 },
            UpsampleKernels: new[] { 16, 16 }, FftSize: 16, HopSize: 4, ResblockKernels: new[] { 3, 7, 11 },
            ResblockDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            SourceKernels: new[] { 7, 11 }, SourceDilations: new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            LeakySlope: 0.1, AudioLimit: 0.99, F0Channels: o.HiftBaseChannels);
        if (hift.UpsampleScale != o.HopSize)
            throw new ArgumentException($"HiFT gives {hift.UpsampleScale} samples per mel frame; HopSize must be {hift.UpsampleScale}.");
        _hift = new HiftGenerator<T>(Engine, _hiftLayers, hift, random);

        // The frozen front end: speech_tokenizer_v1 (S3 at 50 Hz) and the CAM++ x-vector extractor.
        _speechTokenizer = new S3Tokenizer<T>(Engine, _speechTokenizerLayers, new S3TokenizerConfiguration(
            State: o.SpeechTokenizerState, Heads: o.SpeechTokenizerHeads, Layers: o.SpeechTokenizerLayers,
            CodebookSize: o.CodebookSize), random);
        _speakerEncoder = new CamPlusPlusEncoder<T>(Engine, _speakerEncoderLayers, new CamPlusPlusConfiguration(
            EmbeddingSize: o.SpeakerEmbeddingDim, GrowthRate: o.SpeakerEncoderGrowthRate, InitialChannels: o.SpeakerEncoderChannels),
            random);
        _speakerEncoder.SetTrainingMode(false);

        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_flowLayers);
        ComponentLayers.AddRange(_hiftLayers);
        ComponentLayers.AddRange(_speechTokenizerLayers);
        ComponentLayers.AddRange(_speakerEncoderLayers);
    }

    /// <inheritdoc />
    /// <remarks>The frozen S3 tokenizer and CAM++ stay in evaluation mode.</remarks>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        _speakerEncoder?.SetTrainingMode(false);
        foreach (var layer in _speakerEncoderLayers) layer.SetTrainingMode(false);
    }

    // ---------------------------------------------------------------- front end

    /// <summary>The text token ids of <paramref name="text"/>: Whisper's multilingual tiktoken encoding with special
    /// tokens written in the text allowed (<c>encode(text, allowed_special='all')</c>).</summary>
    public int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return _text.Encode(text);
    }

    /// <summary>The text of token ids.</summary>
    public string Detokenize(IEnumerable<int> ids) => _text.Decode(ids);

    /// <summary>The sentences CosyVoice synthesizes <paramref name="text"/> as (<c>text_normalize(text, split=True)</c>
    /// without an external normalizer): see <see cref="CosyVoiceTextFrontEnd"/>.</summary>
    public IReadOnlyList<string> SplitText(string text) => CosyVoiceTextFrontEnd.Sentences(text, s => Tokenize(s).Length);

    /// <inheritdoc />
    /// <remarks>As <c>inference_zero_shot</c> and its siblings do: each sentence of <see cref="SplitText"/> is
    /// synthesized in <see cref="TtsModelBase{T}.Voice"/> and the waveforms are concatenated. Text of punctuation alone
    /// gives no audio.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (text.Trim().Length == 0) throw new ArgumentException("There is no text to synthesize.", nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        var pieces = SplitText(text).Select(sentence => PostprocessAudio(Predict(PreprocessText(sentence)))).ToArray();
        return pieces.Length switch
        {
            0 => new Tensor<T>(new[] { 0 }),
            1 => pieces[0],
            _ => Engine.TensorConcatenate(pieces, 0),
        };
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        var ids = Tokenize(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return Ids(ids);
    }

    /// <summary>
    /// The S3 speech tokens (50 Hz) of a recording <c>[samples]</c> at <paramref name="sampleRate"/>
    /// (<c>_extract_speech_token</c>): resampled to 16 kHz, Whisper's 128-band log-mel, S3Tokenizer v1.
    /// </summary>
    public int[] SpeechTokens(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        var audio = Resample(recording, sampleRate, PromptSampleRate);
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(audio[i]);
        var logMel = S3Tokenizer<T>.LogMel(samples);
        int bands = logMel.GetLength(0), frames = logMel.GetLength(1);
        var mel = new Tensor<T>(new[] { bands, frames });
        for (int m = 0; m < bands; m++)
            for (int f = 0; f < frames; f++) mel[m, f] = NumOps.FromDouble(logMel[m, f]);
        return SpeechTokenizer.Tokenize(mel);
    }

    /// <summary>
    /// The CAM++ x-vector <c>[SpeakerEmbeddingDim]</c> of a recording (<c>_extract_spk_embedding</c>): resampled to
    /// 16 kHz, 80-band Kaldi filterbank energies minus their utterance mean, CAM++ in evaluation mode.
    /// </summary>
    public Tensor<T> SpeakerEmbedding(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        using var _ = new NoGradScope<T>();
        var audio = Resample(recording, sampleRate, PromptSampleRate);
        _fbank ??= new KaldiFbank<T>(Engine, PromptSampleRate, 80, WindowType.Povey);
        var features = _fbank.Forward(audio);
        features = Engine.TensorSubtract(features, Engine.TensorBroadcastTo(Engine.ReduceMean(features, new[] { 0 }, true), features.Shape.ToArray()));
        var embedding = SpeakerEncoder.Forward(Engine.Reshape(features, new[] { 1, features.Shape[0], features.Shape[1] }));
        return Engine.Reshape(embedding, new[] { _options.SpeakerEmbeddingDim });
    }

    /// <summary>
    /// The flow's mel <c>[frames, MelChannels]</c> of a recording at the model's sample rate
    /// (<c>_extract_speech_feat</c>: Matcha's <c>mel_spectrogram</c>, n_fft 1024, hop 256, window 1024, 0–8 kHz,
    /// not centred).
    /// </summary>
    public Tensor<T> MelSpectrogram(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var o = _options;
        _mel ??= new DifferentiableMel<T>(Engine, o.SampleRate, o.FftSize, o.HopSize, o.FftSize, o.MelChannels, 0, 8000);
        return _mel.Forward(Engine.Reshape(audio, new[] { audio.Length }));
    }

    /// <summary>
    /// A voice for <see cref="TtsModelBase{T}.Voice"/> from a recording of the speaker <c>[samples]</c> at
    /// <paramref name="sampleRate"/>: its S3 tokens, its mel (resampled to the model's rate) and its CAM++ x-vector.
    /// With <paramref name="transcript"/> the voice is zero-shot (<c>frontend_zero_shot</c>): the language model
    /// continues the prompt's text and tokens. Without it, cross-lingual (<c>frontend_cross_lingual</c>): only the flow
    /// reads the prompt.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? transcript = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        var tokens = SpeechTokens(recording, sampleRate);
        Tensor<T> mel;
        using (new NoGradScope<T>()) mel = MelSpectrogram(Resample(recording, sampleRate, _options.SampleRate));
        return new TtsVoice<T>
        {
            Reference = mel,
            ReferenceCodecTokens = Ids(tokens),
            // inference_zero_shot normalizes the prompt's transcript unsplit.
            ReferenceTokens = transcript is null ? null : Ids(Tokenize(CosyVoiceTextFrontEnd.Normalize(transcript))),
            SpeakerEmbedding = SpeakerEmbedding(recording, sampleRate),
        };
    }

    /// <summary>A voice of a speaker embedding alone (<c>frontend_sft</c>: a speaker of <c>spk2info</c>).</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> speakerEmbedding)
    {
        if (speakerEmbedding is null) throw new ArgumentNullException(nameof(speakerEmbedding));
        if (speakerEmbedding.Length != _options.SpeakerEmbeddingDim)
            throw new ArgumentException($"A CosyVoice speaker embedding has {_options.SpeakerEmbeddingDim} values.", nameof(speakerEmbedding));
        return new TtsVoice<T> { SpeakerEmbedding = speakerEmbedding };
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : AudioHelper<T>.ResampleBandLimited(Engine, flat, from, to);
    }

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    private int[] IdsOf(Tensor<T> tensor)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
        return ids;
    }

    // ---------------------------------------------------------------- synthesis

    /// <summary>
    /// Generates speech tokens for text ids under <see cref="TtsModelBase{T}.Voice"/> (<c>llm.inference</c>):
    /// prompted by the voice's transcript and tokens in zero-shot, by its x-vector alone otherwise.
    /// </summary>
    public int[] GenerateSpeechTokens(IReadOnlyList<int> text, Random random)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var voice = Voice;
        bool zeroShot = voice?.ReferenceTokens is not null && voice.ReferenceCodecTokens is not null;
        var promptText = zeroShot && voice?.ReferenceTokens is { } transcript ? IdsOf(transcript) : Array.Empty<int>();
        var promptSpeech = zeroShot && voice?.ReferenceCodecTokens is { } prompt ? IdsOf(prompt) : Array.Empty<int>();
        return Lm.Generate(text, promptText, promptSpeech, voice?.SpeakerEmbedding, random).ToArray();
    }

    /// <summary>
    /// The mel <c>[1, MelChannels, frames]</c> of speech tokens (<c>MaskedDiffWithXvec.inference</c>): the voice's
    /// prompt tokens and these encoded together, each part stretched to its frames (the prompt's to its mel's), the
    /// prompt's mel as the condition, Euler sampling from Gaussian noise, and the prompt's frames removed.
    /// </summary>
    public Tensor<T> TokensToMel(IReadOnlyList<int> tokens, Random random)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        var flow = Flow;
        var voice = Voice;
        var promptTokens = voice?.ReferenceCodecTokens is { } codes ? IdsOf(codes) : Array.Empty<int>();
        var promptMel = promptTokens.Length > 0
            ? voice?.Reference ?? throw new InvalidOperationException("A voice with prompt tokens needs the prompt's mel (Reference) for the flow.")
            : null;
        int frames = (promptMel?.Shape[0] ?? 0) + flow.MelFrames(tokens.Count);
        // ConditionalCFM.forward: z = randn_like(mu) at temperature 1.
        var noise = new Tensor<T>(new[] { 1, _options.MelChannels, frames });
        for (int i = 0; i < noise.Length; i++)
            noise[i] = NumOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble()));
        var xvector = voice?.SpeakerEmbedding ?? new Tensor<T>(new[] { _options.SpeakerEmbeddingDim });
        return flow.Inference(tokens, promptTokens, promptMel, xvector, noise);
    }

    /// <summary>The waveform <c>[samples]</c> of a mel <c>[1, MelChannels, frames]</c> (<c>HiFTGenerator.inference</c>).</summary>
    public Tensor<T> MelToAudio(Tensor<T> mel, Random random)
    {
        if (mel is null) throw new ArgumentNullException(nameof(mel));
        var audio = Hift.Forward(mel, random);
        return Engine.Reshape(audio, new[] { audio.Length });
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is text token ids (<see cref="PreprocessText"/>); the waveform <c>[samples]</c>
    /// comes from the language model's tokens, the flow's mel and HiFT, each sampled from a generator seeded by
    /// <see cref="CosyVoiceOptions.SamplingSeed"/>.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var text = IdsOf(input);
            if (text.Length == 0) throw new ArgumentException("There is no text to synthesize.", nameof(input));
            if (text.Any(id => id < 0 || id >= _text.VocabularySize))
                throw new ArgumentOutOfRangeException(nameof(input), $"Text ids lie in [0, {_text.VocabularySize}).");
            var tokens = GenerateSpeechTokens(text, random);
            return MelToAudio(TokensToMel(tokens, random), random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <summary>Encodes audio at the model's sample rate into S3 speech tokens <c>[frames]</c>.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        return Ids(SpeechTokens(audio, _options.SampleRate));
    }

    /// <summary>Decodes S3 speech tokens into audio through the flow and HiFT, in <see cref="TtsModelBase{T}.Voice"/>.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        using var _ = new NoGradScope<T>();
        return MelToAudio(TokensToMel(IdsOf(tokens), random), random);
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample, _trainingRandom, training: true);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), StageOptimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed), training: false);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The stage's objective on one utterance. The language model's: the mean cross-entropy of the speech tokens and
    /// the end (<c>TransformerLM.forward</c>). The flow's: conditional flow matching on the utterance's mel with a
    /// random prompt prefix and condition dropout (<c>MaskedDiffWithXvec.forward</c>). Speech tokens come from
    /// <see cref="TtsTrainingSample{T}.CodecTokens"/> or the recording, the x-vector from
    /// <see cref="TtsTrainingSample{T}.SpeakerReference"/> or the recording, the mel from
    /// <see cref="TtsTrainingSample{T}.Mel"/> (<c>[frames, MelChannels]</c>) or the recording.
    /// </summary>
    internal Func<Tensor<T>> BuildObjective(TtsTrainingSample<T> sample, Random random, bool training)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The objective needs CosyVoice's layers; this model was built from caller-supplied layers.");
        var text = IdsOf(sample.Tokens);
        if (text.Any(id => id < 0 || id >= _text.VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(sample), $"Text ids lie in [0, {_text.VocabularySize}).");
        var speech = SpeechTokensOf(sample);
        var xvector = XVectorOf(sample);

        if (CurrentStage == CosyVoiceTrainingStage.LanguageModel)
            return () => Lm.Loss(text, speech, xvector, training, random);

        Tensor<T> mel;
        if (sample.Mel is { } given)
        {
            if (given.Rank != 2 || given.Shape[1] != _options.MelChannels)
                throw new ArgumentException($"Expected a mel [frames, {_options.MelChannels}].", nameof(sample));
            mel = given;
        }
        else
        {
            var recording = sample.Audio ?? throw new ArgumentException("The flow trains on the utterance's mel; set Mel or Audio.", nameof(sample));
            using (new NoGradScope<T>()) mel = MelSpectrogram(recording);
        }
        var target = Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, _options.MelChannels, mel.Shape[0] });
        return () => Flow.TrainingLoss(speech, target, xvector, training, random);
    }

    // The utterance's x-vector: SpeakerReference when it is one ([SpeakerEmbeddingDim]), else CAM++'s of the recording
    // SpeakerReference or Audio holds, at the model's sample rate, as the recipe's data preparation extracts it.
    private Tensor<T> XVectorOf(TtsTrainingSample<T> sample)
    {
        if (sample.SpeakerReference is { Rank: 1 } reference && reference.Length == _options.SpeakerEmbeddingDim)
            return reference;
        var recording = sample.SpeakerReference ?? sample.Audio
            ?? throw new ArgumentException("CosyVoice trains with the speaker's x-vector; set SpeakerReference (an x-vector or a recording) or Audio.", nameof(sample));
        return SpeakerEmbedding(recording, _options.SampleRate);
    }

    private int[] SpeechTokensOf(TtsTrainingSample<T> sample)
    {
        if (sample.CodecTokens is { } tokens)
        {
            var ids = IdsOf(tokens);
            if (ids.Any(id => id < 0 || id >= _options.CodebookSize))
                throw new ArgumentOutOfRangeException(nameof(sample), $"S3 tokens lie in [0, {_options.CodebookSize}).");
            return ids;
        }
        var audio = sample.Audio ?? throw new ArgumentException("CosyVoice trains on S3 speech tokens, or on the recording.", nameof(sample));
        return SpeechTokens(audio, _options.SampleRate);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? StageOptimizer()
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (!_stageOptimizers.TryGetValue(CurrentStage, out var optimizer))
        {
            optimizer = PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this,
                    CurrentStage == CosyVoiceTrainingStage.LanguageModel ? "language model" : "flow", _options.WarmupSteps)
                ?? throw new InvalidOperationException("CosyVoice declares an optimizer for each stage.");
            _stageOptimizers[CurrentStage] = optimizer;
        }
        return optimizer;
    }

    /// <inheritdoc />
    /// <remarks>A stage updates its own network only; HiFT, the S3 tokenizer and CAM++ stay as loaded.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var layers = CurrentStage == CosyVoiceTrainingStage.LanguageModel ? _lmLayers : _flowLayers;
        var selected = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(layers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToList();
    }

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads the released networks from a CosyVoice-300M directory, each a state dict converted to safetensors:
    /// <c>llm.safetensors</c>, <c>flow.safetensors</c> and <c>hift.safetensors</c> (from <c>llm.pt</c>, <c>flow.pt</c>,
    /// <c>hift.pt</c>), <c>speech_tokenizer_v1.safetensors</c> (xingchensong/S3Tokenizer's PyTorch form of the ONNX
    /// tokenizer) and <c>campplus.safetensors</c> (3D-Speaker's CAM++).
    /// </summary>
    public void LoadPretrainedWeights(string modelDirectory)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(modelDirectory)) throw new ArgumentException("The model directory is required.", nameof(modelDirectory));
        Lm.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "llm.safetensors")), "");
        Flow.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "flow.safetensors")), "");
        Hift.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "hift.safetensors")));
        SpeechTokenizer.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "speech_tokenizer_v1.safetensors")));
        SpeakerEncoder.LoadTorchWeights(Reader(Path.Combine(modelDirectory, "campplus.safetensors")));
        InvalidateWeightCachesAfterSuccessfulWeightUpdate();
    }

    private static Func<string, int[], double[]> Reader(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"The checkpoint {path} is missing.", path);
        var file = AiDotNet.Agentic.Models.Local.SafetensorsReader.Read(File.ReadAllBytes(path));
        return (name, shape) =>
        {
            var entry = file.Get(name) ?? throw new InvalidDataException($"The checkpoint {path} has no tensor '{name}'.");
            // num_batches_tracked is an int64 scalar that nothing reads.
            if (entry.DataType == "I64") return new[] { (double)BitConverter.ToInt64(file.GetRawBytes(name), 0) };
            var values = file.ReadAsDouble(name);
            if (values.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new InvalidDataException($"'{name}' has {values.Length} values in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return values;
        };
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CosyVoice-Native" : "CosyVoice-ONNX",
            Description =
                "CosyVoice: Multilingual TTS with Supervised Semantic Tokens (Du et al., 2024)",
            FeatureCount = _options.LLMDim,
            Complexity = _options.NumLLMLayers,
        };
        m.AdditionalInfo["Architecture"] = "CosyVoice";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(CosyVoice<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
