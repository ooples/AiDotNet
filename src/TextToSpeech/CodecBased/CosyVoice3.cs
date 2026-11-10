using AiDotNet.ModelLoading;
using AiDotNet.Attributes;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Fun-CosyVoice 3: Qwen2 speech-token generation, S3-v3 prompt tokens, causal DiT flow matching and CausalHiFT.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// This implements the released <c>Fun-CosyVoice3-0.5B-2512</c> topology. PyTorch state dictionaries are loaded after
/// conversion to safetensors; the frozen prompt tokenizer remains the release's <c>speech_tokenizer_v3.onnx</c>.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training",
    "https://arxiv.org/abs/2505.17589",
    Year = 2025,
    Authors = "Zhihao Du et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Component = "language model", LearningRate = 1e-5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "FunAudioLLM/CosyVoice training configuration: constant 1e-5 Adam, clip 5, accumulation 2.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Component = "flow", LearningRate = 1e-5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "FunAudioLLM/CosyVoice training configuration for CausalMaskedDiffWithDiT.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Component = "vocoder", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode, Source = "FunAudioLLM/CosyVoice HiFT GAN recipe.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, Component = "discriminator", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode, Source = "FunAudioLLM/CosyVoice HiFT discriminator recipe.")]
public partial class CosyVoice3<T> : CosyVoiceBase<T>
{
    private const string DefaultSystemPrompt = "You are a helpful assistant.<|endofprompt|>";
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly CosyVoice3Options _options;
    private readonly QwenTokenizer _text = QwenTokenizer.CosyVoice3;
    private CosyVoice2Lm<T>? _lm;
    private CosyVoice3Flow<T>? _flow;
    private HiftGenerator<T>? _vocoder;
    private OnnxModel<T>? _speechTokenizerOnnx;
    private OnnxModel<T>? _speakerEncoderOnnx;
    private CosyVoiceFlowConfiguration? _flowContract;

    /// <summary>Creates a whole-model ONNX inference instance.</summary>
    public CosyVoice3(NeuralNetworkArchitecture<T> architecture, string modelPath, CosyVoice3Options? options = null)
        : base(architecture, (options ?? new CosyVoice3Options()).SamplingSeed, modelPath,
            (options ?? new CosyVoice3Options()).OnnxOptions)
    {
        _options = options ?? new CosyVoice3Options();
        Options = _options;
        Validate();
        Initialize();
    }

    /// <summary>Creates the native released networks for training or checkpoint-backed inference.</summary>
    public CosyVoice3(NeuralNetworkArchitecture<T> architecture, CosyVoice3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, (options ?? new CosyVoice3Options()).SamplingSeed, optimizer)
    {
        _options = options ?? new CosyVoice3Options();
        Options = _options;
        Validate();
        Initialize();
        if (!string.IsNullOrWhiteSpace(_options.SpeechTokenizerPath)) OpenSpeechTokenizer(_options.SpeechTokenizerPath!);
        if (!string.IsNullOrWhiteSpace(_options.SpeakerEncoderOnnxPath)) OpenSpeakerEncoder(_options.SpeakerEncoderOnnxPath!);
    }

    private void Validate()
    {
        if (_options.VocabSize < _text.VocabularySize)
            throw new ArgumentException($"VocabSize ({_options.VocabSize}) must cover the tokenizer's {_text.VocabularySize} ids.");
        if (_options.ReservedSpeechTokens < 4)
            throw new ArgumentException("ReservedSpeechTokens must include SOS, EOS, task and fill ids.");
        if (_options.TokenMelRatio != 2)
            throw new ArgumentException("The released CosyVoice3 flow uses exactly two mel frames per S3 token.");
        if (_options.DiTHeads * _options.DiTHeadDim != _options.DiTDim)
            throw new ArgumentException("DiTHeads times DiTHeadDim must equal DiTDim.");
    }

    protected override CosyVoiceFamilyOptions Settings => _options;
    protected override int TextVocabularySize => _options.VocabSize;
    protected override double MelMaximumFrequency => _options.SampleRate / 2.0;

    public override int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return _text.Encode(text);
    }

    public override string Detokenize(IEnumerable<int> ids) => _text.Decode(ids);

    // The native S3 network is deliberately not built: the v3 release distributes and invokes this frozen encoder as ONNX.
    private protected override S3TokenizerConfiguration SpeechTokenizerConfiguration => S3TokenizerConfiguration.V2Default;

    // This contract supplies the shared voice/training orchestration with causal trimming. BuildFlow installs the DiT flow.
    private protected override CosyVoiceFlowConfiguration FlowConfiguration => _flowContract ??= new CosyVoiceFlowConfiguration(
        VocabSize: _options.CodebookSize, InputSize: _options.DiTInputDim, OutputSize: _options.MelChannels,
        SpeakerDim: _options.SpeakerEmbeddingDim, InputFrameRate: _options.CodecFrameRate, SampleRate: _options.SampleRate,
        HopSize: _options.HopSize,
        Encoder: new WenetEncoderConfiguration(_options.DiTInputDim, _options.DiTInputDim, 8, 4 * _options.DiTInputDim,
            0, WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: _options.StaticChunkSize),
        RegulatorStages: 0, DecoderChannels: Array.Empty<int>(), DecoderHeads: _options.DiTHeads,
        DecoderHeadDim: _options.DiTHeadDim, DecoderBlocksPerLevel: 0, DecoderMidBlocks: 0,
        SigmaMin: _options.DiTSigmaMin, TrainingCfgRate: _options.DiTTrainingCfgRate,
        InferenceCfgRate: _options.DiTInferenceCfgRate, Timesteps: _options.DiTTimesteps,
        Loss: CosyVoiceFlowLoss.SquaredError, Causal: true, TokenMelRatio: _options.TokenMelRatio,
        Lookahead: _options.Lookahead);

    private CosyVoice2Lm<T> Lm => _lm ?? throw new InvalidOperationException("The native language model is unavailable.");
    private CosyVoice3Flow<T> DiTFlow => _flow ?? throw new InvalidOperationException("The native DiT flow is unavailable.");
    private HiftGenerator<T> CausalHift => _vocoder ?? throw new InvalidOperationException("The native CausalHiFT is unavailable.");

    internal CosyVoice2Lm<T>? LanguageModel => _lm;
    internal CosyVoice3Flow<T>? DiTFlowModel => _flow;
    internal HiftGenerator<T>? CausalVocoder => _vocoder;

    protected override void BuildLanguageModel(List<LayerBase<T>> layers, Random initialization)
    {
        var o = _options;
        _lm = new CosyVoice2Lm<T>(Engine, layers, new CosyVoice2LmConfiguration(new LlamaConfiguration
        {
            VocabSize = o.VocabSize,
            Hidden = o.LLMDim,
            Layers = o.NumLLMLayers,
            Heads = o.NumHeads,
            KeyValueHeads = o.KeyValueHeads,
            Intermediate = o.LmFeedForwardDim,
            RmsEpsilon = o.RmsEpsilon,
            RopeTheta = o.RopeTheta,
            AttentionBias = true,
            IncludeHead = false,
        }, o.CodebookSize, TopP: o.TopP, TopK: o.TopK, RepetitionWindow: o.RepetitionWindow,
            RepetitionRatio: o.RepetitionRatio, SpeechSpecialTokens: o.ReservedSpeechTokens,
            SpecialsInSpeechEmbedding: true, BiasFreeDecoder: true), initialization);
    }

    protected override void BuildFlow(List<LayerBase<T>> layers, Random initialization) =>
        _flow = new CosyVoice3Flow<T>(Engine, layers, _options, initialization);

    protected override void BuildVocoder(List<LayerBase<T>> layers, Random initialization)
    {
        int stages = _options.HiftUpsampleRates.Length;
        var configuration = new HiftConfiguration(_options.MelChannels, _options.HiftBaseChannels, _options.HiftHarmonics,
            _options.SampleRate, 0.1, 0.003, 10, (int[])_options.HiftUpsampleRates.Clone(),
            (int[])_options.HiftUpsampleKernels.Clone(), 16, 4, new[] { 3, 7, 11 },
            new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } },
            (int[])_options.HiftSourceKernels.Clone(), Enumerable.Range(0, stages).Select(_ => new[] { 1, 3, 5 }).ToArray(),
            0.1, 0.99, _options.HiftBaseChannels, Causal: true, ConvPreLookRight: 4);
        if (configuration.UpsampleScale != _options.HopSize)
            throw new ArgumentException($"CausalHiFT produces {configuration.UpsampleScale} samples per mel frame.");
        _vocoder = new HiftGenerator<T>(Engine, layers, configuration, initialization);
    }

    protected override void BuildSpeechTokenizer(List<LayerBase<T>> layers, Random initialization)
    {
        // speech_tokenizer_v3.onnx is opened when a release directory (or explicit path) is supplied.
    }

    private int[] EnsureSystemPrompt(IReadOnlyList<int> text)
    {
        int marker = _text.AddedTokenId("<|endofprompt|>");
        return text.Contains(marker) ? text.ToArray() : _text.Encode(DefaultSystemPrompt).Concat(text).ToArray();
    }

    protected override int[] GenerateWithLanguageModel(IReadOnlyList<int> text, IReadOnlyList<int> promptText,
        IReadOnlyList<int> promptSpeech, Tensor<T>? xvector, Random random)
    {
        // Zero-shot puts the system marker in the prompt transcript; cross-lingual/SFT puts it in the target text.
        var preparedPrompt = promptSpeech.Count > 0 ? EnsureSystemPrompt(promptText) : promptText.ToArray();
        var preparedText = promptSpeech.Count > 0 ? text.ToArray() : EnsureSystemPrompt(text);
        return Lm.Generate(preparedText, preparedPrompt, promptSpeech, random).ToArray();
    }

    protected override Tensor<T> LanguageModelLoss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> xvector,
        bool training, Random random) => Lm.Loss(EnsureSystemPrompt(text), speech, random);

    private protected override void LoadLanguageModel(TorchTensorSource source) => Lm.LoadTorchWeights(source, "");

    public override Tensor<T> TokensToMel(IReadOnlyList<int> tokens, Random random)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Count == 0) throw new ArgumentException("There are no speech tokens to decode.", nameof(tokens));
        var promptTokens = Voice?.ReferenceCodecTokens is { } codes ? IdsOf(codes) : Array.Empty<int>();
        var promptMel = promptTokens.Length > 0
            ? Voice?.Reference ?? throw new InvalidOperationException("Prompt speech tokens require the prompt mel.")
            : null;
        int frames = (promptMel?.Shape[0] ?? 0) + DiTFlow.MelFrames(tokens.Count);
        var noise = new Tensor<T>(new[] { 1, _options.MelChannels, frames });
        for (int i = 0; i < noise.Length; i++)
            noise.SetFlat(i, Ops.FromDouble(Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble())));
        var speaker = Voice?.SpeakerEmbedding ?? new Tensor<T>(new[] { _options.SpeakerEmbeddingDim });
        return DiTFlow.Inference(tokens, promptTokens, promptMel, speaker, noise);
    }

    protected override Tensor<T> FlowTrainingLoss(IReadOnlyList<int> speech, Tensor<T> target, Tensor<T> xvector,
        bool training, Random random) => DiTFlow.TrainingLoss(speech, target, xvector, training, random);

    protected override Tensor<T> VocoderForward(Tensor<T> mel, Random random) => CausalHift.Forward(mel, random);
    protected override Tensor<T> VocoderPredictF0(Tensor<T> mel) => CausalHift.PredictF0(mel);
    protected override Tensor<T> VocoderSource(Tensor<T> f0, Random random) => CausalHift.Source(f0, random);
    protected override Tensor<T> VocoderDecode(Tensor<T> mel, Tensor<T> source) => CausalHift.Decode(mel, source);

    public override int[] SpeechTokens(Tensor<T> recording, int sampleRate)
    {
        ThrowIfDisposed();
        var tokenizer = _speechTokenizerOnnx ?? throw new InvalidOperationException(
            "CosyVoice3 prompt tokenization needs speech_tokenizer_v3.onnx. Call LoadPretrainedWeights or set SpeechTokenizerPath.");
        var audio = ResampleAudio(recording, sampleRate, 16000);
        if (audio.Length > _options.MaxReferenceSeconds * 16000)
            throw new ArgumentException($"The reference recording may be at most {_options.MaxReferenceSeconds} seconds.", nameof(recording));
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(audio[i]);
        var logMel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(samples, 128);
        int bands = logMel.GetLength(0), frames = logMel.GetLength(1);
        var features = new Tensor<T>(new[] { 1, bands, frames });
        for (int m = 0; m < bands; m++)
            for (int f = 0; f < frames; f++) features[0, m, f] = Ops.FromDouble(logMel[m, f]);
        var length = new Tensor<T>(new[] { 1 });
        length[0] = Ops.FromDouble(frames);
        var inputs = tokenizer.Metadata.Inputs.Select(x => x.Name).ToArray();
        if (inputs.Length != 2)
            throw new InvalidDataException("speech_tokenizer_v3.onnx must expose feature and feature-length inputs.");
        using var _ = new NoGradScope<T>();
        var output = tokenizer.Run(new Dictionary<string, Tensor<T>> { [inputs[0]] = features, [inputs[1]] = length }).Values.First();
        return IdsOf(output);
    }

    private void OpenSpeechTokenizer(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("The CosyVoice3 S3 tokenizer is missing.", path);
        _speechTokenizerOnnx?.Dispose();
        _speechTokenizerOnnx = new OnnxModel<T>(path, _options.OnnxOptions);
        _options.SpeechTokenizerPath = path;
    }

    private protected override void LoadFlowWeights(TorchTensorSource source) => DiTFlow.LoadTorchWeights(source);
    private protected override void LoadVocoderWeights(TorchTensorSource source) => CausalHift.LoadTorchWeights(source);

    private void OpenSpeakerEncoder(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("The original CosyVoice3 CAM++ encoder is missing.", path);
        _speakerEncoderOnnx?.Dispose();
        _speakerEncoderOnnx = new OnnxModel<T>(path, _options.OnnxOptions);
        _options.SpeakerEncoderOnnxPath = path;
    }

    protected override Tensor<T> ForwardSpeakerEncoder(Tensor<T> features)
    {
        if (_speakerEncoderOnnx is null) return base.ForwardSpeakerEncoder(features);
        var inputs = _speakerEncoderOnnx.Metadata.Inputs;
        if (inputs.Count != 1) throw new InvalidDataException("campplus.onnx must expose one Kaldi-feature input.");
        return _speakerEncoderOnnx.Run(new Dictionary<string, Tensor<T>> { [inputs[0].Name] = features }).Values.First();
    }

    protected override void LoadSpeakerEncoderCheckpoint(string modelDirectory)
    {
        string converted = Path.Combine(modelDirectory, "campplus.safetensors");
        string original = Path.Combine(modelDirectory, "campplus.onnx");
        if (File.Exists(converted))
        {
            _speakerEncoderOnnx?.Dispose();
            _speakerEncoderOnnx = null;
            _options.SpeakerEncoderOnnxPath = null;
            base.LoadSpeakerEncoderCheckpoint(modelDirectory);
        }
        else OpenSpeakerEncoder(File.Exists(original) ? original : _options.SpeakerEncoderOnnxPath ?? original);
    }

    protected override void LoadSpeechTokenizerWeights(string modelDirectory, string speechTokenizerFile)
    {
        string path = !string.IsNullOrWhiteSpace(_options.SpeechTokenizerPath) ? _options.SpeechTokenizerPath! :
            File.Exists(Path.Combine(modelDirectory, speechTokenizerFile)) ? Path.Combine(modelDirectory, speechTokenizerFile) :
            Path.Combine(modelDirectory, "speech_tokenizer_v3.batch.onnx");
        OpenSpeechTokenizer(path);
    }

    /// <summary>Loads original <c>llm/flow/hift.pt</c> and frozen ONNX encoders, or converted safetensors components.</summary>
    public void LoadPretrainedWeights(string modelDirectory) => LoadReleasedWeights(modelDirectory, "speech_tokenizer_v3.onnx");

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = IsNative ? "CosyVoice3-Native" : "CosyVoice3-ONNX",
            Description = "Fun-CosyVoice 3: Qwen2, S3-v3, causal DiT flow and CausalHiFT",
            FeatureCount = _options.LLMDim,
            Complexity = _options.NumLLMLayers + _options.DiTDepth,
        };
        metadata.AdditionalInfo["Architecture"] = "Fun-CosyVoice3-0.5B-2512";
        metadata.AdditionalInfo["Mode"] = IsNative ? "Native" : "ONNX";
        metadata.AdditionalInfo["SpeechVocabulary"] = _options.CodebookSize;
        metadata.AdditionalInfo["SpeechStopIds"] = _options.ReservedSpeechTokens;
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate;
        return metadata;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) _speechTokenizerOnnx?.Dispose();
        if (disposing) _speakerEncoderOnnx?.Dispose();
        _speechTokenizerOnnx = null;
        _speakerEncoderOnnx = null;
        base.Dispose(disposing);
    }
}
