using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

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
/// <para>Voices (<see cref="CosyVoiceBase{T}.CreateVoice(Tensor{T}, int, string?)"/>): zero-shot, from a recording and its
/// transcript (the LM continues the prompt's text and tokens); cross-lingual, from a recording alone (only the flow reads
/// the prompt); or a speaker embedding alone (<c>sft</c>). Without a voice the x-vector is zero.</para>
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
[PaperOptimizer(OptimizerKind.Adam, Component = "vocoder", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice examples/libritts/cosyvoice/conf/cosyvoice.yaml "
        + "train_conf_gan: Adam at a constant 2e-4 for HiFT, gradient clipping at 5, no accumulation.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "discriminator", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice cosyvoice.yaml train_conf_gan optim_d: Adam at a "
        + "constant 2e-4 for the discriminators.")]
public partial class CosyVoice<T> : CosyVoiceBase<T>
{
    private readonly CosyVoiceOptions _options;
    private readonly TiktokenEncoding _text;
    private CosyVoiceLm<T>? _lm;
    private CosyVoiceFlowConfiguration? _flowConfiguration;

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
        : base(architecture, (options ?? new CosyVoiceOptions()).SamplingSeed, modelPath, (options ?? new CosyVoiceOptions()).OnnxOptions)
    {
        _options = options ?? new CosyVoiceOptions();
        Options = _options;
        _options.ModelPath = modelPath;
        _text = WhisperTiktoken.Multilingual(_options.TextLanguages);
        // The text vocabulary is the tokenizer's, as text_token_size is.
        _options.VocabSize = _text.VocabularySize;
        Initialize();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">An optimizer for every stage instead of the recipe's per-stage Adam.</param>
    public CosyVoice(
        NeuralNetworkArchitecture<T> architecture,
        CosyVoiceOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, (options ?? new CosyVoiceOptions()).SamplingSeed, optimizer)
    {
        _options = options ?? new CosyVoiceOptions();
        Options = _options;
        _text = WhisperTiktoken.Multilingual(_options.TextLanguages);
        _options.VocabSize = _text.VocabularySize;
        Initialize();
    }

    /// <inheritdoc />
    protected override CosyVoiceFamilyOptions Settings => _options;

    /// <inheritdoc />
    protected override int TextVocabularySize => _text.VocabularySize;

    /// <summary>The text token ids of <paramref name="text"/>: Whisper's multilingual tiktoken encoding with special
    /// tokens written in the text allowed (<c>encode(text, allowed_special='all')</c>).</summary>
    public override int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return _text.Encode(text);
    }

    /// <inheritdoc />
    public override string Detokenize(IEnumerable<int> ids) => _text.Decode(ids);

    /// <inheritdoc />
    /// <remarks><c>speech_tokenizer_v1</c>: S3 at 50 Hz.</remarks>
    private protected override S3TokenizerConfiguration SpeechTokenizerConfiguration => new(
        State: _options.SpeechTokenizerState, Heads: _options.SpeechTokenizerHeads, Layers: _options.SpeechTokenizerLayers,
        CodebookSize: _options.CodebookSize);

    /// <inheritdoc />
    /// <remarks><c>MaskedDiffWithXvec</c>: a Conformer encoder, <c>InterpolateRegulator</c>'s four stages and
    /// <c>ConditionalCFM</c>'s U-Net.</remarks>
    private protected override CosyVoiceFlowConfiguration FlowConfiguration => _flowConfiguration ??= new CosyVoiceFlowConfiguration(
        VocabSize: _options.CodebookSize, InputSize: _options.FlowDim, OutputSize: _options.MelChannels,
        SpeakerDim: _options.SpeakerEmbeddingDim, InputFrameRate: _options.CodecFrameRate, SampleRate: _options.SampleRate,
        HopSize: _options.HopSize,
        Encoder: new WenetEncoderConfiguration(_options.FlowDim, _options.FlowDim, _options.FlowHeads, _options.FlowFeedForwardDim,
            _options.FlowEncoderLayers, WenetInputLayer.Linear, Swish: true, Conformer: true, Dropout: _options.DropoutRate,
            PositionalDropout: _options.DropoutRate, AttentionDropout: _options.FlowAttentionDropoutRate),
        RegulatorStages: 4, DecoderChannels: (int[])_options.DecoderChannels.Clone(), DecoderHeads: _options.DecoderHeads,
        DecoderHeadDim: _options.DecoderHeadDim, DecoderBlocksPerLevel: _options.DecoderBlocks, DecoderMidBlocks: _options.DecoderMidBlocks,
        SigmaMin: 1e-6, TrainingCfgRate: _options.TrainingCfgRate, InferenceCfgRate: _options.InferenceCfgRate,
        Timesteps: _options.FlowTimesteps, Loss: _options.FlowLoss);

    private CosyVoiceLm<T> Lm => _lm ?? throw new InvalidOperationException("CosyVoice's language model exists in native mode only.");

    // The language model, for tests.
    internal CosyVoiceLm<T>? LanguageModel => _lm;

    /// <inheritdoc />
    /// <remarks><c>TransformerLM</c>: a Conformer text encoder (linear input) and a Transformer (linear_legacy input), both
    /// causal (static_chunk_size 1) with ESPnet relative positions; no attention dropout.</remarks>
    protected override void BuildLanguageModel(List<LayerBase<T>> layers, Random initialization)
    {
        var o = _options;
        _lm = new CosyVoiceLm<T>(Engine, layers, new CosyVoiceLmConfiguration(
            TextVocabulary: _text.VocabularySize, SpeechVocabulary: o.CodebookSize, TextEncoderInputSize: o.TextEncoderDim,
            LlmSize: o.LLMDim, SpeakerDim: o.SpeakerEmbeddingDim,
            TextEncoder: new WenetEncoderConfiguration(o.TextEncoderDim, o.LLMDim, o.NumHeads, o.LmFeedForwardDim, o.NumEncoderLayers,
                WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: 1, Dropout: o.DropoutRate,
                PositionalDropout: o.DropoutRate),
            Llm: new WenetEncoderConfiguration(o.LLMDim, o.LLMDim, o.NumHeads, o.LmFeedForwardDim, o.NumLLMLayers,
                WenetInputLayer.LinearLegacy, Swish: false, Conformer: false, StaticChunkSize: 1, Dropout: o.DropoutRate,
                PositionalDropout: o.DropoutRate),
            TopP: o.TopP, TopK: o.TopK, RepetitionWindow: o.RepetitionWindow, RepetitionRatio: o.RepetitionRatio), initialization);
    }

    /// <inheritdoc />
    protected override int[] GenerateWithLanguageModel(IReadOnlyList<int> text, IReadOnlyList<int> promptText,
        IReadOnlyList<int> promptSpeech, Tensor<T>? xvector, Random random) =>
        Lm.Generate(text, promptText, promptSpeech, xvector, random).ToArray();

    /// <inheritdoc />
    /// <remarks>The mean cross-entropy of the speech tokens and the end (<c>TransformerLM.forward</c>).</remarks>
    protected override Tensor<T> LanguageModelLoss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> xvector,
        bool training, Random random) => Lm.Loss(text, speech, xvector, training, random);

    /// <inheritdoc />
    protected override void LoadLanguageModel(Func<string, int[], double[]> read) => Lm.LoadTorchWeights(read, "");

    /// <summary>
    /// Loads the released networks from a CosyVoice-300M directory, each a state dict converted to safetensors:
    /// <c>llm.safetensors</c>, <c>flow.safetensors</c> and <c>hift.safetensors</c> (from <c>llm.pt</c>, <c>flow.pt</c>,
    /// <c>hift.pt</c>), <c>speech_tokenizer_v1.safetensors</c> (xingchensong/S3Tokenizer's PyTorch form of the ONNX
    /// tokenizer) and <c>campplus.safetensors</c> (3D-Speaker's CAM++).
    /// </summary>
    public void LoadPretrainedWeights(string modelDirectory) => LoadReleasedWeights(modelDirectory, "speech_tokenizer_v1.safetensors");

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = IsNative ? "CosyVoice-Native" : "CosyVoice-ONNX",
            Description =
                "CosyVoice: Multilingual TTS with Supervised Semantic Tokens (Du et al., 2024)",
            FeatureCount = _options.LLMDim,
            Complexity = _options.NumLLMLayers,
        };
        m.AdditionalInfo["Architecture"] = "CosyVoice";
        m.AdditionalInfo["Mode"] = IsNative ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }
}
