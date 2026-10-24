using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>CosyVoice 2: streaming TTS with a pretrained LLM, finite-scalar-quantized speech tokens and chunk-aware
/// causal flow matching.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models" (Du et al., 2024)</item>
/// <item>Code: FunAudioLLM/CosyVoice (Apache-2.0), CosyVoice2-0.5B's <c>cosyvoice2.yaml</c>.</item></list></para>
/// <para>Text is tokenized by Qwen2.5's byte-level BPE with CosyVoice 2's added tokens (<see cref="QwenTokenizer.CosyVoice2"/>).
/// The language model, Qwen2.5-0.5B (<c>Qwen2LM</c>), reads <c>[sos, text, task_id]</c> through its own token table and
/// generates the 25 Hz S3Tokenizer v2's FSQ tokens by repetition-aware sampling; it reads no x-vector. The causal flow
/// (<c>CausalMaskedDiffWithXvec</c>) encodes the tokens with an upsampling Conformer giving two mel frames per token and
/// samples an 80-band mel with a causal U-Net estimator, conditioned on the x-vector and a prompt's mel; in streaming
/// both attend within chunks. HiFT turns the mel into 24 kHz audio (upsampling 8·5·3 with SineGen2's excitation). The S3
/// tokenizer v2 and CAM++ provide a prompt's tokens and x-vector, frozen as in the release.</para>
/// <para>Voices (<see cref="CosyVoiceBase{T}.CreateVoice(Tensor{T}, int, string?)"/>): zero-shot with a transcript,
/// cross-lingual without, or a speaker embedding alone.</para>
/// <para><b>For Beginners:</b> CosyVoice 2 reads text with a small general-purpose language model that has learned to
/// "speak" in speech tokens, then turns those tokens into sound. It can imitate a voice from a few seconds of
/// recording.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new CosyVoice2&lt;float&gt;(architecture, new CosyVoice2Options());
/// model.LoadPretrainedWeights("pretrained_models/CosyVoice2-0.5B");
/// model.Voice = model.CreateVoice(promptAudio, 16000, "The transcript of the prompt.");
/// var audio = model.Synthesize("Hello from CosyVoice 2.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models",
    "https://arxiv.org/abs/2412.10117",
    Year = 2024,
    Authors = "Du et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Component = "language model", LearningRate = 1e-5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice examples/libritts/cosyvoice2/conf/cosyvoice2.yaml train_conf: "
        + "Adam at a constant 1e-5 (constantlr; the recipe fine-tunes from the released model), gradient clipping at 5 and "
        + "2-step accumulation; run.sh trains llm and flow with it in turn.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "flow", LearningRate = 1e-5, MaxGradientNorm = 5,
    GradientAccumulationSteps = 2, Provenance = RecipeProvenance.ReleasedCode,
    Source = "FunAudioLLM/CosyVoice examples/libritts/cosyvoice2/conf/cosyvoice2.yaml train_conf, the flow's run of the "
        + "same recipe as the language model's.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "vocoder", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice cosyvoice2.yaml train_conf_gan: Adam at a constant 2e-4 "
        + "for HiFT, gradient clipping at 5, no accumulation.")]
[PaperOptimizer(OptimizerKind.Adam, Component = "discriminator", LearningRate = 2e-4, MaxGradientNorm = 5,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "The paper names no optimizer. FunAudioLLM/CosyVoice cosyvoice2.yaml train_conf_gan optim_d: Adam at a "
        + "constant 2e-4 for the discriminators.")]
public partial class CosyVoice2<T> : CosyVoiceBase<T>
{
    private readonly CosyVoice2Options _options;
    private readonly QwenTokenizer _text = QwenTokenizer.CosyVoice2;
    private CosyVoice2Lm<T>? _lm;
    private CosyVoiceFlowConfiguration? _flowConfiguration;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice2{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public CosyVoice2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        CosyVoice2Options? options = null
    )
        : base(architecture, (options ?? new CosyVoice2Options()).SamplingSeed, modelPath, (options ?? new CosyVoice2Options()).OnnxOptions)
    {
        _options = options ?? new CosyVoice2Options();
        _options.ModelPath = modelPath;
        Validate();
        Initialize();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice2{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">An optimizer for every stage instead of the recipe's per-stage Adam.</param>
    public CosyVoice2(
        NeuralNetworkArchitecture<T> architecture,
        CosyVoice2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, (options ?? new CosyVoice2Options()).SamplingSeed, optimizer)
    {
        _options = options ?? new CosyVoice2Options();
        Validate();
        Initialize();
    }

    private void Validate()
    {
        // Qwen's token table is wider than the tokenizer (151,936 rows for 151,682 ids), as the release's is.
        if (_options.VocabSize < _text.VocabularySize)
            throw new ArgumentException($"VocabSize ({_options.VocabSize}) must cover the tokenizer's {_text.VocabularySize} ids.",
                nameof(_options.VocabSize));
        if (_options.TokenMelRatio <= 0 || _options.StaticChunkSize <= 0)
            throw new ArgumentException("TokenMelRatio and StaticChunkSize must be positive.");
    }

    /// <inheritdoc />
    protected override CosyVoiceFamilyOptions Settings => _options;

    /// <inheritdoc />
    /// <remarks>Qwen's token table: every id the tokenizer gives lies below it.</remarks>
    protected override int TextVocabularySize => _options.VocabSize;

    /// <summary>The text token ids of <paramref name="text"/>: Qwen2.5's tokenizer with CosyVoice 2's added tokens
    /// (<c>get_qwen_tokenizer(skip_special_tokens=True)</c>), special tokens written in the text allowed.</summary>
    public override int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return _text.Encode(text);
    }

    /// <inheritdoc />
    public override string Detokenize(IEnumerable<int> ids) => _text.Decode(ids);

    /// <inheritdoc />
    /// <remarks><c>speech_tokenizer_v2</c>: S3 at 25 Hz with FSQ over 6,561 codes.</remarks>
    private protected override S3TokenizerConfiguration SpeechTokenizerConfiguration => S3TokenizerConfiguration.V2Default with
    {
        State = _options.SpeechTokenizerState,
        Heads = _options.SpeechTokenizerHeads,
        Layers = _options.SpeechTokenizerLayers,
        CodebookSize = _options.CodebookSize,
    };

    /// <inheritdoc />
    /// <remarks><c>CausalMaskedDiffWithXvec</c>: an <c>UpsampleConformerEncoder</c> (relative-position attention without
    /// macaron or convolution modules, chunks of <see cref="CosyVoice2Options.StaticChunkSize"/>), no length regulator, and
    /// <c>CausalConditionalCFM</c>'s causal U-Net (chunks of that times <see cref="CosyVoice2Options.TokenMelRatio"/>).</remarks>
    private protected override CosyVoiceFlowConfiguration FlowConfiguration => _flowConfiguration ??= new CosyVoiceFlowConfiguration(
        VocabSize: _options.CodebookSize, InputSize: _options.FlowDim, OutputSize: _options.MelChannels,
        SpeakerDim: _options.SpeakerEmbeddingDim, InputFrameRate: _options.CodecFrameRate, SampleRate: _options.SampleRate,
        HopSize: _options.HopSize,
        Encoder: new WenetEncoderConfiguration(_options.FlowDim, _options.FlowDim, _options.FlowHeads, _options.FlowFeedForwardDim,
            _options.FlowEncoderLayers, WenetInputLayer.Linear, Swish: true, Conformer: true, StaticChunkSize: _options.StaticChunkSize,
            Dropout: _options.DropoutRate, PositionalDropout: _options.DropoutRate, AttentionDropout: _options.FlowAttentionDropoutRate),
        RegulatorStages: 0, DecoderChannels: (int[])_options.DecoderChannels.Clone(), DecoderHeads: _options.DecoderHeads,
        DecoderHeadDim: _options.DecoderHeadDim, DecoderBlocksPerLevel: _options.DecoderBlocks, DecoderMidBlocks: _options.DecoderMidBlocks,
        SigmaMin: 1e-6, TrainingCfgRate: _options.TrainingCfgRate, InferenceCfgRate: _options.InferenceCfgRate,
        Timesteps: _options.FlowTimesteps, Loss: _options.FlowLoss, Causal: true, TokenMelRatio: _options.TokenMelRatio,
        Lookahead: _options.Lookahead);

    private CosyVoice2Lm<T> Lm => _lm ?? throw new InvalidOperationException("CosyVoice 2's language model exists in native mode only.");

    // The language model, for tests.
    internal CosyVoice2Lm<T>? LanguageModel => _lm;

    /// <inheritdoc />
    /// <remarks><c>Qwen2LM</c>: Qwen2 with q/k/v biases and no head of its own, grouped-query attention, RoPE and RMSNorm.</remarks>
    protected override void BuildLanguageModel(List<LayerBase<T>> layers, Random initialization)
    {
        var o = _options;
        _lm = new CosyVoice2Lm<T>(Engine, layers, new CosyVoice2LmConfiguration(new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.LLMDim, Layers = o.NumLLMLayers, Heads = o.NumHeads, KeyValueHeads = o.KeyValueHeads,
            Intermediate = o.LmFeedForwardDim, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, AttentionBias = true,
            IncludeHead = false,
        }, o.CodebookSize, TopP: o.TopP, TopK: o.TopK, RepetitionWindow: o.RepetitionWindow, RepetitionRatio: o.RepetitionRatio),
            initialization);
    }

    /// <inheritdoc />
    /// <remarks><c>Qwen2LM</c> reads no x-vector: the speaker reaches the language model only through a zero-shot
    /// prompt.</remarks>
    protected override int[] GenerateWithLanguageModel(IReadOnlyList<int> text, IReadOnlyList<int> promptText,
        IReadOnlyList<int> promptSpeech, Tensor<T>? xvector, Random random) =>
        Lm.Generate(text, promptText, promptSpeech, random).ToArray();

    /// <inheritdoc />
    /// <remarks>The mean cross-entropy of <c>prepare_lm_input_target</c>'s scored positions, unistream or (half the time,
    /// when the utterance qualifies) bistream.</remarks>
    protected override Tensor<T> LanguageModelLoss(IReadOnlyList<int> text, IReadOnlyList<int> speech, Tensor<T> xvector,
        bool training, Random random) => Lm.Loss(text, speech, random);

    /// <inheritdoc />
    protected override void LoadLanguageModel(Func<string, int[], double[]> read) => Lm.LoadTorchWeights(read, "");

    /// <summary>
    /// Loads the released networks from a CosyVoice2-0.5B directory, each a state dict converted to safetensors:
    /// <c>llm.safetensors</c>, <c>flow.safetensors</c> and <c>hift.safetensors</c> (from <c>llm.pt</c>, <c>flow.pt</c>,
    /// <c>hift.pt</c>), <c>speech_tokenizer_v2.safetensors</c> (xingchensong/S3Tokenizer's PyTorch form of the ONNX
    /// tokenizer) and <c>campplus.safetensors</c> (3D-Speaker's CAM++).
    /// </summary>
    public void LoadPretrainedWeights(string modelDirectory) => LoadReleasedWeights(modelDirectory, "speech_tokenizer_v2.safetensors");

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = IsNative ? "CosyVoice2-Native" : "CosyVoice2-ONNX",
            Description = "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models (Du et al., 2024)",
            FeatureCount = _options.LLMDim,
            Complexity = _options.NumLLMLayers,
        };
        m.AdditionalInfo["Architecture"] = "CosyVoice2";
        m.AdditionalInfo["Mode"] = IsNative ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }
}
