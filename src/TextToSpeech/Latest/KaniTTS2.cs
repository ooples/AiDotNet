using AiDotNet.Attributes;
using AiDotNet.Audio.Foundations;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// KaniTTS 2 (nineninesix.ai, 2026): Kani's LFM2 language model writing NVIDIA NanoCodec codes, with frame-level rotary
/// positions, a learnable rotary scale per attention layer and a speaker embedding from a recording (WavLM-Large speaker
/// embeddings) inserted after the first token, so it speaks in a reference speaker's voice.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>KaniTTS 2 has no paper; its release is the reference: kani_tts/model.py's <c>KaniTTS2ForCausalLM</c> (positions,
/// learnable rotary scale, speaker insertion, generation), kani_tts/core.py (the prompt with a language tag, sampling, the
/// speech span), kani_tts/speaker_embedder.py, and kani-tts-2-pretrain (the data collator and trainer).</para>
/// <para><b>For Beginners:</b> Give KaniTTS 2 a few seconds of someone speaking and it reads text aloud in that voice;
/// optionally a language or accent tag picks how it pronounces the text.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new KaniTTS2&lt;float&gt;(architecture, new KaniTTS2Options());
/// model.LoadReferenceWeights("kani-tts-2-en");                                 // model.safetensors, tokenizer.json, config.json
/// model.LoadCodecWeights("nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo");
/// model.LoadSpeakerEmbedderWeights("speaker-emb-tbr");                         // nineninesix/speaker-emb-tbr
/// model.SpeakerEmbedding = model.EmbedSpeaker(referenceRecording);              // 22.05 kHz
/// model.LanguageTag = "en_us";
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Kani TTS 2", "https://github.com/nineninesix-ai/kani-tts-2", Year = 2026, Authors = "nineninesix.ai")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 9e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    MaxGradientNorm = 3, WarmupSteps = 2000, GradientAccumulationSteps = 6, ReferenceBatchSize = 6,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "No paper. nineninesix-ai/kani-tts-2-pretrain configs/training_config.yaml: adamw_torch_fused at 9e-4 (weight "
             + "decay the Trainer's default 0), cosine schedule after 2000 warm-up steps, max_grad_norm 3, batch 6 with 6-step "
             + "accumulation; utils/trainer.py's LearnableRoPETrainer puts the rotary scales (alpha_weight) in their own group at "
             + "alpha_lr_ratio 1.5 times the rate without weight decay; the loss on every token but the speaker slot.")]
public partial class KaniTTS2<T> : KaniTTSBase<T>
{
    private readonly List<LayerBase<T>> _embedderLayers = new();
    private KaniSpeakerEmbedder<T>? _embedder;

    /// <summary>Creates an ONNX-backed KaniTTS 2 for inference.</summary>
    public KaniTTS2(NeuralNetworkArchitecture<T> architecture, string modelPath, KaniTTS2Options? options = null)
        : base(architecture, modelPath, options ?? new KaniTTS2Options())
    {
    }

    /// <summary>Creates a native, trainable KaniTTS 2.</summary>
    public KaniTTS2(NeuralNetworkArchitecture<T> architecture, KaniTTS2Options? options = null)
        : base(architecture, options ?? new KaniTTS2Options())
    {
    }

    private KaniTTS2Options Options2 => (KaniTTS2Options)KaniOptions;

    /// <summary>The language or accent tag synthesis puts before the text ("en_us: text"); null for none. Unset, a
    /// voice's <see cref="TtsVoice{T}.LanguageId"/> picks one of <see cref="KaniTTS2Options.LanguageTags"/>.</summary>
    public string? LanguageTag { get; set; }

    /// <summary>The speaker embedding synthesis speaks with (<see cref="KaniTTS2Options.SpeakerEmbeddingDim"/> values, such
    /// as <see cref="EmbedSpeaker"/> returns); null to use the voice's reference recording, or no speaker.</summary>
    public double[]? SpeakerEmbedding { get; set; }

    /// <summary>Speaker embeddings by training sample <see cref="TtsTrainingSample{T}.SpeakerId"/> — the release trains
    /// each sample on its speaker's averaged embedding (<see cref="AverageSpeakerEmbedding"/>). A sample whose speaker has
    /// none uses its <see cref="TtsTrainingSample{T}.SpeakerReference"/> recording, or no speaker.</summary>
    public IDictionary<int, double[]> TrainingSpeakerEmbeddings { get; } = new Dictionary<int, double[]>();

    internal KaniSpeakerEmbedder<T> Embedder => _embedder ?? throw NotNative();

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        base.InitializeLayers();
        if (!HasPaperLayers) return;
        var o = Options2;
        _embedder = new KaniSpeakerEmbedder<T>(Engine, _embedderLayers, new KaniSpeakerEmbedderConfiguration
        {
            WavLm = new WavLmConfiguration
            {
                ConvDims = (int[])o.EmbedderConvDims.Clone(), HiddenSize = o.EmbedderWidth, Layers = o.EmbedderLayers, Heads = o.EmbedderHeads,
                IntermediateSize = o.EmbedderIntermediate, PositionalConvKernel = o.EmbedderPositionalKernel,
                PositionalConvGroups = o.EmbedderPositionalGroups, LayerNormFeatureEncoder = true, StableLayerNorm = true,
            },
            IntermediateSize = o.EmbedderHeadWidth, EmbeddingSize = o.SpeakerEmbeddingDim, MaxSeconds = o.MaxReferenceSeconds,
        });
        ComponentLayers.AddRange(_embedderLayers);
    }

    /// <summary>
    /// The speaker embedding of a recording at the model's sample rate (kani_tts's <c>SpeakerEmbedder.embed_audio</c>:
    /// resampled to 16 kHz by torchaudio's resampler, at most 30 seconds, through the WavLM-Large embedder).
    /// </summary>
    public double[] EmbedSpeaker(Tensor<T> recording)
    {
        if (!HasPaperLayers) throw NotNative();
        Validation.Guard.NotNull(recording);
        using var _ = new NoGradScope<T>();
        var audio = Engine.Reshape(recording, new[] { recording.Length });
        var audio16k = KaniOptions.SampleRate == 16000 ? audio
            : new TorchaudioResample<T>(Engine, KaniOptions.SampleRate, 16000).Forward(audio);
        return Embedder.Embed(audio16k).ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
    }

    /// <summary>The release's per-speaker embedding (kanitts-2-dataset-pipeline's regrouping): the plain mean of the
    /// speaker's recordings' embeddings.</summary>
    public double[] AverageSpeakerEmbedding(IEnumerable<Tensor<T>> recordings)
    {
        var embeddings = recordings.Select(EmbedSpeaker).ToList();
        if (embeddings.Count == 0) throw new ArgumentException("There are no recordings.", nameof(recordings));
        return Enumerable.Range(0, embeddings[0].Length).Select(i => embeddings.Average(e => e[i])).ToArray();
    }

    private Tensor<T> AsSpeaker(double[] embedding)
    {
        if (embedding.Length != KaniOptions.SpeakerEmbeddingDim)
            throw new ArgumentException($"A speaker embedding has {KaniOptions.SpeakerEmbeddingDim} values.", nameof(embedding));
        return new Tensor<T>(new[] { 1, embedding.Length }, new Vector<T>(embedding.Select(v => NumOps.FromDouble(v)).ToArray()));
    }

    private string? Tag(int? languageId)
    {
        var tags = Options2.LanguageTags;
        if (languageId is null || tags.Length == 0) return LanguageTag;
        if (languageId < 0 || languageId >= tags.Length)
            throw new ArgumentOutOfRangeException(nameof(languageId), $"Language {languageId} is not one of the checkpoint's {tags.Length} tags.");
        return tags[languageId.Value];
    }

    /// <inheritdoc />
    /// <remarks>KaniTTS 2 prefixes a language tag (<see cref="LanguageTag"/>, or the voice's language).</remarks>
    protected override string? SynthesisPrefix() => LanguageTag ?? Tag(Voice?.LanguageId);

    /// <inheritdoc />
    /// <remarks>The sample's language tag (lowercased by the pipeline).</remarks>
    protected override string? TrainingPrefix(TtsTrainingSample<T> sample) => Tag(sample.LanguageId);

    /// <inheritdoc />
    protected override Tensor<T>? SynthesisSpeakerEmbedding()
    {
        if (SpeakerEmbedding is { } embedding) return AsSpeaker(embedding);
        return Voice?.ReferenceAudio is { } reference ? AsSpeaker(EmbedSpeaker(reference)) : null;
    }

    /// <inheritdoc />
    protected override Tensor<T>? TrainingSpeakerEmbedding(TtsTrainingSample<T> sample)
    {
        if (sample.SpeakerId is int id && TrainingSpeakerEmbeddings.TryGetValue(id, out var embedding)) return AsSpeaker(embedding);
        return sample.SpeakerReference is { } reference ? AsSpeaker(EmbedSpeaker(reference)) : null;
    }

    /// <inheritdoc />
    /// <remarks>KaniTTS 2's config gives its language tags and its position and rotary settings, which must match these
    /// options.</remarks>
    protected override void ReadCheckpointConfig(JObject config)
    {
        var o = Options2;
        if (config["language_settings"]?["language_tags_list"] is JArray tags) o.LanguageTags = tags.Select(t => (string)t!).ToArray();
        if (config["audio_step"] is JValue step) o.AudioStep = (double)step;
        if (config["use_learnable_rope"] is JValue rope && (bool)rope != o.LearnableRope)
            throw new InvalidDataException($"The checkpoint's use_learnable_rope is {(bool)rope}; these options build {o.LearnableRope}.");
        if (config["speaker_emb_dim"] is JValue dim && (int)dim != o.SpeakerEmbeddingDim)
            throw new InvalidDataException($"The checkpoint's speaker_emb_dim is {(int)dim}; these options build {o.SpeakerEmbeddingDim}.");
    }

    /// <summary>Loads the speaker embedder: nineninesix/speaker-emb-tbr's <c>model.safetensors</c> (a file, or the folder
    /// holding it).</summary>
    public void LoadSpeakerEmbedderWeights(string path)
    {
        if (!HasPaperLayers) throw NotNative();
        string file = Directory.Exists(path) ? Path.Combine(path, "model.safetensors") : path;
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
            weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        Embedder.LoadTorchWeights(name => weights.TryGetValue(name, out var v) ? v : null);
    }
}
