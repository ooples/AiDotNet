using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>
/// YourTTS: zero-shot multi-speaker text-to-speech and voice conversion — VITS conditioned on speaker embeddings from a
/// pretrained speaker encoder, with language embeddings for multilingual training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone"
/// (Casanova et al., ICML 2022) and its implementation in Coqui TTS (<c>TTS/tts/models/vits.py</c>, the
/// <c>recipes/vctk/yourtts</c> recipe) for what the paper leaves unstated.</para>
/// <para>
/// The network is VITS's (<see cref="VITS{T}"/>) with YourTTS's changes (§2): the speaker is a d-vector from the H/ASP
/// speaker encoder (§3.1) — of the given reference recording, or of the utterance itself in training — and conditions the
/// posterior encoder and the flow's coupling layers through their WaveNets' global conditioning, the stochastic duration
/// predictor and the decoder through projections added to their inputs; with several languages, a 4-dimensional language
/// embedding is concatenated to every character embedding and also conditions the duration predictor. The speaker
/// encoder is pretrained and frozen (load it with <see cref="LoadSpeakerEncoder"/>). With
/// <see cref="YourTTSOptions.UseSpeakerConsistencyLoss"/> the generator also minimizes
/// <c>−α · cos(φ(g), φ(h))</c> between the speaker embeddings of the real and generated decoder windows (Eq. 1; with
/// gradient through φ into the generated audio, as the erratum and Coqui ≥ 0.12 have it).
/// </para>
/// <para><b>For Beginners:</b> YourTTS can speak in the voice of someone it has never heard before: give it a few
/// seconds of their speech and it extracts a "voice fingerprint" that steers every part of the synthesis.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone",
    "https://arxiv.org/abs/2112.02418",
    Year = 2022,
    Authors = "Casanova et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, Epsilon = 1e-9, WeightDecay = 0.01,
                DecayRate = 0.999875, ReferenceBatchSize = 64,
                Source = "Casanova et al. 2022, Sec. 3.3: AdamW with betas 0.8 and 0.99, weight decay 0.01 and an initial learning "
                        + "rate of 2e-4 decaying exponentially by a gamma of 0.999875, batch size 64.")]
public partial class YourTTS<T> : StochasticDurationVitsBase<T>
{
    private HaspSpeakerEncoder<T>? _speakerEncoder;

    /// <summary>Creates a YourTTS model that runs an exported ONNX graph.</summary>
    public YourTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, YourTTSOptions? options = null)
        : base(architecture, modelPath, options ?? new YourTTSOptions())
    {
    }

    /// <summary>Creates a trainable YourTTS model.</summary>
    public YourTTS(NeuralNetworkArchitecture<T> architecture, YourTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new YourTTSOptions(), optimizer)
    {
    }

    private YourTTSOptions PaperOptions => (YourTTSOptions)VitsOptions;

    /// <inheritdoc />
    protected override int ExternalSpeakerChannels => PaperOptions.SpeakerEncoderDim;

    /// <inheritdoc />
    /// <remarks>Coqui's <c>VitsConfig.grad_clip</c> (1000 for each optimizer).</remarks>
    protected override double GradientClipNorm => 1000.0;

    private HaspSpeakerEncoder<T> SpeakerEncoder
    {
        get
        {
            if (_speakerEncoder is null)
            {
                _speakerEncoder = new HaspSpeakerEncoder<T>(Engine, PaperOptions.SpeakerEncoderDim, PaperOptions.SpeakerEncoderFilters);
                // Frozen and pretrained: persisted with the model, never selected for training.
                ComponentLayers.AddRange(_speakerEncoder.Layers);
            }
            return _speakerEncoder;
        }
    }

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        base.InitializeLayers();
        if (HasPaperLayers) _ = SpeakerEncoder;
    }

    /// <summary>
    /// Loads the pretrained H/ASP speaker encoder from a Coqui <c>ResNetSpeakerEncoder</c> state dictionary (the
    /// <c>model</c> entry of <c>model_se.pth.tar</c>), keyed by its parameter names (e.g. <c>layer1.0.conv1.weight</c>).
    /// </summary>
    public void LoadSpeakerEncoder(IReadOnlyDictionary<string, Tensor<T>> state) => SpeakerEncoder.LoadState(state);

    /// <summary>The d-vector <c>[512]</c> of a recording <c>[samples]</c> at the model's sample rate.</summary>
    public Tensor<T> ComputeSpeakerEmbedding(Tensor<T> recording)
    {
        using var _ = new NoGradScope<T>();
        return SpeakerEncoder.DVector(ToEncoderRate(recording));
    }

    /// <inheritdoc />
    protected override Tensor<T> ExternalSpeaker(Tensor<T> recording)
        => Engine.Reshape(ComputeSpeakerEmbedding(recording), new[] { 1, PaperOptions.SpeakerEncoderDim, 1 });

    private Tensor<T> ToEncoderRate(Tensor<T> audio)
        => HaspSpeakerEncoder<T>.Resample(Engine, Engine.Reshape(audio, new[] { audio.Length }), PaperOptions.SampleRate, HaspSpeakerEncoder<T>.SampleRate);

    /// <inheritdoc />
    /// <remarks>The speaker consistency loss (Eq. 1) when enabled: <c>−α · cos(φ(g), φ(h))</c> for the real and generated
    /// decoder windows, both resampled to 16 kHz and embedded with L2 normalization (Coqui <c>VitsGeneratorLoss</c>).</remarks>
    protected override Tensor<T>? ExtraGeneratorLoss(VitsUtterance data, Tensor<T> realSegment, Tensor<T> generatedSegment)
    {
        var o = PaperOptions;
        if (!o.UseSpeakerConsistencyLoss) return null;
        Tensor<T> real;
        using (new NoGradScope<T>())
        {
            var r = SpeakerEncoder.Embed(ToEncoderRate(realSegment));
            real = new Tensor<T>(r._shape, r.ToVector());
        }
        var generated = SpeakerEncoder.Embed(ToEncoderRate(generatedSegment));
        // Both embeddings have unit norm, so the cosine similarity is their dot product.
        var cosine = Engine.ReduceSum(Engine.TensorMultiply(real, generated), new[] { 0 }, keepDims: false);
        return Engine.TensorMultiplyScalar(cosine, NumOps.FromDouble(-o.SpeakerConsistencyLossWeight));
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var o = PaperOptions;
        var m = new ModelMetadata<T>
        {
            Name = IsOnnxMode ? "YourTTS-ONNX" : "YourTTS-Native",
            Description = "YourTTS: zero-shot multi-speaker TTS and voice conversion (Casanova et al., 2022)",
            FeatureCount = o.HiddenDim,
            Complexity = o.NumEncoderLayers + o.NumFlowSteps,
        };
        m.AdditionalInfo["Architecture"] = "YourTTS";
        m.AdditionalInfo["SampleRate"] = o.SampleRate.ToString();
        return m;
    }
}
