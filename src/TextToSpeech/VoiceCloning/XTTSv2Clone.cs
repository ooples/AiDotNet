using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Voice-cloning facade over the canonical XTTS-v2 zero-shot architecture.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// XTTS-v2 voice cloning is not a second model. This facade inherits the exact dVAE tokenizer, conditioning
/// encoder/Perceiver, GPT-2, H/ASP speaker encoder and speaker-conditioned HiFi-GAN from <see cref="XTTSv2{T}"/>
/// and adapts its reference-audio overload to <see cref="IVoiceCloner{T}"/>.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model",
    "https://arxiv.org/abs/2406.04904",
    Year = 2024,
    Authors = "Casanova et al.")]
public partial class XTTSv2Clone<T> : XTTSv2<T>, IVoiceCloner<T>
{
    private readonly XTTSv2CloneOptions _cloneOptions;

    public XTTSv2Clone(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        XTTSv2CloneOptions? options = null)
        : base(architecture, modelPath, options ?? new XTTSv2CloneOptions())
    {
        _cloneOptions = options ?? new XTTSv2CloneOptions();
    }

    public XTTSv2Clone(
        NeuralNetworkArchitecture<T> architecture,
        XTTSv2CloneOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new XTTSv2CloneOptions(), optimizer)
    {
        _cloneOptions = options ?? new XTTSv2CloneOptions();
    }

    public double MinReferenceDuration => _cloneOptions.MinReferenceDurationSec;
    public int SpeakerEmbeddingDim => _cloneOptions.SpeakerEmbeddingDim;

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        ValidateEnrollment(referenceAudio);
        return Synthesize(text, _cloneOptions.Language, referenceAudio);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.Name = metadata.Name.Replace("XTTS-v2", "XTTS-v2 Clone");
        metadata.Description = "Voice-cloning facade over the canonical XTTS-v2 zero-shot graph";
        metadata.AdditionalInfo["TaskFacade"] = "VoiceCloning";
        metadata.AdditionalInfo["MinimumReferenceSeconds"] = MinReferenceDuration;
        return metadata;
    }

    private void ValidateEnrollment(Tensor<T> referenceAudio)
    {
        Validation.Guard.NotNull(referenceAudio);
        int minimum = checked((int)Math.Ceiling(MinReferenceDuration * _cloneOptions.SampleRate));
        if (referenceAudio.Length < minimum)
            throw new ArgumentException(
                $"XTTS-v2 enrollment requires at least {MinReferenceDuration:0.###} seconds of reference audio "
                + $"({minimum} samples at {_cloneOptions.SampleRate} Hz).", nameof(referenceAudio));
    }
}
