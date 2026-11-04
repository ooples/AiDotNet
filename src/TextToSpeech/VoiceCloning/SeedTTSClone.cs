using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Voice-cloning facade over Seed-TTS's audio/text in-context continuation pipeline.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Seed-TTS voice cloning is not a second neural architecture. This class deliberately inherits the same speech
/// tokenizer, causal token LM, token diffusion model, timbre conditioner, and acoustic vocoder. It only adapts the
/// general Seed-TTS prompt API to <see cref="IVoiceCloner{T}"/>.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Seed-TTS: A Family of High-Quality Versatile Speech Generation Models",
    "https://arxiv.org/abs/2406.02430",
    Year = 2024,
    Authors = "Anastassiou et al."
)]
public partial class SeedTTSClone<T> : SeedTTS<T>, IVoiceCloner<T>
{
    private readonly SeedTTSCloneOptions _cloneOptions;

    /// <summary>Creates the facade in ONNX mode.</summary>
    public SeedTTSClone(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SeedTTSCloneOptions? options = null)
        : base(architecture, modelPath, options ?? new SeedTTSCloneOptions())
    {
        _cloneOptions = options ?? new SeedTTSCloneOptions();
    }

    /// <summary>Creates the facade over the native Seed-TTS pipeline.</summary>
    public SeedTTSClone(
        NeuralNetworkArchitecture<T> architecture,
        SeedTTSCloneOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new SeedTTSCloneOptions(), optimizer)
    {
        _cloneOptions = options ?? new SeedTTSCloneOptions();
    }

    /// <inheritdoc />
    public double MinReferenceDuration => _cloneOptions.MinReferenceDurationSec;

    /// <inheritdoc />
    public int SpeakerEmbeddingDim => _cloneOptions.SpeakerEmbeddingDim;

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
        => SynthesizeWithVoice(text, _cloneOptions.ReferenceText, referenceAudio);

    /// <summary>Runs the report's full text-and-audio in-context continuation path.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, string referenceText, Tensor<T> referenceAudio)
    {
        ValidateReferenceAudio(referenceAudio);
        return SynthesizeWithPrompt(text, referenceText, referenceAudio);
    }

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        ValidateReferenceAudio(referenceAudio);
        return ExtractTimbreEmbedding(referenceAudio);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.Name = metadata.Name.Replace("Seed-TTS", "Seed-TTS-Clone");
        metadata.Description = "Seed-TTS voice-cloning facade over the native in-context continuation architecture";
        metadata.AdditionalInfo["Architecture"] = "Seed-TTS";
        metadata.AdditionalInfo["TaskFacade"] = "VoiceCloning";
        metadata.AdditionalInfo["ReferenceDurationRange"] = "3-20 seconds in the paper evaluation";
        return metadata;
    }

    private void ValidateReferenceAudio(Tensor<T> referenceAudio)
    {
        if (referenceAudio is null) throw new ArgumentNullException(nameof(referenceAudio));
        int minimumSamples = checked((int)Math.Ceiling(MinReferenceDuration * _cloneOptions.SampleRate));
        if (referenceAudio.Length < minimumSamples)
            throw new ArgumentException(
                $"Seed-TTS evaluation uses at least {MinReferenceDuration:0.###} seconds of reference audio "
                + $"({minimumSamples} samples at {_cloneOptions.SampleRate} Hz).",
                nameof(referenceAudio));
    }
}
