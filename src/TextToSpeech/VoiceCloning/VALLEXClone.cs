using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Voice-cloning facade over VALL-E X's cross-lingual neural-codec language model.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Voice cloning is the central inference task of VALL-E X, not a second network. This class inherits the exact native
/// bilingual phoneme table, language embeddings, AR first-codebook Transformer, prompted NAR remaining-codebook
/// Transformer and 24 kHz EnCodec pipeline from <see cref="VALLEX{T}"/>. It only adapts that prompt API to
/// <see cref="IVoiceCloner{T}"/>. The reference transcript remains mandatory because Eq. 3 conditions the AR model on
/// both the enrolled phonemes and enrolled acoustic codes.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling",
    "https://arxiv.org/abs/2303.03926",
    Year = 2023,
    Authors = "Zhang et al.")]
public partial class VALLEXClone<T> : VALLEX<T>, IVoiceCloner<T>
{
    private readonly VALLEXCloneOptions _cloneOptions;

    /// <summary>Creates the facade in ONNX inference mode.</summary>
    public VALLEXClone(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VALLEXCloneOptions? options = null)
        : base(architecture, modelPath, options ?? new VALLEXCloneOptions())
    {
        _cloneOptions = options ?? new VALLEXCloneOptions();
    }

    /// <summary>Creates the facade over the native VALL-E X paper architecture.</summary>
    public VALLEXClone(
        NeuralNetworkArchitecture<T> architecture,
        VALLEXCloneOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VALLEXCloneOptions(), optimizer)
    {
        _cloneOptions = options ?? new VALLEXCloneOptions();
    }

    /// <inheritdoc />
    public double MinReferenceDuration => _cloneOptions.MinReferenceDurationSec;

    /// <summary>
    /// Width of each frame in the prompt representation. VALL-E X has no fixed speaker vector: its "speaker embedding"
    /// is the variable-length sequence of all EnCodec codebooks returned by <see cref="ExtractSpeakerEmbedding"/>.
    /// </summary>
    public int SpeakerEmbeddingDim => _cloneOptions.NumCodebooks;

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        if (string.IsNullOrWhiteSpace(_cloneOptions.ReferenceText))
            throw new InvalidOperationException(
                "VALL-E X requires the reference transcript. Set VALLEXCloneOptions.ReferenceText or use the overload that accepts it.");
        return SynthesizeWithVoice(text, _cloneOptions.ReferenceText, referenceAudio);
    }

    /// <summary>Runs Eq. 3–4 with a source-language recording/transcript and target-language text.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, string referenceText, Tensor<T> referenceAudio)
    {
        ValidateReference(referenceAudio, referenceText);
        var previousVoice = Voice;
        try
        {
            Voice = CreateVoice(referenceAudio, referenceText);
            return Synthesize(text);
        }
        finally
        {
            Voice = previousVoice;
        }
    }

    /// <summary>
    /// Returns VALL-E X's actual variable-length in-context speaker representation: <c>[frames, codebooks]</c> EnCodec
    /// ids. The paper does not use or define a fixed-dimensional speaker embedding.
    /// </summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        ValidateReferenceAudio(referenceAudio);
        return EncodeToTokens(referenceAudio);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.Name = metadata.Name.Replace("VALL-E X", "VALL-E X Clone");
        metadata.Description = "VALL-E X voice-cloning facade over the bilingual AR/NAR neural-codec architecture";
        metadata.AdditionalInfo["Architecture"] = "VALL-E X";
        metadata.AdditionalInfo["TaskFacade"] = "VoiceCloning";
        metadata.AdditionalInfo["SpeakerRepresentation"] = "variable-length EnCodec prompt (no fixed speaker vector)";
        metadata.AdditionalInfo["ReferenceTranscriptRequired"] = true;
        metadata.AdditionalInfo["MinReferenceDurationSec"] = MinReferenceDuration;
        return metadata;
    }

    private void ValidateReference(Tensor<T> referenceAudio, string referenceText)
    {
        ValidateReferenceAudio(referenceAudio);
        if (string.IsNullOrWhiteSpace(referenceText))
            throw new ArgumentException("VALL-E X requires the transcript paired with the reference audio.", nameof(referenceText));
    }

    private void ValidateReferenceAudio(Tensor<T> referenceAudio)
    {
        Validation.Guard.NotNull(referenceAudio);
        int minimumSamples = checked((int)Math.Ceiling(MinReferenceDuration * _cloneOptions.SampleRate));
        if (referenceAudio.Length < minimumSamples)
            throw new ArgumentException(
                $"VALL-E X enrollment requires at least {MinReferenceDuration:0.###} seconds of reference audio "
                + $"({minimumSamples} samples at {_cloneOptions.SampleRate} Hz).",
                nameof(referenceAudio));
    }
}
