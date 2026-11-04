using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>
/// Voice-cloning facade over the released CosyVoice supervised-token, conditional-flow-matching, and HiFT pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>This class deliberately inherits <see cref="CosyVoice{T}"/>. Voice cloning is an inference task of the same
/// CosyVoice checkpoint, not a second neural architecture: CAM++ extracts the speaker x-vector, the S3 tokenizer extracts
/// prompt speech tokens, the conditional flow reads the prompt mel, and HiFT synthesizes the final waveform.</para>
/// <para><b>For Beginners:</b> Give the model a recording and it creates a reusable CosyVoice prompt internally before
/// synthesizing the requested text in that voice. All generation and training behavior comes from the real CosyVoice
/// implementation, so this convenience API cannot drift into a placeholder model.</para>
/// </remarks>
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
public partial class CosyVoiceClone<T> : CosyVoice<T>, IVoiceCloner<T>
{
    private readonly CosyVoiceCloneOptions _cloneOptions;

    /// <summary>Creates the facade in ONNX inference mode.</summary>
    public CosyVoiceClone(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        CosyVoiceCloneOptions? options = null)
        : base(architecture, modelPath, options ?? new CosyVoiceCloneOptions())
    {
        _cloneOptions = options ?? new CosyVoiceCloneOptions();
    }

    /// <summary>Creates the facade with CosyVoice's native paper architecture.</summary>
    public CosyVoiceClone(
        NeuralNetworkArchitecture<T> architecture,
        CosyVoiceCloneOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new CosyVoiceCloneOptions(), optimizer)
    {
        _cloneOptions = options ?? new CosyVoiceCloneOptions();
    }

    /// <inheritdoc />
    public double MinReferenceDuration => _cloneOptions.MinReferenceDurationSec;

    /// <inheritdoc />
    public int SpeakerEmbeddingDim => _cloneOptions.SpeakerEmbeddingDim;

    /// <inheritdoc />
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> referenceAudio)
    {
        ValidateReferenceAudio(referenceAudio);

        // IVoiceCloner has no transcript parameter. This maps to CosyVoice's released cross-lingual path:
        // the flow reads the prompt tokens/mel while the LM starts a new utterance.
        var previousVoice = Voice;
        try
        {
            Voice = CreateVoice(referenceAudio, _cloneOptions.SampleRate);
            return Synthesize(text);
        }
        finally
        {
            Voice = previousVoice;
        }
    }

    /// <inheritdoc />
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        ValidateReferenceAudio(referenceAudio);
        return SpeakerEmbedding(referenceAudio, _cloneOptions.SampleRate);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.Name = metadata.Name.Replace("CosyVoice", "CosyVoice-Clone");
        metadata.Description = "CosyVoice voice-cloning facade over the native released architecture (Du et al., 2024)";
        metadata.AdditionalInfo["Architecture"] = "CosyVoice";
        metadata.AdditionalInfo["TaskFacade"] = "VoiceCloning";
        metadata.AdditionalInfo["MinReferenceDurationSec"] = MinReferenceDuration;
        return metadata;
    }

    private void ValidateReferenceAudio(Tensor<T> referenceAudio)
    {
        if (referenceAudio is null)
            throw new ArgumentNullException(nameof(referenceAudio));

        int minimumSamples = checked((int)Math.Ceiling(MinReferenceDuration * _cloneOptions.SampleRate));
        if (referenceAudio.Length < minimumSamples)
        {
            throw new ArgumentException(
                $"CosyVoice voice cloning needs at least {MinReferenceDuration:0.###} seconds " +
                $"({minimumSamples} samples at {_cloneOptions.SampleRate} Hz) of reference audio.",
                nameof(referenceAudio));
        }
    }
}
