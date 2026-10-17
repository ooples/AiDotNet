using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// VALL-E: a neural codec language model for zero-shot text-to-speech, which continues a 3-second recording of an unseen
/// speaker by predicting EnCodec codes from phonemes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (Wang et al., 2023). Microsoft
/// released no code; what the paper leaves open follows the reproduction lifeiteng/vall-e, against which this model is
/// tested (see <see cref="VALLEOptions"/>).
/// </para>
/// <para>
/// <b>Model</b> (§4): EnCodec turns 24 kHz speech into eight codebooks of codes at 75 frames a second. An
/// autoregressive Transformer predicts the first codebook from the phonemes and the codes before it; a
/// non-autoregressive Transformer, told the stage through adaptive layer norm, predicts each later codebook from the
/// phonemes, the codebooks below it and an acoustic prompt — in training a random 3-second segment of the same
/// utterance (§5.1). Each is trained on its own (<see cref="VallEModelBase{T}.CurrentStage"/>).
/// </para>
/// <para>
/// <b>Synthesis</b> (§4.3, "VALL-E"): the prompt's transcript precedes the text, the prompt's first-codebook codes
/// start the AR decoding (sampling until the end token, or 16 codes per phoneme token), the NAR fills the other
/// codebooks greedily after the prompt (its text without the prompt's transcript, as the reference does), and EnCodec
/// decodes the new frames. A voice without a transcript is an empty enrolled transcript.
/// </para>
/// <para>
/// <b>Training data</b>: <see cref="TtsTrainingSample{T}.Tokens"/> are the phoneme ids
/// (<see cref="VallEModelBase{T}.EncodePhonemes"/>), <see cref="TtsTrainingSample{T}.CodecTokens"/> the EnCodec codes
/// <c>[frames, 8]</c> (or <see cref="TtsTrainingSample{T}.Audio"/> at 24 kHz, which the model encodes). The paper crops
/// each utterance to a random 10–20 seconds together with its aligned phonemes; that needs the alignment, so it is the
/// caller's.
/// </para>
/// <para><b>For Beginners:</b> VALL-E treats speech as text-like tokens. Given a few seconds of someone's voice and a
/// sentence, it writes the tokens of that person saying the sentence, and a codec turns them into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var valle = new VALLE&lt;float&gt;(architecture, new VALLEOptions());
/// valle.Voice = valle.CreateVoice(promptAudio24kHz, "the prompt's transcript");
/// var audio = valle.Synthesize("Hello there.");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
    "https://arxiv.org/abs/2301.02111",
    Year = 2023,
    Authors = "Wang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "autoregressive", Provenance = RecipeProvenance.Stated,
    Source = "Section 5.1: AdamW, the learning rate warmed up over the first 32k updates to a peak of 5e-4, then "
             + "decayed linearly, for 800k steps. The paper names no betas or weight decay; these are PyTorch's AdamW "
             + "defaults.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 32_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "non-autoregressive", Provenance = RecipeProvenance.Stated,
    Source = "Section 5.1: AdamW, the learning rate warmed up over the first 32k updates to a peak of 5e-4, then "
             + "decayed linearly, for 800k steps. The paper names no betas or weight decay; these are PyTorch's AdamW "
             + "defaults.")]
public partial class VALLE<T> : VallEModelBase<T>
{
    /// <summary>Creates an ONNX-backed VALL-E for inference.</summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLEOptions? options = null)
        : base(architecture, modelPath, options ?? new VALLEOptions())
    {
    }

    /// <summary>Creates a native, trainable VALL-E.</summary>
    public VALLE(NeuralNetworkArchitecture<T> architecture, VALLEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VALLEOptions(), optimizer)
    {
    }

    /// <inheritdoc />
    public override ModelOptions GetOptions() => Settings;

    /// <inheritdoc />
    /// <remarks>The espeak table of lifeiteng/vall-e's LibriTTS recipe (<see cref="LibriTtsPhonemeTable"/>).</remarks>
    protected override IReadOnlyList<string> PhonemeTable => LibriTtsPhonemeTable.Symbols;

    /// <inheritdoc />
    protected override IReadOnlyList<string> PhonemizeText(string text) => EnglishG2P.Default.Phonemize(text);

    /// <inheritdoc />
    /// <remarks>The paper's training prompt (§5.1): a random 3-second segment of the same utterance, at most a quarter
    /// of it (the reference's <c>prefix_mode 2</c>).</remarks>
    protected override Func<Tensor<T>, Tensor<T>, Tensor<T>> NonAutoRegressiveObjective(TtsTrainingSample<T> sample,
        int[] text, int[,] codes, bool training, Random random)
    {
        int promptFrames = (int)Math.Round(Settings.PromptSeconds * CodecFrameRate);
        return (_, _) => PaperCore.NarLoss(text, codes, promptFrames, training, random);
    }

    /// <inheritdoc />
    /// <remarks>The AR model reads the prompt's transcript and the text; the NAR reads <c>&lt;bos&gt;</c> and the text
    /// from the separator before it (the reference's <c>prefix_mode 2</c>, without the enrolled phonemes).</remarks>
    protected override int[,] Generate(int[] target, int[] enrolled, int[,] prompt, TtsVoice<T> voice, Random random)
    {
        var core = PaperCore;
        var full = JoinPrompt(enrolled, target);
        var first = core.ArGenerate(full, FirstCodebook(prompt), Settings.Temperature, Settings.TopK,
            Settings.MaxCodesPerTextToken * full.Length + 1, random);
        int enrolledLength = enrolled.Length + 2;
        var narText = enrolled.Length > 0 ? new[] { BeginToken }.Concat(full.Skip(enrolledLength - 1)).ToArray() : full;
        return core.NarGenerate(narText, prompt, first.ToArray(), random);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = IsNative ? "VALL-E-Native" : "VALL-E-ONNX",
            Description = "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (Wang et al., 2023)",
            FeatureCount = Settings.HiddenDim,
            Complexity = Settings.NumEncoderLayers + Settings.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "VALL-E";
        metadata.AdditionalInfo["SampleRate"] = Settings.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
