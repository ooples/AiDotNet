using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Fish Speech 1.5 (Liao et al. 2024, "Fish-Speech: Leveraging Large Language Models for Advanced Multilingual
/// Text-to-Speech Synthesis"): a Dual-AR transformer — a slow transformer over text and semantic tokens, a fast transformer
/// over each frame's eight codebooks — speaking Firefly-GAN's grouped-FSQ codes, which Firefly-GAN turns into 44.1 kHz
/// audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper leaves the sizes and the sequence format to the release (fish-speech v1.5.1 and the
/// <c>fishaudio/fish-speech-1.5</c> checkpoint), whose code fills the gaps: each frame is the token
/// <c>&lt;|semantic:c0|&gt;</c> with all eight codes' embeddings added, prompts open with the system message "Speak out the
/// provided text." when references are given, and the training sample is that conversation with the speech scored from
/// <c>&lt;|voice|&gt;</c> on. Training follows the paper's recipe by default; DPO, which the paper names as its last stage
/// without details, is standard DPO (<see cref="FishSpeechBase{T}.TrainPreference"/>), and the release's reference-free
/// variant is <see cref="FishSpeechBase{T}.TrainPreferenceReferenceFree"/>.</para>
/// <para><b>For Beginners:</b> Fish Speech reads text and writes speech one short frame at a time, like a language model
/// writing words. Give it a few seconds of someone speaking (and what they said) and it speaks new text in that voice.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var model = new FishSpeechV15&lt;float&gt;(architecture, new FishSpeechV15Options());
/// model.LoadReferenceWeights("fish-speech-1.5");      // model.pth, the Firefly-GAN generator, tokenizer.tiktoken
/// model.Voice = model.CreateVoice(reference, referenceRate, "What the reference says.");
/// var speech = model.Synthesize("Text to speak in that voice.");   // 44.1 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis", "https://arxiv.org/abs/2411.01156",
    Year = 2024, Authors = "Shijia Liao, Yuxuan Wang, Tianyu Li, Yifan Cheng, Ruoyi Zhang, Rongzhi Zhou, Yijin Xing")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-8, WeightDecay = 0.01,
    Source = "Appendix A: AdamW (β1 0.9, β2 0.98, ε 1e-8) at 5e-4, weight decay 0.01, 1M-token batches for 500K steps, cosine decay "
             + "after 2K warm-up steps to a final ratio of 0.1. The release's lit_module exempts the token embeddings and the norms from "
             + "decay; its gradient clipping at 1.0 is kept.")]
[PaperOptimizer(OptimizerKind.AdamW, Variant = nameof(FishSpeechRecipe.ReleaseFineTune), LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95,
    Epsilon = 1e-5, WeightDecay = 0.0, MaxGradientNorm = 1.0, Provenance = RecipeProvenance.ReleasedCode,
    Source = "fish-speech v1.5.1 fish_speech/configs/text2semantic_finetune.yaml: AdamW at 1e-4, betas (0.9, 0.95), eps 1e-5, no weight "
             + "decay, LambdaLR constant after 10 warm-up steps, gradient norm clipped at 1.0, 10,000 steps.")]
public class FishSpeechV15<T> : FishSpeechBase<T>
{
    /// <summary>Creates a native, trainable Fish Speech 1.5.</summary>
    public FishSpeechV15(NeuralNetworkArchitecture<T> architecture, FishSpeechV15Options? options = null)
        : base(architecture, options ?? new FishSpeechV15Options(), FishSpeechVersion.V15)
    {
    }

    /// <summary>Creates an ONNX-backed Fish Speech 1.5 for inference.</summary>
    public FishSpeechV15(NeuralNetworkArchitecture<T> architecture, string modelPath, FishSpeechV15Options? options = null)
        : base(architecture, modelPath, options ?? new FishSpeechV15Options(), FishSpeechVersion.V15)
    {
    }

    private protected override string ModelName => "FishSpeechV15";
}
