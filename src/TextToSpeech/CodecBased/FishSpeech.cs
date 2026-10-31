using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Fish Speech 1.4: the Dual-AR transformer and Firefly-GAN of <see cref="FishSpeechV15{T}"/> (Liao et al. 2024) in the
/// 1.4 release's format — a 32,000-token byte-level BPE vocabulary and one <c>&lt;|semantic|&gt;</c> token per frame
/// whose eight codes are stored plus one (zero pads text positions).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The release (fish-speech v1.4.3 and the <c>fishaudio/fish-speech-1.4</c> checkpoint) defines everything the
/// paper leaves open: the prompt is <c>&lt;|im_start|&gt;user\n{text}&lt;|im_end|&gt;&lt;|im_start|&gt;assistant\n</c> with no
/// system message, the training sample scores the token row throughout and the codes from the first frame, text is cleaned
/// more aggressively (brackets, dashes and newlines dropped) and, as the 1.4 API does by default, Chinese numbers are
/// written out first. Training follows the release's fine-tuning recipe by default.</para>
/// <para><b>For Beginners:</b> This is the earlier Fish Speech release. It speaks text in the voice of a short reference
/// recording, one frame of audio codes at a time.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var model = new FishSpeech&lt;float&gt;(architecture, new FishSpeechOptions());
/// model.LoadReferenceWeights("fish-speech-1.4");      // model.pth, the Firefly-GAN generator, tokenizer.json
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
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-5, WeightDecay = 0.0, MaxGradientNorm = 1.0,
    Provenance = RecipeProvenance.ReleasedCode,
    Source = "fish-speech v1.4.3 fish_speech/configs/text2semantic_finetune.yaml: AdamW at 1e-4, betas (0.9, 0.95), eps 1e-5, no weight "
             + "decay, LambdaLR constant after 10 warm-up steps, gradient norm clipped at 1.0, 1,000 steps.")]
[PaperOptimizer(OptimizerKind.AdamW, Variant = nameof(FishSpeechRecipe.Paper), LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-8,
    WeightDecay = 0.01,
    Source = "Appendix A: AdamW (β1 0.9, β2 0.98, ε 1e-8) at 5e-4, weight decay 0.01, 500K steps, cosine decay after 2K warm-up steps "
             + "to a final ratio of 0.1.")]
public class FishSpeech<T> : FishSpeechBase<T>
{
    /// <summary>Creates a native, trainable Fish Speech 1.4.</summary>
    public FishSpeech(NeuralNetworkArchitecture<T> architecture, FishSpeechOptions? options = null)
        : base(architecture, options ?? new FishSpeechOptions(), FishSpeechVersion.V14)
    {
    }

    /// <summary>Creates an ONNX-backed Fish Speech 1.4 for inference.</summary>
    public FishSpeech(NeuralNetworkArchitecture<T> architecture, string modelPath, FishSpeechOptions? options = null)
        : base(architecture, modelPath, options ?? new FishSpeechOptions(), FishSpeechVersion.V14)
    {
    }

    private protected override string ModelName => "FishSpeech";
}
