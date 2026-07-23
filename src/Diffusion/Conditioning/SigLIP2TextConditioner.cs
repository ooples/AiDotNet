using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// SigLIP 2 text encoder conditioning module (Tschannen et al. 2025).
/// Improves on SigLIP via additional captioning loss + MAP head; the
/// text-encoder body remains a standard CLIP-style stack.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features",
    "https://arxiv.org/abs/2502.14786",
    Year = 2025,
    Authors = "Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai")]
public class SigLIP2TextConditioner<T> : TextConditioningBase<T>
{
    private readonly SigLIP2Variant _variant;

    public override bool ProducesPooledOutput => true;

    public SigLIP2TextConditioner(
        ITokenizer tokenizer,
        SigLIP2Variant variant = SigLIP2Variant.Base,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer,
            maxSequenceLength: 64,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        Guard.NotNull(tokenizer);
        _variant = variant;
    }

    /// <summary>
    /// Loads a paper-canonical SigLIP 2 conditioner with its real
    /// pretrained tokenizer from HuggingFace.
    /// </summary>
    public static SigLIP2TextConditioner<T> FromPretrained(
        SigLIP2Variant variant = SigLIP2Variant.Base,
        string huggingFaceModelName = "google/siglip2-base-patch16-224",
        string? cacheDir = null)
    {
        var tokenizer = AutoTokenizer.FromPretrained(huggingFaceModelName, cacheDir);
        return new SigLIP2TextConditioner<T>(tokenizer, variant);
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultSigLIP2TextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new SigLIP2TextConditioner<T>(Tokenizer, _variant, Architecture);

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(SigLIP2Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 1);

    private static int GetEmbeddingDim(SigLIP2Variant variant) => variant switch
    {
        SigLIP2Variant.Base => 768,
        SigLIP2Variant.Large => 1024,
        SigLIP2Variant.So400M => 1152,
        _ => 768,
    };
    private static int GetHiddenSize(SigLIP2Variant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(SigLIP2Variant variant) => variant switch
    {
        SigLIP2Variant.Base => 12,
        SigLIP2Variant.Large => 24,
        SigLIP2Variant.So400M => 27,
        _ => 12,
    };
    private static int GetNumHeads(SigLIP2Variant variant) => variant switch
    {
        SigLIP2Variant.Base => 12,
        SigLIP2Variant.Large => 16,
        SigLIP2Variant.So400M => 16,
        _ => 12,
    };
}
