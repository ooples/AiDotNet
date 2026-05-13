using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// SigLIP text encoder conditioning module (Zhai et al., ICCV 2023).
/// Same encoder architecture as CLIP; the paper's contribution is the
/// sigmoid contrastive loss (vs CLIP's softmax), which is upstream of
/// this encoder body.
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
    "Sigmoid Loss for Language Image Pre-Training",
    "https://arxiv.org/abs/2303.15343",
    Year = 2023,
    Authors = "Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer")]
public class SigLIPTextConditioner<T> : TextConditioningBase<T>
{
    private readonly SigLIPVariant _variant;

    public override bool ProducesPooledOutput => true;

    public SigLIPTextConditioner(
        SigLIPVariant variant = SigLIPVariant.Base,
        ITokenizer? tokenizer = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer ?? ClipTokenizerFactory.CreateSimple(),
            maxSequenceLength: 64,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        _variant = variant;
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultSigLIPTextLayers(
            vocabSize: VocabSize,
            maxSeqLen: MaxSequenceLength,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant),
            projectionDim: EmbeddingDimension);

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new SigLIPTextConditioner<T>(_variant, Tokenizer, Architecture);

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(SigLIPVariant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Deep,
            inputSize: 64,
            outputSize: GetEmbeddingDim(variant));

    private static int GetEmbeddingDim(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 768,
        SigLIPVariant.Large => 1024,
        SigLIPVariant.So400M => 1152,
        _ => 768,
    };
    private static int GetHiddenSize(SigLIPVariant variant) => GetEmbeddingDim(variant);
    private static int GetNumLayers(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.Large => 24,
        SigLIPVariant.So400M => 27,
        _ => 12,
    };
    private static int GetNumHeads(SigLIPVariant variant) => variant switch
    {
        SigLIPVariant.Base => 12,
        SigLIPVariant.Large => 16,
        SigLIPVariant.So400M => 16,
        _ => 12,
    };
}
