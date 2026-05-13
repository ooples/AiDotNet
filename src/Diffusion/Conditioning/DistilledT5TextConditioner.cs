using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Distilled T5 text encoder conditioning module — same architecture as T5
/// but half the layer count, per the DistilBERT-style knowledge-distillation
/// recipe (Sanh et al., 2019).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
[ModelDomain(ModelDomain.NaturalLanguageProcessing)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Embedding)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter",
    "https://arxiv.org/abs/1910.01108",
    Year = 2019,
    Authors = "Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf")]
public class DistilledT5TextConditioner<T> : TextConditioningBase<T>
{
    private readonly DistilledT5Variant _variant;

    public override bool ProducesPooledOutput => false;

    public DistilledT5TextConditioner(
        DistilledT5Variant variant = DistilledT5Variant.Base,
        ITokenizer? tokenizer = null,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(
            architecture: architecture ?? BuildDefaultArchitecture(variant),
            tokenizer: tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5),
            maxSequenceLength: 512,
            embeddingDimension: GetEmbeddingDim(variant))
    {
        _variant = variant;
    }

    protected override IEnumerable<ILayer<T>> CreateDefaultLayers() =>
        LayerHelper<T>.CreateDefaultDistilledT5TextLayers(
            vocabSize: VocabSize,
            hiddenSize: GetHiddenSize(_variant),
            numLayers: GetNumLayers(_variant),
            numHeads: GetNumHeads(_variant));

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new DistilledT5TextConditioner<T>(_variant, Tokenizer, Architecture);

    private static NeuralNetworkArchitecture<T> BuildDefaultArchitecture(DistilledT5Variant variant) =>
        new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Custom,
            complexity: NetworkComplexity.Medium,
            inputSize: 512,
            outputSize: GetEmbeddingDim(variant));

    private static int GetEmbeddingDim(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 512,
        DistilledT5Variant.Base => 768,
        DistilledT5Variant.Large => 1024,
        _ => 768,
    };
    private static int GetHiddenSize(DistilledT5Variant variant) => GetEmbeddingDim(variant);
    // Half the layers vs full T5.
    private static int GetNumLayers(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 3,
        DistilledT5Variant.Base => 6,
        DistilledT5Variant.Large => 12,
        _ => 6,
    };
    private static int GetNumHeads(DistilledT5Variant variant) => variant switch
    {
        DistilledT5Variant.Small => 8,
        DistilledT5Variant.Base => 12,
        DistilledT5Variant.Large => 16,
        _ => 12,
    };
}
